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


# kernel path: /tmp/torchinductor_sahanp/2q/c2q7p7npaq253tnmyivd4uti6mpvtrweyqscra455u2rxcnywgoh.py
# Topologically Sorted Source Nodes: [x_262], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_262 => convolution_76
# Graph fragment:
#   %convolution_76 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (150528*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tv/ctv4qz7kzvk6dpru4j5y6b4sy22cla3n4ljoinozh66iauqntjbm.py
# Topologically Sorted Source Nodes: [x_262], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_262 => convolution_76
# Graph fragment:
#   %convolution_76 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (147*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tr/ctrd2mbfbbn4cggiwk6os4yp5qeaw4s5l3mgjbm56o32s3477ilv.py
# Topologically Sorted Source Nodes: [group_norm_72], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_72 => var_mean_73
# Graph fragment:
#   %var_mean_73 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_217, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_2 = async_compile.triton('triton_per_fused_native_group_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 18944
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 37
    x2 = (xindex // 2368)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 8137, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (8137*x1)
    tmp4 = tl.full([1, 1], 301056, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((96*((r3 + (128*x0) + (8137*x1)) % 3136)) + (301056*x2) + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = 0.0
    tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tmp19 = 1.0
    tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
    tmp23 = tl.where(tmp2, tmp21, tmp22)
    tmp24 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp25 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp28 = tl.where(xmask, tmp24, 0)
    tmp29 = tl.where(xmask, tmp25, 0)
    tmp30 = tl.where(xmask, tmp26, 0)
    tmp31, tmp32, tmp33 = triton_helpers.welford(tmp28, tmp29, tmp30, 1)
    tmp34 = tmp31[:, None]
    tmp35 = tmp32[:, None]
    tmp36 = tmp33[:, None]
    tl.store(out_ptr0 + (x5), tmp34, xmask)
    tl.store(out_ptr1 + (x5), tmp35, xmask)
    tl.store(out_ptr2 + (x5), tmp36, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ly/clyicxdbg7fu36m5c33tqirovrckf2cw4vxh6r6sici72rytmrg3.py
# Topologically Sorted Source Nodes: [group_norm_72], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_72 => var_mean_73
# Graph fragment:
#   %var_mean_73 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_217, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_3 = async_compile.triton('triton_per_fused_native_group_norm_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 296
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (64*x0)), xmask, other=0.0)
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
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/be/cbenw325m3wkuczsrvluu3wopccz4wzx62ktvoslbbvnzq5kr6nh.py
# Topologically Sorted Source Nodes: [group_norm_72], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_72 => var_mean_73
# Graph fragment:
#   %var_mean_73 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_217, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_4 = async_compile.triton('triton_per_fused_native_group_norm_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 37
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (37*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (37*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (37*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/7i/c7icrsy25actwqyzu6uzvlogal5vtxddrqz47si5yltvdjlnwvy6.py
# Topologically Sorted Source Nodes: [group_norm_72], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_72 => add_255, mul_327
# Graph fragment:
#   %mul_327 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_218, %unsqueeze_437), kwargs = {})
#   %add_255 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_327, %unsqueeze_434), kwargs = {})
triton_poi_fused_native_group_norm_5 = async_compile.triton('triton_poi_fused_native_group_norm_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 96
    x2 = (xindex // 301056)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 301056.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5s/c5s2lguduwupdjkkf3pb3qc66leazsvomtcytohyasm55kuej4xd.py
# Topologically Sorted Source Nodes: [x_262, y_36, sub_36, mul_72, x_263], Original ATen: [aten.convolution, aten.avg_pool2d, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_72 => mul_328
#   sub_36 => sub_110
#   x_262 => convolution_76
#   x_263 => add_256
#   y_36 => avg_pool2d_36
# Graph fragment:
#   %convolution_76 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %avg_pool2d_36 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_255, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
#   %sub_110 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_36, %add_255), kwargs = {})
#   %mul_328 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_110, %view_219), kwargs = {})
#   %add_256 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_76, %mul_328), kwargs = {})
triton_poi_fused_add_avg_pool2d_convolution_mul_sub_6 = async_compile.triton('triton_poi_fused_add_avg_pool2d_convolution_mul_sub_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_mul_sub_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_mul_sub_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 56) % 56
    y0 = yindex % 56
    x3 = xindex
    y6 = yindex
    y2 = (yindex // 3136)
    y4 = yindex % 3136
    tmp54 = tl.load(in_ptr1 + (x3 + (96*y6)), xmask & ymask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr0 + (x3 + (96*y6)), xmask & ymask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + y0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5472) + x3 + (96*y6)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-5376) + x3 + (96*y6)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + y0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-5280) + x3 + (96*y6)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-96) + x3 + (96*y6)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x3 + (96*y6)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (96 + x3 + (96*y6)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + y1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (5280 + x3 + (96*y6)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (5376 + x3 + (96*y6)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5472 + x3 + (96*y6)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = (((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))*((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) + (((56) * ((56) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (56)))*((56) * ((56) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (56)))) + ((-1)*((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))*((56) * ((56) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (56)))) + ((-1)*((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))*((56) * ((56) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (56))))
    tmp53 = tmp51 / tmp52
    tmp56 = tmp54 + tmp55
    tmp58 = tmp53 - tmp57
    tmp60 = tmp58 * tmp59
    tmp61 = tmp56 + tmp60
    tl.store(out_ptr1 + (y4 + (3136*x3) + (301056*y2)), tmp61, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hw/chwpjt4yye2jxg2ab3cxy7pooa3gtqfijacndgvfjg5rshvbrjiq.py
# Topologically Sorted Source Nodes: [group_norm_73], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_73 => var_mean_74
# Graph fragment:
#   %var_mean_74 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_220, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_7 = async_compile.triton('triton_red_fused_native_group_norm_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_7(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 296
    rnumel = 8137
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 37
    x1 = (xindex // 37)
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (8137*x0)
        tmp1 = tl.full([1, 1], 301056, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((301056*x1) + ((r2 + (8137*x0)) % 301056)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = 1.0
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp11 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_combine(
            tmp13_mean, tmp13_m2, tmp13_weight,
            tmp10, tmp11, tmp12
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ri/crimrqqj2564cbnnd6o3yrj6oauft23no7hnpixct3tvaovw763e.py
# Topologically Sorted Source Nodes: [group_norm_73], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_73 => add_258, mul_330
# Graph fragment:
#   %mul_330 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_221, %unsqueeze_443), kwargs = {})
#   %add_258 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_330, %unsqueeze_440), kwargs = {})
triton_poi_fused_native_group_norm_8 = async_compile.triton('triton_poi_fused_native_group_norm_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 96)
    y0 = yindex % 96
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 301056.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bj/cbj5nn7urbesgssv5ag3j5aunl3h6hmvzet3j6rgfwhxa7ymmoce.py
# Topologically Sorted Source Nodes: [group_norm_73, x_264, x_265], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
# Source node to ATen node mapping:
#   group_norm_73 => add_258, mul_330
#   x_264 => convolution_77
#   x_265 => add_259, erf_36, mul_331, mul_332, mul_333
# Graph fragment:
#   %mul_330 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_221, %unsqueeze_443), kwargs = {})
#   %add_258 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_330, %unsqueeze_440), kwargs = {})
#   %convolution_77 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_258, %arg8_1, %arg9_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_77, 0.5), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_77, 0.7071067811865476), kwargs = {})
#   %erf_36 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_332,), kwargs = {})
#   %add_259 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_36, 1), kwargs = {})
#   %mul_333 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_331, %add_259), kwargs = {})
triton_poi_fused_convolution_gelu_native_group_norm_9 = async_compile.triton('triton_poi_fused_convolution_gelu_native_group_norm_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_native_group_norm_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_native_group_norm_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9633792
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


# kernel path: /tmp/torchinductor_sahanp/pu/cpuiwdlc5cpqg6z3kuz3yevgo6p4gmo5bpneozxrsct6r2gou36r.py
# Topologically Sorted Source Nodes: [group_norm_74], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_74 => var_mean_75
# Graph fragment:
#   %var_mean_75 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_223, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_10 = async_compile.triton('triton_per_fused_native_group_norm_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 18944
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 37
    x2 = (xindex // 2368)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 8137, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (8137*x1)
    tmp4 = tl.full([1, 1], 301056, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((301056*x2) + ((r3 + (128*x0) + (8137*x1)) % 301056)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((96*((r3 + (128*x0) + (8137*x1)) % 3136)) + (301056*x2) + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x0) + (8137*x1)) // 3136) % 96), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(xmask, tmp28, 0)
    tmp33 = tl.where(xmask, tmp29, 0)
    tmp34 = tl.where(xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x5), tmp38, xmask)
    tl.store(out_ptr1 + (x5), tmp39, xmask)
    tl.store(out_ptr2 + (x5), tmp40, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/f5/cf5v7pqju6auyf2llgkioag2yxvxbkbgjddkt26cvrzrn3lazpwk.py
# Topologically Sorted Source Nodes: [group_norm_74], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_74 => add_262, mul_336
# Graph fragment:
#   %mul_336 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_224, %unsqueeze_449), kwargs = {})
#   %add_262 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_336, %unsqueeze_446), kwargs = {})
triton_poi_fused_native_group_norm_11 = async_compile.triton('triton_poi_fused_native_group_norm_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y1), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 301056.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ro/crogkmlapor5rufixtah647mqwgymxlnh4eae3fl7n5uqeig3sqs.py
# Topologically Sorted Source Nodes: [group_norm_73, x_264, x_265, x_267, mul_73, x_269, y_37, sub_37, mul_74, x_270], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
# Source node to ATen node mapping:
#   group_norm_73 => add_258, mul_330
#   mul_73 => mul_334
#   mul_74 => mul_337
#   sub_37 => sub_113
#   x_264 => convolution_77
#   x_265 => add_259, erf_36, mul_331, mul_332, mul_333
#   x_267 => convolution_78
#   x_269 => add_260
#   x_270 => add_263
#   y_37 => avg_pool2d_37
# Graph fragment:
#   %mul_330 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_221, %unsqueeze_443), kwargs = {})
#   %add_258 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_330, %unsqueeze_440), kwargs = {})
#   %convolution_77 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_258, %arg8_1, %arg9_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_77, 0.5), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_77, 0.7071067811865476), kwargs = {})
#   %erf_36 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_332,), kwargs = {})
#   %add_259 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_36, 1), kwargs = {})
#   %mul_333 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_331, %add_259), kwargs = {})
#   %convolution_78 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_333, %arg10_1, %arg11_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_78, %view_222), kwargs = {})
#   %add_260 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_256, %mul_334), kwargs = {})
#   %avg_pool2d_37 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_262, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_37, %add_262), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %view_225), kwargs = {})
#   %add_263 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_260, %mul_337), kwargs = {})
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_12 = async_compile.triton('triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 56)
    x1 = xindex % 56
    x5 = xindex
    y0 = yindex
    y3 = yindex % 96
    y4 = (yindex // 96)
    tmp54 = tl.load(in_out_ptr0 + (x5 + (3136*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr1 + (y3 + (96*x5) + (301056*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr0 + (x5 + (3136*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp0 = (-1) + x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-57) + x5 + (3136*y0)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-56) + x5 + (3136*y0)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-55) + x5 + (3136*y0)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x5 + (3136*y0)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x5 + (3136*y0)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x5 + (3136*y0)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (55 + x5 + (3136*y0)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (56 + x5 + (3136*y0)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (57 + x5 + (3136*y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((0) * ((0) >= ((-1) + x2)) + ((-1) + x2) * (((-1) + x2) > (0)))) + (((56) * ((56) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (56)))*((56) * ((56) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (56)))) + ((-1)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((56) * ((56) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (56)))) + ((-1)*((0) * ((0) >= ((-1) + x2)) + ((-1) + x2) * (((-1) + x2) > (0)))*((56) * ((56) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (56))))
    tmp53 = tmp51 / tmp52
    tmp57 = tmp55 + tmp56
    tmp59 = tmp57 * tmp58
    tmp60 = tmp54 + tmp59
    tmp62 = tmp53 - tmp61
    tmp64 = tmp62 * tmp63
    tmp65 = tmp60 + tmp64
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (3136*y0)), tmp65, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nd/cndlaudesmzhu5fkobfy335cntapej7git6g4e53mbqe7dgrt2qh.py
# Topologically Sorted Source Nodes: [group_norm_83, x_299, x_300, x_302, mul_83, x_304], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
# Source node to ATen node mapping:
#   group_norm_83 => add_293, mul_375
#   mul_83 => mul_379
#   x_299 => convolution_87
#   x_300 => add_294, erf_41, mul_376, mul_377, mul_378
#   x_302 => convolution_88
#   x_304 => add_295
# Graph fragment:
#   %mul_375 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_251, %unsqueeze_503), kwargs = {})
#   %add_293 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_375, %unsqueeze_500), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_293, %arg58_1, %arg59_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_376 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.5), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.7071067811865476), kwargs = {})
#   %erf_41 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_377,), kwargs = {})
#   %add_294 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_41, 1), kwargs = {})
#   %mul_378 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_376, %add_294), kwargs = {})
#   %convolution_88 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_378, %arg60_1, %arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, %view_252), kwargs = {})
#   %add_295 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_291, %mul_379), kwargs = {})
triton_poi_fused_add_convolution_gelu_mul_native_group_norm_13 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mul_native_group_norm_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_native_group_norm_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mul_native_group_norm_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (96*y3)), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g4/cg4h6hs23dphikil34tkmxgmxtkzh6wygsjbpswjl2bzcolmlc7a.py
# Topologically Sorted Source Nodes: [group_norm_83, x_299, x_300, x_302, mul_83, x_304, x_305], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
# Source node to ATen node mapping:
#   group_norm_83 => add_293, mul_375
#   mul_83 => mul_379
#   x_299 => convolution_87
#   x_300 => add_294, erf_41, mul_376, mul_377, mul_378
#   x_302 => convolution_88
#   x_304 => add_295
#   x_305 => convolution_89
# Graph fragment:
#   %mul_375 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_251, %unsqueeze_503), kwargs = {})
#   %add_293 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_375, %unsqueeze_500), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_293, %arg58_1, %arg59_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_376 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.5), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.7071067811865476), kwargs = {})
#   %erf_41 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_377,), kwargs = {})
#   %add_294 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_41, 1), kwargs = {})
#   %mul_378 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_376, %add_294), kwargs = {})
#   %convolution_88 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_378, %arg60_1, %arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, %view_252), kwargs = {})
#   %add_295 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_291, %mul_379), kwargs = {})
#   %convolution_89 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_295, %arg63_1, %arg64_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_gelu_mul_native_group_norm_14 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mul_native_group_norm_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_native_group_norm_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mul_native_group_norm_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (864*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gx/cgxbij3at6hmt5kf7dpfuccgy4xj7jrhro7fv4a5mppqquyghyio.py
# Topologically Sorted Source Nodes: [group_norm_84], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_84 => var_mean_85
# Graph fragment:
#   %var_mean_85 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_253, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_15 = async_compile.triton('triton_per_fused_native_group_norm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_15(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9424
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x1 = (xindex // 19) % 62
    x0 = xindex % 19
    x2 = (xindex // 1178)
    x4 = xindex % 1178
    tmp0 = r3 + (128*x1)
    tmp1 = tl.full([1, 1], 7923, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x1) + (7923*x0)
    tmp4 = tl.full([1, 1], 150528, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((192*((r3 + (128*x1) + (7923*x0)) % 784)) + (150528*x2) + (((r3 + (128*x1) + (7923*x0)) // 784) % 192)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + (((r3 + (128*x1) + (7923*x0)) // 784) % 192), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = 0.0
    tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tmp19 = 1.0
    tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
    tmp23 = tl.where(tmp2, tmp21, tmp22)
    tmp24 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp25 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp28 = tl.where(xmask, tmp24, 0)
    tmp29 = tl.where(xmask, tmp25, 0)
    tmp30 = tl.where(xmask, tmp26, 0)
    tmp31, tmp32, tmp33 = triton_helpers.welford(tmp28, tmp29, tmp30, 1)
    tmp34 = tmp31[:, None]
    tmp35 = tmp32[:, None]
    tmp36 = tmp33[:, None]
    tl.store(out_ptr0 + (x4 + (1184*x2)), tmp34, xmask)
    tl.store(out_ptr1 + (x4 + (1184*x2)), tmp35, xmask)
    tl.store(out_ptr2 + (x4 + (1184*x2)), tmp36, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ve/cve3cijp5itqgfxfzwlr7zoanhvqdtgjhtulroxwilhcaqxwq4ah.py
# Topologically Sorted Source Nodes: [group_norm_84], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_84 => var_mean_85
# Graph fragment:
#   %var_mean_85 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_253, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_16 = async_compile.triton('triton_per_fused_native_group_norm_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 62
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 19
    x1 = (xindex // 19)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (19*r2) + (1184*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (19*r2) + (1184*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (19*r2) + (1184*x1)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wj/cwjh7sixav3uqyivj6nfabfeatslx2yqbxbm7esjkd63s5em6ra2.py
# Topologically Sorted Source Nodes: [group_norm_84], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_84 => var_mean_85
# Graph fragment:
#   %var_mean_85 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_253, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_17 = async_compile.triton('triton_per_fused_native_group_norm_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_17(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (19*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (19*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (19*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/gd/cgd33mipumdn5dcu7jdbx5rvkaonbw5relsz4k5arj66ys74fo6g.py
# Topologically Sorted Source Nodes: [group_norm_84], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_84 => add_297, mul_381
# Graph fragment:
#   %mul_381 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_254, %unsqueeze_509), kwargs = {})
#   %add_297 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_381, %unsqueeze_506), kwargs = {})
triton_poi_fused_native_group_norm_18 = async_compile.triton('triton_poi_fused_native_group_norm_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 192
    x2 = (xindex // 150528)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 150528.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ew/cew7t3tf7w2f4rcpdzfejt3udq676t4xbkzyur4jpcv3ynahy3ab.py
# Topologically Sorted Source Nodes: [group_norm_83, x_299, x_300, x_302, mul_83, x_304, x_305, y_42, sub_42, mul_84, x_306], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
# Source node to ATen node mapping:
#   group_norm_83 => add_293, mul_375
#   mul_83 => mul_379
#   mul_84 => mul_382
#   sub_42 => sub_128
#   x_299 => convolution_87
#   x_300 => add_294, erf_41, mul_376, mul_377, mul_378
#   x_302 => convolution_88
#   x_304 => add_295
#   x_305 => convolution_89
#   x_306 => add_298
#   y_42 => avg_pool2d_42
# Graph fragment:
#   %mul_375 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_251, %unsqueeze_503), kwargs = {})
#   %add_293 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_375, %unsqueeze_500), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_293, %arg58_1, %arg59_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_376 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.5), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.7071067811865476), kwargs = {})
#   %erf_41 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_377,), kwargs = {})
#   %add_294 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_41, 1), kwargs = {})
#   %mul_378 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_376, %add_294), kwargs = {})
#   %convolution_88 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_378, %arg60_1, %arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, %view_252), kwargs = {})
#   %add_295 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_291, %mul_379), kwargs = {})
#   %convolution_89 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_295, %arg63_1, %arg64_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %avg_pool2d_42 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_297, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
#   %sub_128 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_42, %add_297), kwargs = {})
#   %mul_382 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_128, %view_255), kwargs = {})
#   %add_298 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_89, %mul_382), kwargs = {})
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_19 = async_compile.triton('triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 28) % 28
    y0 = yindex % 28
    x3 = xindex
    y6 = yindex
    y2 = (yindex // 784)
    y4 = yindex % 784
    tmp54 = tl.load(in_ptr1 + (x3 + (192*y6)), xmask & ymask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr0 + (x3 + (192*y6)), xmask & ymask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + y0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5568) + x3 + (192*y6)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-5376) + x3 + (192*y6)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + y0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-5184) + x3 + (192*y6)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-192) + x3 + (192*y6)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x3 + (192*y6)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (192 + x3 + (192*y6)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + y1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (5184 + x3 + (192*y6)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (5376 + x3 + (192*y6)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5568 + x3 + (192*y6)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = (((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))*((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) + (((28) * ((28) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (28)))*((28) * ((28) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (28)))) + ((-1)*((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))*((28) * ((28) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (28)))) + ((-1)*((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))*((28) * ((28) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (28))))
    tmp53 = tmp51 / tmp52
    tmp56 = tmp54 + tmp55
    tmp58 = tmp53 - tmp57
    tmp60 = tmp58 * tmp59
    tmp61 = tmp56 + tmp60
    tl.store(out_ptr1 + (y4 + (784*x3) + (150528*y2)), tmp61, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2u/c2uhvmhjq5cadptshh4c624k5zpf6gesc5akmeqorpt6h3ler37m.py
# Topologically Sorted Source Nodes: [group_norm_85], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_85 => var_mean_86
# Graph fragment:
#   %var_mean_86 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_256, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_20 = async_compile.triton('triton_red_fused_native_group_norm_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_20(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 7923
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7923*x0)
        tmp1 = tl.full([1, 1], 150528, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((150528*x1) + ((r2 + (7923*x0)) % 150528)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = 1.0
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp11 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_combine(
            tmp13_mean, tmp13_m2, tmp13_weight,
            tmp10, tmp11, tmp12
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bw/cbwu3umvawnaorm6libdi7qtqg5g27xsyczdnammcjh57dik6yyb.py
# Topologically Sorted Source Nodes: [group_norm_85], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_85 => add_300, mul_384
# Graph fragment:
#   %mul_384 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_257, %unsqueeze_515), kwargs = {})
#   %add_300 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_384, %unsqueeze_512), kwargs = {})
triton_poi_fused_native_group_norm_21 = async_compile.triton('triton_poi_fused_native_group_norm_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 192)
    y0 = yindex % 192
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 150528.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nc/cncd27khybeatgxdqi62l5fbv7b6qmbue67zbtd4v6znrbara547.py
# Topologically Sorted Source Nodes: [group_norm_85, x_307, x_308], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
# Source node to ATen node mapping:
#   group_norm_85 => add_300, mul_384
#   x_307 => convolution_90
#   x_308 => add_301, erf_42, mul_385, mul_386, mul_387
# Graph fragment:
#   %mul_384 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_257, %unsqueeze_515), kwargs = {})
#   %add_300 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_384, %unsqueeze_512), kwargs = {})
#   %convolution_90 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_300, %arg70_1, %arg71_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_90, 0.5), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_90, 0.7071067811865476), kwargs = {})
#   %erf_42 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_386,), kwargs = {})
#   %add_301 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_42, 1), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_385, %add_301), kwargs = {})
triton_poi_fused_convolution_gelu_native_group_norm_22 = async_compile.triton('triton_poi_fused_convolution_gelu_native_group_norm_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_native_group_norm_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_native_group_norm_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 768
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


# kernel path: /tmp/torchinductor_sahanp/jy/cjygu56fwuinb2xwcfl3dzj6nngi5lipvxvhgvniviy54d3kv7q2.py
# Topologically Sorted Source Nodes: [group_norm_86], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_86 => var_mean_87
# Graph fragment:
#   %var_mean_87 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_259, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_23 = async_compile.triton('triton_per_fused_native_group_norm_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9424
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = xindex % 62
    x1 = (xindex // 62) % 19
    x2 = (xindex // 1178)
    x5 = xindex % 1178
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 7923, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (7923*x1)
    tmp4 = tl.full([1, 1], 150528, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((150528*x2) + ((r3 + (128*x0) + (7923*x1)) % 150528)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((192*((r3 + (128*x0) + (7923*x1)) % 784)) + (150528*x2) + (((r3 + (128*x0) + (7923*x1)) // 784) % 192)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + (((r3 + (128*x0) + (7923*x1)) // 784) % 192), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x0) + (7923*x1)) // 784) % 192), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(xmask, tmp28, 0)
    tmp33 = tl.where(xmask, tmp29, 0)
    tmp34 = tl.where(xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x5 + (1184*x2)), tmp38, xmask)
    tl.store(out_ptr1 + (x5 + (1184*x2)), tmp39, xmask)
    tl.store(out_ptr2 + (x5 + (1184*x2)), tmp40, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/x6/cx63egm6zjdgbrnvmlql6znppopgd6jgp4ol5ncz3kcdw4ni464l.py
# Topologically Sorted Source Nodes: [group_norm_86], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_86 => var_mean_87
# Graph fragment:
#   %var_mean_87 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_259, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_24 = async_compile.triton('triton_per_fused_native_group_norm_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 62
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 19
    x1 = (xindex // 19)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (62*x0) + (1184*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (62*x0) + (1184*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2 + (62*x0) + (1184*x1)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kx/ckx4lwwblvxam6rgjb6qle3eft4udte2dvwdttakwbginfhikgfh.py
# Topologically Sorted Source Nodes: [group_norm_86], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_86 => add_304, mul_390
# Graph fragment:
#   %mul_390 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_260, %unsqueeze_521), kwargs = {})
#   %add_304 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_390, %unsqueeze_518), kwargs = {})
triton_poi_fused_native_group_norm_25 = async_compile.triton('triton_poi_fused_native_group_norm_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y1), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 150528.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/l7/cl7bavi3cap5x7seqo7ewunwcbdqwohlosbctg4bxn3pqt52nngd.py
# Topologically Sorted Source Nodes: [group_norm_85, x_307, x_308, x_310, mul_85, x_312, y_43, sub_43, mul_86, x_313], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
# Source node to ATen node mapping:
#   group_norm_85 => add_300, mul_384
#   mul_85 => mul_388
#   mul_86 => mul_391
#   sub_43 => sub_131
#   x_307 => convolution_90
#   x_308 => add_301, erf_42, mul_385, mul_386, mul_387
#   x_310 => convolution_91
#   x_312 => add_302
#   x_313 => add_305
#   y_43 => avg_pool2d_43
# Graph fragment:
#   %mul_384 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_257, %unsqueeze_515), kwargs = {})
#   %add_300 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_384, %unsqueeze_512), kwargs = {})
#   %convolution_90 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_300, %arg70_1, %arg71_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_90, 0.5), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_90, 0.7071067811865476), kwargs = {})
#   %erf_42 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_386,), kwargs = {})
#   %add_301 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_42, 1), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_385, %add_301), kwargs = {})
#   %convolution_91 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_387, %arg72_1, %arg73_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_91, %view_258), kwargs = {})
#   %add_302 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_298, %mul_388), kwargs = {})
#   %avg_pool2d_43 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_304, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
#   %sub_131 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_43, %add_304), kwargs = {})
#   %mul_391 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_131, %view_261), kwargs = {})
#   %add_305 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_302, %mul_391), kwargs = {})
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_26 = async_compile.triton('triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 28)
    x1 = xindex % 28
    x5 = xindex
    y0 = yindex
    y3 = yindex % 192
    y4 = (yindex // 192)
    tmp54 = tl.load(in_out_ptr0 + (x5 + (784*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr1 + (y3 + (192*x5) + (150528*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr0 + (x5 + (784*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp0 = (-1) + x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-29) + x5 + (784*y0)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-28) + x5 + (784*y0)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-27) + x5 + (784*y0)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x5 + (784*y0)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x5 + (784*y0)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x5 + (784*y0)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (27 + x5 + (784*y0)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (28 + x5 + (784*y0)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (29 + x5 + (784*y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((0) * ((0) >= ((-1) + x2)) + ((-1) + x2) * (((-1) + x2) > (0)))) + (((28) * ((28) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (28)))*((28) * ((28) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (28)))) + ((-1)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((28) * ((28) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (28)))) + ((-1)*((0) * ((0) >= ((-1) + x2)) + ((-1) + x2) * (((-1) + x2) > (0)))*((28) * ((28) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (28))))
    tmp53 = tmp51 / tmp52
    tmp57 = tmp55 + tmp56
    tmp59 = tmp57 * tmp58
    tmp60 = tmp54 + tmp59
    tmp62 = tmp53 - tmp61
    tmp64 = tmp62 * tmp63
    tmp65 = tmp60 + tmp64
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (784*y0)), tmp65, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3f/c3fjwjt6ey6jnhk3nnuoshqurbucdmgtnxjgpnvhrxvgtxzrzk6n.py
# Topologically Sorted Source Nodes: [group_norm_95, x_342, x_343, x_345, mul_95, x_347], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
# Source node to ATen node mapping:
#   group_norm_95 => add_335, mul_429
#   mul_95 => mul_433
#   x_342 => convolution_100
#   x_343 => add_336, erf_47, mul_430, mul_431, mul_432
#   x_345 => convolution_101
#   x_347 => add_337
# Graph fragment:
#   %mul_429 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_287, %unsqueeze_575), kwargs = {})
#   %add_335 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_429, %unsqueeze_572), kwargs = {})
#   %convolution_100 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_335, %arg120_1, %arg121_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_100, 0.5), kwargs = {})
#   %mul_431 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_100, 0.7071067811865476), kwargs = {})
#   %erf_47 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_431,), kwargs = {})
#   %add_336 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_47, 1), kwargs = {})
#   %mul_432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_430, %add_336), kwargs = {})
#   %convolution_101 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_432, %arg122_1, %arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_101, %view_288), kwargs = {})
#   %add_337 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_333, %mul_433), kwargs = {})
triton_poi_fused_add_convolution_gelu_mul_native_group_norm_27 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mul_native_group_norm_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_native_group_norm_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mul_native_group_norm_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (192*y3)), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7r/c7ryxrjccpupkc7dldujlwljuq6s7dvkwdldsw242h3mnnrddfkz.py
# Topologically Sorted Source Nodes: [group_norm_95, x_342, x_343, x_345, mul_95, x_347, x_348], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
# Source node to ATen node mapping:
#   group_norm_95 => add_335, mul_429
#   mul_95 => mul_433
#   x_342 => convolution_100
#   x_343 => add_336, erf_47, mul_430, mul_431, mul_432
#   x_345 => convolution_101
#   x_347 => add_337
#   x_348 => convolution_102
# Graph fragment:
#   %mul_429 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_287, %unsqueeze_575), kwargs = {})
#   %add_335 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_429, %unsqueeze_572), kwargs = {})
#   %convolution_100 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_335, %arg120_1, %arg121_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_100, 0.5), kwargs = {})
#   %mul_431 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_100, 0.7071067811865476), kwargs = {})
#   %erf_47 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_431,), kwargs = {})
#   %add_336 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_47, 1), kwargs = {})
#   %mul_432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_430, %add_336), kwargs = {})
#   %convolution_101 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_432, %arg122_1, %arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_101, %view_288), kwargs = {})
#   %add_337 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_333, %mul_433), kwargs = {})
#   %convolution_102 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_337, %arg125_1, %arg126_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_gelu_mul_native_group_norm_28 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mul_native_group_norm_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_native_group_norm_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mul_native_group_norm_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1728*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ll/cllys4zhqjpdr7m5xma6aaj65uh4goiqvberdak6yfcrpyenndod.py
# Topologically Sorted Source Nodes: [group_norm_96], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_96 => var_mean_97
# Graph fragment:
#   %var_mean_97 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_289, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_29 = async_compile.triton('triton_per_fused_native_group_norm_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_29(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4720
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x1 = (xindex // 10) % 59
    x0 = xindex % 10
    x2 = (xindex // 590)
    x4 = xindex
    tmp0 = r3 + (128*x1)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x1) + (7527*x0)
    tmp4 = tl.full([1, 1], 75264, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((384*((r3 + (128*x1) + (7527*x0)) % 196)) + (75264*x2) + (((r3 + (128*x1) + (7527*x0)) // 196) % 384)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + (((r3 + (128*x1) + (7527*x0)) // 196) % 384), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = 0.0
    tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tmp19 = 1.0
    tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
    tmp23 = tl.where(tmp2, tmp21, tmp22)
    tmp24 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp25 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp28 = tl.where(xmask, tmp24, 0)
    tmp29 = tl.where(xmask, tmp25, 0)
    tmp30 = tl.where(xmask, tmp26, 0)
    tmp31, tmp32, tmp33 = triton_helpers.welford(tmp28, tmp29, tmp30, 1)
    tmp34 = tmp31[:, None]
    tmp35 = tmp32[:, None]
    tmp36 = tmp33[:, None]
    tl.store(out_ptr0 + (x4), tmp34, xmask)
    tl.store(out_ptr1 + (x4), tmp35, xmask)
    tl.store(out_ptr2 + (x4), tmp36, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4d/c4dsi5ht6obrajrfkqiifndx3h456qi2wymt6vr62nup2tfjkx3l.py
# Topologically Sorted Source Nodes: [group_norm_96], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_96 => var_mean_97
# Graph fragment:
#   %var_mean_97 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_289, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_30 = async_compile.triton('triton_per_fused_native_group_norm_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_30(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 59
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 10
    x1 = (xindex // 10)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (10*r2) + (590*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (10*r2) + (590*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (10*r2) + (590*x1)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zh/czh4tszbjljvk7klkvjjiqklpdb3bzoswieoirwhdq3ltudmy4dw.py
# Topologically Sorted Source Nodes: [group_norm_96], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_96 => var_mean_97
# Graph fragment:
#   %var_mean_97 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_289, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_31 = async_compile.triton('triton_per_fused_native_group_norm_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_31(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (10*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (10*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (10*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/fa/cfa55rpnp4bmhnzaoolofyktfg4v5ddsnkcqzniptquo6chxkmqp.py
# Topologically Sorted Source Nodes: [group_norm_96], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_96 => add_339, mul_435
# Graph fragment:
#   %mul_435 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_290, %unsqueeze_581), kwargs = {})
#   %add_339 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_435, %unsqueeze_578), kwargs = {})
triton_poi_fused_native_group_norm_32 = async_compile.triton('triton_poi_fused_native_group_norm_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 384
    x2 = (xindex // 75264)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 75264.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/up/cupriw3flz3pm7uhz7kj5jcqxxbtj7d7q3be4icjgoeirwmvqi2v.py
# Topologically Sorted Source Nodes: [group_norm_95, x_342, x_343, x_345, mul_95, x_347, x_348, y_48, sub_48, mul_96, x_349], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
# Source node to ATen node mapping:
#   group_norm_95 => add_335, mul_429
#   mul_95 => mul_433
#   mul_96 => mul_436
#   sub_48 => sub_146
#   x_342 => convolution_100
#   x_343 => add_336, erf_47, mul_430, mul_431, mul_432
#   x_345 => convolution_101
#   x_347 => add_337
#   x_348 => convolution_102
#   x_349 => add_340
#   y_48 => avg_pool2d_48
# Graph fragment:
#   %mul_429 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_287, %unsqueeze_575), kwargs = {})
#   %add_335 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_429, %unsqueeze_572), kwargs = {})
#   %convolution_100 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_335, %arg120_1, %arg121_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_100, 0.5), kwargs = {})
#   %mul_431 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_100, 0.7071067811865476), kwargs = {})
#   %erf_47 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_431,), kwargs = {})
#   %add_336 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_47, 1), kwargs = {})
#   %mul_432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_430, %add_336), kwargs = {})
#   %convolution_101 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_432, %arg122_1, %arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_101, %view_288), kwargs = {})
#   %add_337 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_333, %mul_433), kwargs = {})
#   %convolution_102 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_337, %arg125_1, %arg126_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %avg_pool2d_48 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_339, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
#   %sub_146 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_48, %add_339), kwargs = {})
#   %mul_436 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_146, %view_291), kwargs = {})
#   %add_340 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_102, %mul_436), kwargs = {})
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_33 = async_compile.triton('triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 14) % 14
    y0 = yindex % 14
    x3 = xindex
    y6 = yindex
    y2 = (yindex // 196)
    y4 = yindex % 196
    tmp54 = tl.load(in_ptr1 + (x3 + (384*y6)), xmask & ymask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr0 + (x3 + (384*y6)), xmask & ymask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + y0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5760) + x3 + (384*y6)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-5376) + x3 + (384*y6)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + y0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-4992) + x3 + (384*y6)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-384) + x3 + (384*y6)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x3 + (384*y6)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (384 + x3 + (384*y6)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + y1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (4992 + x3 + (384*y6)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (5376 + x3 + (384*y6)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5760 + x3 + (384*y6)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = (((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))*((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) + (((14) * ((14) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (14)))*((14) * ((14) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (14)))) + ((-1)*((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))*((14) * ((14) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (14)))) + ((-1)*((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))*((14) * ((14) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (14))))
    tmp53 = tmp51 / tmp52
    tmp56 = tmp54 + tmp55
    tmp58 = tmp53 - tmp57
    tmp60 = tmp58 * tmp59
    tmp61 = tmp56 + tmp60
    tl.store(out_ptr1 + (y4 + (196*x3) + (75264*y2)), tmp61, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6e/c6eg5y6wjf7xznkplu3foeqoy7be5r2u2wttpaq7didzgnfeplnm.py
# Topologically Sorted Source Nodes: [group_norm_97], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_97 => var_mean_98
# Graph fragment:
#   %var_mean_98 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_292, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_34 = async_compile.triton('triton_red_fused_native_group_norm_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_34(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 10
    x1 = (xindex // 10)
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x0)
        tmp1 = tl.full([1, 1], 75264, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((75264*x1) + ((r2 + (7527*x0)) % 75264)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = 1.0
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp11 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_combine(
            tmp13_mean, tmp13_m2, tmp13_weight,
            tmp10, tmp11, tmp12
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/if/ciflfrcvuulc32etwjk4fvm7hymzthydtei4c6rrgrpk43aytuqj.py
# Topologically Sorted Source Nodes: [group_norm_97], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_97 => add_342, mul_438
# Graph fragment:
#   %mul_438 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_293, %unsqueeze_587), kwargs = {})
#   %add_342 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_438, %unsqueeze_584), kwargs = {})
triton_poi_fused_native_group_norm_35 = async_compile.triton('triton_poi_fused_native_group_norm_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 75264.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/43/c43dhhi5dje4fqcurspvj24igz3oyqw55sv37wlm2jvyi55li255.py
# Topologically Sorted Source Nodes: [group_norm_97, x_350, x_351], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
# Source node to ATen node mapping:
#   group_norm_97 => add_342, mul_438
#   x_350 => convolution_103
#   x_351 => add_343, erf_48, mul_439, mul_440, mul_441
# Graph fragment:
#   %mul_438 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_293, %unsqueeze_587), kwargs = {})
#   %add_342 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_438, %unsqueeze_584), kwargs = {})
#   %convolution_103 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_342, %arg132_1, %arg133_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_439 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_103, 0.5), kwargs = {})
#   %mul_440 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_103, 0.7071067811865476), kwargs = {})
#   %erf_48 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_440,), kwargs = {})
#   %add_343 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_48, 1), kwargs = {})
#   %mul_441 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_439, %add_343), kwargs = {})
triton_poi_fused_convolution_gelu_native_group_norm_36 = async_compile.triton('triton_poi_fused_convolution_gelu_native_group_norm_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_native_group_norm_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_native_group_norm_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
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


# kernel path: /tmp/torchinductor_sahanp/ix/cix3zve5yubabx5m23xdzcqrid5hgwgzvoycpzx2w3nmzqxnf32l.py
# Topologically Sorted Source Nodes: [group_norm_98], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_98 => var_mean_99
# Graph fragment:
#   %var_mean_99 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_295, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_37 = async_compile.triton('triton_per_fused_native_group_norm_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4720
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = xindex % 59
    x1 = (xindex // 59) % 10
    x2 = (xindex // 590)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (7527*x1)
    tmp4 = tl.full([1, 1], 75264, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((75264*x2) + ((r3 + (128*x0) + (7527*x1)) % 75264)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((384*((r3 + (128*x0) + (7527*x1)) % 196)) + (75264*x2) + (((r3 + (128*x0) + (7527*x1)) // 196) % 384)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + (((r3 + (128*x0) + (7527*x1)) // 196) % 384), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x0) + (7527*x1)) // 196) % 384), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(xmask, tmp28, 0)
    tmp33 = tl.where(xmask, tmp29, 0)
    tmp34 = tl.where(xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x5), tmp38, xmask)
    tl.store(out_ptr1 + (x5), tmp39, xmask)
    tl.store(out_ptr2 + (x5), tmp40, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jt/cjtlfwbmwlhygs5ap355gx7m2bbnuxojpcne47uaxtjnglxb7rvw.py
# Topologically Sorted Source Nodes: [group_norm_98], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_98 => var_mean_99
# Graph fragment:
#   %var_mean_99 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_295, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_38 = async_compile.triton('triton_per_fused_native_group_norm_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_38(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 59
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (59*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (59*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (59*x0)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xm/cxmhqlwojsr7u75rjpnicff6zjyzet76q2muencneysbo6nnequu.py
# Topologically Sorted Source Nodes: [group_norm_98], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_98 => add_346, mul_444
# Graph fragment:
#   %mul_444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_296, %unsqueeze_593), kwargs = {})
#   %add_346 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_444, %unsqueeze_590), kwargs = {})
triton_poi_fused_native_group_norm_39 = async_compile.triton('triton_poi_fused_native_group_norm_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 75264.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3m/c3mj64cvaplhg5klr3zsrru2scapglrcmxsog4uo34ojb4nipo6f.py
# Topologically Sorted Source Nodes: [group_norm_97, x_350, x_351, x_353, mul_97, x_355, y_49, sub_49, mul_98, x_356], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
# Source node to ATen node mapping:
#   group_norm_97 => add_342, mul_438
#   mul_97 => mul_442
#   mul_98 => mul_445
#   sub_49 => sub_149
#   x_350 => convolution_103
#   x_351 => add_343, erf_48, mul_439, mul_440, mul_441
#   x_353 => convolution_104
#   x_355 => add_344
#   x_356 => add_347
#   y_49 => avg_pool2d_49
# Graph fragment:
#   %mul_438 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_293, %unsqueeze_587), kwargs = {})
#   %add_342 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_438, %unsqueeze_584), kwargs = {})
#   %convolution_103 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_342, %arg132_1, %arg133_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_439 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_103, 0.5), kwargs = {})
#   %mul_440 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_103, 0.7071067811865476), kwargs = {})
#   %erf_48 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_440,), kwargs = {})
#   %add_343 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_48, 1), kwargs = {})
#   %mul_441 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_439, %add_343), kwargs = {})
#   %convolution_104 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_441, %arg134_1, %arg135_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_104, %view_294), kwargs = {})
#   %add_344 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_340, %mul_442), kwargs = {})
#   %avg_pool2d_49 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_346, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
#   %sub_149 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_49, %add_346), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_149, %view_297), kwargs = {})
#   %add_347 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_344, %mul_445), kwargs = {})
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40 = async_compile.triton('triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 14)
    x1 = xindex % 14
    x5 = xindex
    y0 = yindex
    y3 = yindex % 384
    y4 = (yindex // 384)
    tmp54 = tl.load(in_out_ptr0 + (x5 + (196*y0)), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr1 + (y3 + (384*x5) + (75264*y4)), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr2 + (y3), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr3 + (y3), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr0 + (x5 + (196*y0)), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr4 + (y3), None, eviction_policy='evict_last')
    tmp0 = (-1) + x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-15) + x5 + (196*y0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-14) + x5 + (196*y0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-13) + x5 + (196*y0)), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x5 + (196*y0)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x5 + (196*y0)), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x5 + (196*y0)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (13 + x5 + (196*y0)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (14 + x5 + (196*y0)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (15 + x5 + (196*y0)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((0) * ((0) >= ((-1) + x2)) + ((-1) + x2) * (((-1) + x2) > (0)))) + (((14) * ((14) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (14)))*((14) * ((14) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (14)))) + ((-1)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((14) * ((14) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (14)))) + ((-1)*((0) * ((0) >= ((-1) + x2)) + ((-1) + x2) * (((-1) + x2) > (0)))*((14) * ((14) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (14))))
    tmp53 = tmp51 / tmp52
    tmp57 = tmp55 + tmp56
    tmp59 = tmp57 * tmp58
    tmp60 = tmp54 + tmp59
    tmp62 = tmp53 - tmp61
    tmp64 = tmp62 * tmp63
    tmp65 = tmp60 + tmp64
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (196*y0)), tmp65, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lc/clc5qq7tmzm4ydfc267wjpngqeitkr6xiblobd2pm76ex4x2uxx7.py
# Topologically Sorted Source Nodes: [group_norm_131, x_469, x_470, x_472, mul_131, x_474], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
# Source node to ATen node mapping:
#   group_norm_131 => add_461, mul_591
#   mul_131 => mul_595
#   x_469 => convolution_137
#   x_470 => add_462, erf_65, mul_592, mul_593, mul_594
#   x_472 => convolution_138
#   x_474 => add_463
# Graph fragment:
#   %mul_591 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_395, %unsqueeze_791), kwargs = {})
#   %add_461 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_591, %unsqueeze_788), kwargs = {})
#   %convolution_137 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_461, %arg302_1, %arg303_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_592 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_137, 0.5), kwargs = {})
#   %mul_593 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_137, 0.7071067811865476), kwargs = {})
#   %erf_65 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_593,), kwargs = {})
#   %add_462 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_65, 1), kwargs = {})
#   %mul_594 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_592, %add_462), kwargs = {})
#   %convolution_138 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_594, %arg304_1, %arg305_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_595 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_138, %view_396), kwargs = {})
#   %add_463 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_459, %mul_595), kwargs = {})
triton_poi_fused_add_convolution_gelu_mul_native_group_norm_41 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mul_native_group_norm_41', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_native_group_norm_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mul_native_group_norm_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (384*y3)), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kc/ckcvq4zvuxxazlbiy662ri4ysuqypnx56yc2pnke3b3edc3habx2.py
# Topologically Sorted Source Nodes: [group_norm_131, x_469, x_470, x_472, mul_131, x_474, x_475], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
# Source node to ATen node mapping:
#   group_norm_131 => add_461, mul_591
#   mul_131 => mul_595
#   x_469 => convolution_137
#   x_470 => add_462, erf_65, mul_592, mul_593, mul_594
#   x_472 => convolution_138
#   x_474 => add_463
#   x_475 => convolution_139
# Graph fragment:
#   %mul_591 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_395, %unsqueeze_791), kwargs = {})
#   %add_461 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_591, %unsqueeze_788), kwargs = {})
#   %convolution_137 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_461, %arg302_1, %arg303_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_592 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_137, 0.5), kwargs = {})
#   %mul_593 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_137, 0.7071067811865476), kwargs = {})
#   %erf_65 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_593,), kwargs = {})
#   %add_462 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_65, 1), kwargs = {})
#   %mul_594 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_592, %add_462), kwargs = {})
#   %convolution_138 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_594, %arg304_1, %arg305_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_595 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_138, %view_396), kwargs = {})
#   %add_463 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_459, %mul_595), kwargs = {})
#   %convolution_139 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_463, %arg307_1, %arg308_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_gelu_mul_native_group_norm_42 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mul_native_group_norm_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[524288, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_native_group_norm_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mul_native_group_norm_42(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 294912
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (3456*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wn/cwngsatsdcaip3c3lfzkr7y5hoofla7ec5fwrttl46nviglrvijo.py
# Topologically Sorted Source Nodes: [group_norm_132], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_132 => var_mean_133
# Graph fragment:
#   %var_mean_133 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_397, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_43 = async_compile.triton('triton_per_fused_native_group_norm_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_43(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2360
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = xindex % 59
    x1 = (xindex // 59) % 5
    x2 = (xindex // 295)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (7527*x1)
    tmp4 = tl.full([1, 1], 37632, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + (((r3 + (128*x0) + (7527*x1)) // 49) % 768), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = 0.0
    tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tmp19 = 1.0
    tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
    tmp23 = tl.where(tmp2, tmp21, tmp22)
    tmp24 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp25 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp28 = tl.where(xmask, tmp24, 0)
    tmp29 = tl.where(xmask, tmp25, 0)
    tmp30 = tl.where(xmask, tmp26, 0)
    tmp31, tmp32, tmp33 = triton_helpers.welford(tmp28, tmp29, tmp30, 1)
    tmp34 = tmp31[:, None]
    tmp35 = tmp32[:, None]
    tmp36 = tmp33[:, None]
    tl.store(out_ptr0 + (x5), tmp34, xmask)
    tl.store(out_ptr1 + (x5), tmp35, xmask)
    tl.store(out_ptr2 + (x5), tmp36, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uo/cuoagmmclqhtrtandkeyuiehmfinpgm7hnbhepcxvo5ldmgguzgx.py
# Topologically Sorted Source Nodes: [group_norm_132], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_132 => var_mean_133
# Graph fragment:
#   %var_mean_133 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_397, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_44 = async_compile.triton('triton_per_fused_native_group_norm_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[64, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_44(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 59
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (59*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (59*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (59*x0)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vs/cvsvcgek5jjqhh2wgb3kgr5vm6st4z4ega62nd63w3t4gncsgdyx.py
# Topologically Sorted Source Nodes: [group_norm_132], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_132 => var_mean_133
# Graph fragment:
#   %var_mean_133 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_397, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_45 = async_compile.triton('triton_per_fused_native_group_norm_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_45(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (5*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (5*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (5*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/s7/cs7m67ti6vyrwbqjndfbj6tx44qr7bczwnmsaejwkbdcevwayvzy.py
# Topologically Sorted Source Nodes: [group_norm_132], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_132 => add_465, mul_597
# Graph fragment:
#   %mul_597 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_398, %unsqueeze_797), kwargs = {})
#   %add_465 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_597, %unsqueeze_794), kwargs = {})
triton_poi_fused_native_group_norm_46 = async_compile.triton('triton_poi_fused_native_group_norm_46', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 768
    x2 = (xindex // 37632)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 37632.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pc/cpc4g266kcmp2itetqel5zj7nt4footkmxi53p4gcfanh3nsnn5v.py
# Topologically Sorted Source Nodes: [group_norm_131, x_469, x_470, x_472, mul_131, x_474, x_475, y_66, sub_66, mul_132, x_476], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
# Source node to ATen node mapping:
#   group_norm_131 => add_461, mul_591
#   mul_131 => mul_595
#   mul_132 => mul_598
#   sub_66 => sub_200
#   x_469 => convolution_137
#   x_470 => add_462, erf_65, mul_592, mul_593, mul_594
#   x_472 => convolution_138
#   x_474 => add_463
#   x_475 => convolution_139
#   x_476 => add_466
#   y_66 => avg_pool2d_66
# Graph fragment:
#   %mul_591 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_395, %unsqueeze_791), kwargs = {})
#   %add_461 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_591, %unsqueeze_788), kwargs = {})
#   %convolution_137 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_461, %arg302_1, %arg303_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_592 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_137, 0.5), kwargs = {})
#   %mul_593 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_137, 0.7071067811865476), kwargs = {})
#   %erf_65 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_593,), kwargs = {})
#   %add_462 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_65, 1), kwargs = {})
#   %mul_594 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_592, %add_462), kwargs = {})
#   %convolution_138 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_594, %arg304_1, %arg305_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_595 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_138, %view_396), kwargs = {})
#   %add_463 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_459, %mul_595), kwargs = {})
#   %convolution_139 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_463, %arg307_1, %arg308_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %avg_pool2d_66 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_465, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
#   %sub_200 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_66, %add_465), kwargs = {})
#   %mul_598 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_200, %view_399), kwargs = {})
#   %add_466 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_139, %mul_598), kwargs = {})
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_47 = async_compile.triton('triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 7) % 7
    y0 = yindex % 7
    x3 = xindex
    y6 = yindex
    y2 = (yindex // 49)
    y4 = yindex % 49
    tmp54 = tl.load(in_ptr1 + (x3 + (768*y6)), xmask & ymask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr0 + (x3 + (768*y6)), xmask & ymask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 7, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + y0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-6144) + x3 + (768*y6)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-5376) + x3 + (768*y6)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + y0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-4608) + x3 + (768*y6)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-768) + x3 + (768*y6)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x3 + (768*y6)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (768 + x3 + (768*y6)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + y1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (4608 + x3 + (768*y6)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (5376 + x3 + (768*y6)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (6144 + x3 + (768*y6)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = (((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))*((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) + (((7) * ((7) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (7)))*((7) * ((7) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (7)))) + ((-1)*((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))*((7) * ((7) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (7)))) + ((-1)*((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))*((7) * ((7) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (7))))
    tmp53 = tmp51 / tmp52
    tmp56 = tmp54 + tmp55
    tmp58 = tmp53 - tmp57
    tmp60 = tmp58 * tmp59
    tmp61 = tmp56 + tmp60
    tl.store(out_ptr1 + (y4 + (49*x3) + (37632*y2)), tmp61, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oy/coydl6x7yrcmqdtddbmxvzbjqo32rdxyflujb5be5gw4wrjhgdzy.py
# Topologically Sorted Source Nodes: [group_norm_133], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_133 => var_mean_134
# Graph fragment:
#   %var_mean_134 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_400, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_48 = async_compile.triton('triton_red_fused_native_group_norm_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_48(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x1 = (xindex // 5)
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x0)
        tmp1 = tl.full([1, 1], 37632, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((37632*x1) + ((r2 + (7527*x0)) % 37632)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = 1.0
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp11 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_combine(
            tmp13_mean, tmp13_m2, tmp13_weight,
            tmp10, tmp11, tmp12
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tt/ctti66gutwt3xpjkoqoxkmtg5sy4th57eqlb3dvgkdcenuhuyrwc.py
# Topologically Sorted Source Nodes: [group_norm_133], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_133 => add_468, mul_600
# Graph fragment:
#   %mul_600 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_401, %unsqueeze_803), kwargs = {})
#   %add_468 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_600, %unsqueeze_800), kwargs = {})
triton_poi_fused_native_group_norm_49 = async_compile.triton('triton_poi_fused_native_group_norm_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 37632.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (768*x2) + (37632*y1)), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uz/cuzokfdlwlzth4gvy3el5mdvjv6vxjj3fs7oqwjrnexec7ex4fwv.py
# Topologically Sorted Source Nodes: [group_norm_133, x_477, x_478], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
# Source node to ATen node mapping:
#   group_norm_133 => add_468, mul_600
#   x_477 => convolution_140
#   x_478 => add_469, erf_66, mul_601, mul_602, mul_603
# Graph fragment:
#   %mul_600 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_401, %unsqueeze_803), kwargs = {})
#   %add_468 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_600, %unsqueeze_800), kwargs = {})
#   %convolution_140 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_468, %arg314_1, %arg315_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_601 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_140, 0.5), kwargs = {})
#   %mul_602 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_140, 0.7071067811865476), kwargs = {})
#   %erf_66 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_602,), kwargs = {})
#   %add_469 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_66, 1), kwargs = {})
#   %mul_603 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_601, %add_469), kwargs = {})
triton_poi_fused_convolution_gelu_native_group_norm_50 = async_compile.triton('triton_poi_fused_convolution_gelu_native_group_norm_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_native_group_norm_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_native_group_norm_50(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
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


# kernel path: /tmp/torchinductor_sahanp/6k/c6kyoo53uxai5clmn7xzhzk5g4jdrp5pnpyin6hoksfyulvf6jge.py
# Topologically Sorted Source Nodes: [group_norm_134], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_134 => var_mean_135
# Graph fragment:
#   %var_mean_135 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_403, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_51 = async_compile.triton('triton_per_fused_native_group_norm_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2360
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = xindex % 59
    x1 = (xindex // 59) % 5
    x2 = (xindex // 295)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (7527*x1)
    tmp4 = tl.full([1, 1], 37632, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((37632*x2) + ((r3 + (128*x0) + (7527*x1)) % 37632)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + (((r3 + (128*x0) + (7527*x1)) // 49) % 768), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x0) + (7527*x1)) // 49) % 768), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(xmask, tmp28, 0)
    tmp33 = tl.where(xmask, tmp29, 0)
    tmp34 = tl.where(xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x5), tmp38, xmask)
    tl.store(out_ptr1 + (x5), tmp39, xmask)
    tl.store(out_ptr2 + (x5), tmp40, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6j/c6jwd4yomuruh5pebj56x2sltikh4mu7ubsxojrv7fbjz2u2d464.py
# Topologically Sorted Source Nodes: [group_norm_134], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_134 => add_472, mul_606
# Graph fragment:
#   %mul_606 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_404, %unsqueeze_809), kwargs = {})
#   %add_472 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_606, %unsqueeze_806), kwargs = {})
triton_poi_fused_native_group_norm_52 = async_compile.triton('triton_poi_fused_native_group_norm_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 37632.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/od/codxpdsc373zcqrg22npollp4s5243atcyditzngq7spssrca62c.py
# Topologically Sorted Source Nodes: [group_norm_133, x_477, x_478, x_480, mul_133, x_482, y_67, sub_67, mul_134, x_483], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
# Source node to ATen node mapping:
#   group_norm_133 => add_468, mul_600
#   mul_133 => mul_604
#   mul_134 => mul_607
#   sub_67 => sub_203
#   x_477 => convolution_140
#   x_478 => add_469, erf_66, mul_601, mul_602, mul_603
#   x_480 => convolution_141
#   x_482 => add_470
#   x_483 => add_473
#   y_67 => avg_pool2d_67
# Graph fragment:
#   %mul_600 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_401, %unsqueeze_803), kwargs = {})
#   %add_468 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_600, %unsqueeze_800), kwargs = {})
#   %convolution_140 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_468, %arg314_1, %arg315_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_601 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_140, 0.5), kwargs = {})
#   %mul_602 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_140, 0.7071067811865476), kwargs = {})
#   %erf_66 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_602,), kwargs = {})
#   %add_469 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_66, 1), kwargs = {})
#   %mul_603 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_601, %add_469), kwargs = {})
#   %convolution_141 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_603, %arg316_1, %arg317_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_604 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_141, %view_402), kwargs = {})
#   %add_470 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_466, %mul_604), kwargs = {})
#   %avg_pool2d_67 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_472, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
#   %sub_203 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_67, %add_472), kwargs = {})
#   %mul_607 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_203, %view_405), kwargs = {})
#   %add_473 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_470, %mul_607), kwargs = {})
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_53 = async_compile.triton('triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_53', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_53(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 7)
    x1 = xindex % 7
    x5 = xindex
    y0 = yindex
    y3 = yindex % 768
    y4 = (yindex // 768)
    tmp54 = tl.load(in_out_ptr0 + (x5 + (49*y0)), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr1 + (y3 + (768*x5) + (37632*y4)), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr2 + (y3), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr3 + (y3), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr0 + (x5 + (49*y0)), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr4 + (y3), None, eviction_policy='evict_last')
    tmp0 = (-1) + x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 7, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-8) + x5 + (49*y0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-7) + x5 + (49*y0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-6) + x5 + (49*y0)), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x5 + (49*y0)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x5 + (49*y0)), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x5 + (49*y0)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (6 + x5 + (49*y0)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (7 + x5 + (49*y0)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (8 + x5 + (49*y0)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((0) * ((0) >= ((-1) + x2)) + ((-1) + x2) * (((-1) + x2) > (0)))) + (((7) * ((7) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (7)))*((7) * ((7) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (7)))) + ((-1)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((7) * ((7) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (7)))) + ((-1)*((0) * ((0) >= ((-1) + x2)) + ((-1) + x2) * (((-1) + x2) > (0)))*((7) * ((7) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (7))))
    tmp53 = tmp51 / tmp52
    tmp57 = tmp55 + tmp56
    tmp59 = tmp57 * tmp58
    tmp60 = tmp54 + tmp59
    tmp62 = tmp53 - tmp61
    tmp64 = tmp62 * tmp63
    tmp65 = tmp60 + tmp64
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (49*y0)), tmp65, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gv/cgvhdp4l7loekuzubkv3as2ntcgohtc7n2hubdpz32g2gw3msjxg.py
# Topologically Sorted Source Nodes: [group_norm_143, x_512, x_513, x_515, mul_143, x_517, x_518], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.mean]
# Source node to ATen node mapping:
#   group_norm_143 => add_503, mul_645
#   mul_143 => mul_649
#   x_512 => convolution_150
#   x_513 => add_504, erf_71, mul_646, mul_647, mul_648
#   x_515 => convolution_151
#   x_517 => add_505
#   x_518 => mean_1
# Graph fragment:
#   %mul_645 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_431, %unsqueeze_863), kwargs = {})
#   %add_503 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_645, %unsqueeze_860), kwargs = {})
#   %convolution_150 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_503, %arg364_1, %arg365_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_646 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_150, 0.5), kwargs = {})
#   %mul_647 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_150, 0.7071067811865476), kwargs = {})
#   %erf_71 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_647,), kwargs = {})
#   %add_504 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_71, 1), kwargs = {})
#   %mul_648 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_646, %add_504), kwargs = {})
#   %convolution_151 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_648, %arg366_1, %arg367_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_649 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_151, %view_432), kwargs = {})
#   %add_505 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_501, %mul_649), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_505, [-1, -2], True), kwargs = {})
triton_per_fused_add_convolution_gelu_mean_mul_native_group_norm_54 = async_compile.triton('triton_per_fused_add_convolution_gelu_mean_mul_native_group_norm_54', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_gelu_mean_mul_native_group_norm_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_gelu_mean_mul_native_group_norm_54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (37632*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/br/cbrumpkjxpf6rua6lvlplxzkoo2gcj7mivk6nru2v7m2ubky63is.py
# Topologically Sorted Source Nodes: [x_520], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_520 => add_506, add_507, mul_650, mul_651, rsqrt_145, sub_217, var_mean_145
# Graph fragment:
#   %var_mean_145 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_3, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_217 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_3, %getitem_291), kwargs = {})
#   %add_506 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_290, 1e-06), kwargs = {})
#   %rsqrt_145 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_506,), kwargs = {})
#   %mul_650 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_217, %rsqrt_145), kwargs = {})
#   %mul_651 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_650, %arg369_1), kwargs = {})
#   %add_507 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_651, %arg370_1), kwargs = {})
triton_per_fused_native_layer_norm_55 = async_compile.triton('triton_per_fused_native_layer_norm_55', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_55(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp26 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 768, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 768.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-06
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp29, rmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1 = args
    args.clear()
    assert_size_stride(arg0_1, (96, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (96, ), (1, ))
    assert_size_stride(arg2_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg3_1, (96, ), (1, ))
    assert_size_stride(arg4_1, (96, ), (1, ))
    assert_size_stride(arg5_1, (96, ), (1, ))
    assert_size_stride(arg6_1, (96, ), (1, ))
    assert_size_stride(arg7_1, (96, ), (1, ))
    assert_size_stride(arg8_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg9_1, (384, ), (1, ))
    assert_size_stride(arg10_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg11_1, (96, ), (1, ))
    assert_size_stride(arg12_1, (96, ), (1, ))
    assert_size_stride(arg13_1, (96, ), (1, ))
    assert_size_stride(arg14_1, (96, ), (1, ))
    assert_size_stride(arg15_1, (96, ), (1, ))
    assert_size_stride(arg16_1, (96, ), (1, ))
    assert_size_stride(arg17_1, (96, ), (1, ))
    assert_size_stride(arg18_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg19_1, (384, ), (1, ))
    assert_size_stride(arg20_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg21_1, (96, ), (1, ))
    assert_size_stride(arg22_1, (96, ), (1, ))
    assert_size_stride(arg23_1, (96, ), (1, ))
    assert_size_stride(arg24_1, (96, ), (1, ))
    assert_size_stride(arg25_1, (96, ), (1, ))
    assert_size_stride(arg26_1, (96, ), (1, ))
    assert_size_stride(arg27_1, (96, ), (1, ))
    assert_size_stride(arg28_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg29_1, (384, ), (1, ))
    assert_size_stride(arg30_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg31_1, (96, ), (1, ))
    assert_size_stride(arg32_1, (96, ), (1, ))
    assert_size_stride(arg33_1, (96, ), (1, ))
    assert_size_stride(arg34_1, (96, ), (1, ))
    assert_size_stride(arg35_1, (96, ), (1, ))
    assert_size_stride(arg36_1, (96, ), (1, ))
    assert_size_stride(arg37_1, (96, ), (1, ))
    assert_size_stride(arg38_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg41_1, (96, ), (1, ))
    assert_size_stride(arg42_1, (96, ), (1, ))
    assert_size_stride(arg43_1, (96, ), (1, ))
    assert_size_stride(arg44_1, (96, ), (1, ))
    assert_size_stride(arg45_1, (96, ), (1, ))
    assert_size_stride(arg46_1, (96, ), (1, ))
    assert_size_stride(arg47_1, (96, ), (1, ))
    assert_size_stride(arg48_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg49_1, (384, ), (1, ))
    assert_size_stride(arg50_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg51_1, (96, ), (1, ))
    assert_size_stride(arg52_1, (96, ), (1, ))
    assert_size_stride(arg53_1, (96, ), (1, ))
    assert_size_stride(arg54_1, (96, ), (1, ))
    assert_size_stride(arg55_1, (96, ), (1, ))
    assert_size_stride(arg56_1, (96, ), (1, ))
    assert_size_stride(arg57_1, (96, ), (1, ))
    assert_size_stride(arg58_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg59_1, (384, ), (1, ))
    assert_size_stride(arg60_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg61_1, (96, ), (1, ))
    assert_size_stride(arg62_1, (96, ), (1, ))
    assert_size_stride(arg63_1, (192, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg64_1, (192, ), (1, ))
    assert_size_stride(arg65_1, (192, ), (1, ))
    assert_size_stride(arg66_1, (192, ), (1, ))
    assert_size_stride(arg67_1, (192, ), (1, ))
    assert_size_stride(arg68_1, (192, ), (1, ))
    assert_size_stride(arg69_1, (192, ), (1, ))
    assert_size_stride(arg70_1, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg73_1, (192, ), (1, ))
    assert_size_stride(arg74_1, (192, ), (1, ))
    assert_size_stride(arg75_1, (192, ), (1, ))
    assert_size_stride(arg76_1, (192, ), (1, ))
    assert_size_stride(arg77_1, (192, ), (1, ))
    assert_size_stride(arg78_1, (192, ), (1, ))
    assert_size_stride(arg79_1, (192, ), (1, ))
    assert_size_stride(arg80_1, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg83_1, (192, ), (1, ))
    assert_size_stride(arg84_1, (192, ), (1, ))
    assert_size_stride(arg85_1, (192, ), (1, ))
    assert_size_stride(arg86_1, (192, ), (1, ))
    assert_size_stride(arg87_1, (192, ), (1, ))
    assert_size_stride(arg88_1, (192, ), (1, ))
    assert_size_stride(arg89_1, (192, ), (1, ))
    assert_size_stride(arg90_1, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg93_1, (192, ), (1, ))
    assert_size_stride(arg94_1, (192, ), (1, ))
    assert_size_stride(arg95_1, (192, ), (1, ))
    assert_size_stride(arg96_1, (192, ), (1, ))
    assert_size_stride(arg97_1, (192, ), (1, ))
    assert_size_stride(arg98_1, (192, ), (1, ))
    assert_size_stride(arg99_1, (192, ), (1, ))
    assert_size_stride(arg100_1, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg103_1, (192, ), (1, ))
    assert_size_stride(arg104_1, (192, ), (1, ))
    assert_size_stride(arg105_1, (192, ), (1, ))
    assert_size_stride(arg106_1, (192, ), (1, ))
    assert_size_stride(arg107_1, (192, ), (1, ))
    assert_size_stride(arg108_1, (192, ), (1, ))
    assert_size_stride(arg109_1, (192, ), (1, ))
    assert_size_stride(arg110_1, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg113_1, (192, ), (1, ))
    assert_size_stride(arg114_1, (192, ), (1, ))
    assert_size_stride(arg115_1, (192, ), (1, ))
    assert_size_stride(arg116_1, (192, ), (1, ))
    assert_size_stride(arg117_1, (192, ), (1, ))
    assert_size_stride(arg118_1, (192, ), (1, ))
    assert_size_stride(arg119_1, (192, ), (1, ))
    assert_size_stride(arg120_1, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg123_1, (192, ), (1, ))
    assert_size_stride(arg124_1, (192, ), (1, ))
    assert_size_stride(arg125_1, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg126_1, (384, ), (1, ))
    assert_size_stride(arg127_1, (384, ), (1, ))
    assert_size_stride(arg128_1, (384, ), (1, ))
    assert_size_stride(arg129_1, (384, ), (1, ))
    assert_size_stride(arg130_1, (384, ), (1, ))
    assert_size_stride(arg131_1, (384, ), (1, ))
    assert_size_stride(arg132_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg133_1, (1536, ), (1, ))
    assert_size_stride(arg134_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg135_1, (384, ), (1, ))
    assert_size_stride(arg136_1, (384, ), (1, ))
    assert_size_stride(arg137_1, (384, ), (1, ))
    assert_size_stride(arg138_1, (384, ), (1, ))
    assert_size_stride(arg139_1, (384, ), (1, ))
    assert_size_stride(arg140_1, (384, ), (1, ))
    assert_size_stride(arg141_1, (384, ), (1, ))
    assert_size_stride(arg142_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg143_1, (1536, ), (1, ))
    assert_size_stride(arg144_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (384, ), (1, ))
    assert_size_stride(arg147_1, (384, ), (1, ))
    assert_size_stride(arg148_1, (384, ), (1, ))
    assert_size_stride(arg149_1, (384, ), (1, ))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (384, ), (1, ))
    assert_size_stride(arg152_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg153_1, (1536, ), (1, ))
    assert_size_stride(arg154_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg155_1, (384, ), (1, ))
    assert_size_stride(arg156_1, (384, ), (1, ))
    assert_size_stride(arg157_1, (384, ), (1, ))
    assert_size_stride(arg158_1, (384, ), (1, ))
    assert_size_stride(arg159_1, (384, ), (1, ))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (384, ), (1, ))
    assert_size_stride(arg162_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg163_1, (1536, ), (1, ))
    assert_size_stride(arg164_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg165_1, (384, ), (1, ))
    assert_size_stride(arg166_1, (384, ), (1, ))
    assert_size_stride(arg167_1, (384, ), (1, ))
    assert_size_stride(arg168_1, (384, ), (1, ))
    assert_size_stride(arg169_1, (384, ), (1, ))
    assert_size_stride(arg170_1, (384, ), (1, ))
    assert_size_stride(arg171_1, (384, ), (1, ))
    assert_size_stride(arg172_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg173_1, (1536, ), (1, ))
    assert_size_stride(arg174_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg175_1, (384, ), (1, ))
    assert_size_stride(arg176_1, (384, ), (1, ))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (384, ), (1, ))
    assert_size_stride(arg179_1, (384, ), (1, ))
    assert_size_stride(arg180_1, (384, ), (1, ))
    assert_size_stride(arg181_1, (384, ), (1, ))
    assert_size_stride(arg182_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg183_1, (1536, ), (1, ))
    assert_size_stride(arg184_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg185_1, (384, ), (1, ))
    assert_size_stride(arg186_1, (384, ), (1, ))
    assert_size_stride(arg187_1, (384, ), (1, ))
    assert_size_stride(arg188_1, (384, ), (1, ))
    assert_size_stride(arg189_1, (384, ), (1, ))
    assert_size_stride(arg190_1, (384, ), (1, ))
    assert_size_stride(arg191_1, (384, ), (1, ))
    assert_size_stride(arg192_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg193_1, (1536, ), (1, ))
    assert_size_stride(arg194_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg195_1, (384, ), (1, ))
    assert_size_stride(arg196_1, (384, ), (1, ))
    assert_size_stride(arg197_1, (384, ), (1, ))
    assert_size_stride(arg198_1, (384, ), (1, ))
    assert_size_stride(arg199_1, (384, ), (1, ))
    assert_size_stride(arg200_1, (384, ), (1, ))
    assert_size_stride(arg201_1, (384, ), (1, ))
    assert_size_stride(arg202_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg203_1, (1536, ), (1, ))
    assert_size_stride(arg204_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg205_1, (384, ), (1, ))
    assert_size_stride(arg206_1, (384, ), (1, ))
    assert_size_stride(arg207_1, (384, ), (1, ))
    assert_size_stride(arg208_1, (384, ), (1, ))
    assert_size_stride(arg209_1, (384, ), (1, ))
    assert_size_stride(arg210_1, (384, ), (1, ))
    assert_size_stride(arg211_1, (384, ), (1, ))
    assert_size_stride(arg212_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg213_1, (1536, ), (1, ))
    assert_size_stride(arg214_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg215_1, (384, ), (1, ))
    assert_size_stride(arg216_1, (384, ), (1, ))
    assert_size_stride(arg217_1, (384, ), (1, ))
    assert_size_stride(arg218_1, (384, ), (1, ))
    assert_size_stride(arg219_1, (384, ), (1, ))
    assert_size_stride(arg220_1, (384, ), (1, ))
    assert_size_stride(arg221_1, (384, ), (1, ))
    assert_size_stride(arg222_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg223_1, (1536, ), (1, ))
    assert_size_stride(arg224_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg225_1, (384, ), (1, ))
    assert_size_stride(arg226_1, (384, ), (1, ))
    assert_size_stride(arg227_1, (384, ), (1, ))
    assert_size_stride(arg228_1, (384, ), (1, ))
    assert_size_stride(arg229_1, (384, ), (1, ))
    assert_size_stride(arg230_1, (384, ), (1, ))
    assert_size_stride(arg231_1, (384, ), (1, ))
    assert_size_stride(arg232_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg233_1, (1536, ), (1, ))
    assert_size_stride(arg234_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg235_1, (384, ), (1, ))
    assert_size_stride(arg236_1, (384, ), (1, ))
    assert_size_stride(arg237_1, (384, ), (1, ))
    assert_size_stride(arg238_1, (384, ), (1, ))
    assert_size_stride(arg239_1, (384, ), (1, ))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (384, ), (1, ))
    assert_size_stride(arg242_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg243_1, (1536, ), (1, ))
    assert_size_stride(arg244_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg245_1, (384, ), (1, ))
    assert_size_stride(arg246_1, (384, ), (1, ))
    assert_size_stride(arg247_1, (384, ), (1, ))
    assert_size_stride(arg248_1, (384, ), (1, ))
    assert_size_stride(arg249_1, (384, ), (1, ))
    assert_size_stride(arg250_1, (384, ), (1, ))
    assert_size_stride(arg251_1, (384, ), (1, ))
    assert_size_stride(arg252_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg253_1, (1536, ), (1, ))
    assert_size_stride(arg254_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg255_1, (384, ), (1, ))
    assert_size_stride(arg256_1, (384, ), (1, ))
    assert_size_stride(arg257_1, (384, ), (1, ))
    assert_size_stride(arg258_1, (384, ), (1, ))
    assert_size_stride(arg259_1, (384, ), (1, ))
    assert_size_stride(arg260_1, (384, ), (1, ))
    assert_size_stride(arg261_1, (384, ), (1, ))
    assert_size_stride(arg262_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg263_1, (1536, ), (1, ))
    assert_size_stride(arg264_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg265_1, (384, ), (1, ))
    assert_size_stride(arg266_1, (384, ), (1, ))
    assert_size_stride(arg267_1, (384, ), (1, ))
    assert_size_stride(arg268_1, (384, ), (1, ))
    assert_size_stride(arg269_1, (384, ), (1, ))
    assert_size_stride(arg270_1, (384, ), (1, ))
    assert_size_stride(arg271_1, (384, ), (1, ))
    assert_size_stride(arg272_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg273_1, (1536, ), (1, ))
    assert_size_stride(arg274_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg275_1, (384, ), (1, ))
    assert_size_stride(arg276_1, (384, ), (1, ))
    assert_size_stride(arg277_1, (384, ), (1, ))
    assert_size_stride(arg278_1, (384, ), (1, ))
    assert_size_stride(arg279_1, (384, ), (1, ))
    assert_size_stride(arg280_1, (384, ), (1, ))
    assert_size_stride(arg281_1, (384, ), (1, ))
    assert_size_stride(arg282_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg283_1, (1536, ), (1, ))
    assert_size_stride(arg284_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg285_1, (384, ), (1, ))
    assert_size_stride(arg286_1, (384, ), (1, ))
    assert_size_stride(arg287_1, (384, ), (1, ))
    assert_size_stride(arg288_1, (384, ), (1, ))
    assert_size_stride(arg289_1, (384, ), (1, ))
    assert_size_stride(arg290_1, (384, ), (1, ))
    assert_size_stride(arg291_1, (384, ), (1, ))
    assert_size_stride(arg292_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg293_1, (1536, ), (1, ))
    assert_size_stride(arg294_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg295_1, (384, ), (1, ))
    assert_size_stride(arg296_1, (384, ), (1, ))
    assert_size_stride(arg297_1, (384, ), (1, ))
    assert_size_stride(arg298_1, (384, ), (1, ))
    assert_size_stride(arg299_1, (384, ), (1, ))
    assert_size_stride(arg300_1, (384, ), (1, ))
    assert_size_stride(arg301_1, (384, ), (1, ))
    assert_size_stride(arg302_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg303_1, (1536, ), (1, ))
    assert_size_stride(arg304_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg305_1, (384, ), (1, ))
    assert_size_stride(arg306_1, (384, ), (1, ))
    assert_size_stride(arg307_1, (768, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg308_1, (768, ), (1, ))
    assert_size_stride(arg309_1, (768, ), (1, ))
    assert_size_stride(arg310_1, (768, ), (1, ))
    assert_size_stride(arg311_1, (768, ), (1, ))
    assert_size_stride(arg312_1, (768, ), (1, ))
    assert_size_stride(arg313_1, (768, ), (1, ))
    assert_size_stride(arg314_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg315_1, (3072, ), (1, ))
    assert_size_stride(arg316_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg317_1, (768, ), (1, ))
    assert_size_stride(arg318_1, (768, ), (1, ))
    assert_size_stride(arg319_1, (768, ), (1, ))
    assert_size_stride(arg320_1, (768, ), (1, ))
    assert_size_stride(arg321_1, (768, ), (1, ))
    assert_size_stride(arg322_1, (768, ), (1, ))
    assert_size_stride(arg323_1, (768, ), (1, ))
    assert_size_stride(arg324_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg325_1, (3072, ), (1, ))
    assert_size_stride(arg326_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg327_1, (768, ), (1, ))
    assert_size_stride(arg328_1, (768, ), (1, ))
    assert_size_stride(arg329_1, (768, ), (1, ))
    assert_size_stride(arg330_1, (768, ), (1, ))
    assert_size_stride(arg331_1, (768, ), (1, ))
    assert_size_stride(arg332_1, (768, ), (1, ))
    assert_size_stride(arg333_1, (768, ), (1, ))
    assert_size_stride(arg334_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg335_1, (3072, ), (1, ))
    assert_size_stride(arg336_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg337_1, (768, ), (1, ))
    assert_size_stride(arg338_1, (768, ), (1, ))
    assert_size_stride(arg339_1, (768, ), (1, ))
    assert_size_stride(arg340_1, (768, ), (1, ))
    assert_size_stride(arg341_1, (768, ), (1, ))
    assert_size_stride(arg342_1, (768, ), (1, ))
    assert_size_stride(arg343_1, (768, ), (1, ))
    assert_size_stride(arg344_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg345_1, (3072, ), (1, ))
    assert_size_stride(arg346_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg347_1, (768, ), (1, ))
    assert_size_stride(arg348_1, (768, ), (1, ))
    assert_size_stride(arg349_1, (768, ), (1, ))
    assert_size_stride(arg350_1, (768, ), (1, ))
    assert_size_stride(arg351_1, (768, ), (1, ))
    assert_size_stride(arg352_1, (768, ), (1, ))
    assert_size_stride(arg353_1, (768, ), (1, ))
    assert_size_stride(arg354_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg355_1, (3072, ), (1, ))
    assert_size_stride(arg356_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg357_1, (768, ), (1, ))
    assert_size_stride(arg358_1, (768, ), (1, ))
    assert_size_stride(arg359_1, (768, ), (1, ))
    assert_size_stride(arg360_1, (768, ), (1, ))
    assert_size_stride(arg361_1, (768, ), (1, ))
    assert_size_stride(arg362_1, (768, ), (1, ))
    assert_size_stride(arg363_1, (768, ), (1, ))
    assert_size_stride(arg364_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg365_1, (3072, ), (1, ))
    assert_size_stride(arg366_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg367_1, (768, ), (1, ))
    assert_size_stride(arg368_1, (768, ), (1, ))
    assert_size_stride(arg369_1, (768, ), (1, ))
    assert_size_stride(arg370_1, (768, ), (1, ))
    assert_size_stride(arg371_1, (1000, 768), (768, 1))
    assert_size_stride(arg372_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_262], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg2_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg2_1
        buf1 = empty_strided_cuda((96, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_262], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 288, 49, grid=grid(288, 49), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_262], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del buf1
        buf3 = empty_strided_cuda((8, 1, 1, 1, 37, 64), (2368, 18944, 18944, 18944, 64, 1), torch.float32)
        buf4 = empty_strided_cuda((8, 1, 1, 1, 37, 64), (2368, 18944, 18944, 18944, 64, 1), torch.float32)
        buf5 = empty_strided_cuda((8, 1, 1, 1, 37, 64), (2368, 18944, 18944, 18944, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_72], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_2.run(buf2, arg1_1, buf3, buf4, buf5, 18944, 128, grid=grid(18944), stream=stream0)
        buf6 = empty_strided_cuda((8, 1, 1, 1, 37), (37, 296, 296, 296, 1), torch.float32)
        buf7 = empty_strided_cuda((8, 1, 1, 1, 37), (37, 296, 296, 296, 1), torch.float32)
        buf8 = empty_strided_cuda((8, 1, 1, 1, 37), (37, 296, 296, 296, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_72], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf3, buf4, buf5, buf6, buf7, buf8, 296, 64, grid=grid(296), stream=stream0)
        buf9 = empty_strided_cuda((8, 1, 1, 1), (1, 8, 8, 8), torch.float32)
        buf10 = empty_strided_cuda((8, 1, 1, 1), (1, 8, 8, 8), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_72], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf6, buf7, buf8, buf9, buf10, 8, 37, grid=grid(8), stream=stream0)
        buf12 = empty_strided_cuda((8, 96, 56, 56), (301056, 1, 5376, 96), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_72], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_5.run(buf2, arg1_1, buf9, buf10, arg3_1, arg4_1, buf12, 2408448, grid=grid(2408448), stream=stream0)
        del arg3_1
        del arg4_1
        buf14 = empty_strided_cuda((8, 96, 56, 56), (301056, 3136, 56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_262, y_36, sub_36, mul_72, x_263], Original ATen: [aten.convolution, aten.avg_pool2d, aten.sub, aten.mul, aten.add]
        triton_poi_fused_add_avg_pool2d_convolution_mul_sub_6.run(buf12, buf2, arg1_1, arg5_1, buf14, 25088, 96, grid=grid(25088, 96), stream=stream0)
        del arg1_1
        del arg5_1
        del buf12
        buf15 = buf8; del buf8  # reuse
        buf16 = buf7; del buf7  # reuse
        buf17 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [group_norm_73], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_7.run(buf14, buf15, buf16, buf17, 296, 8137, grid=grid(296), stream=stream0)
        buf18 = buf9; del buf9  # reuse
        buf19 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [group_norm_73], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf15, buf16, buf17, buf18, buf19, 8, 37, grid=grid(8), stream=stream0)
        buf21 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [group_norm_73], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_8.run(buf14, buf18, buf19, arg6_1, arg7_1, buf21, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg6_1
        del arg7_1
        # Topologically Sorted Source Nodes: [group_norm_73, x_264], Original ATen: [aten.native_group_norm, aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg8_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 384, 56, 56), (1204224, 1, 21504, 384))
        del arg8_1
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [group_norm_73, x_264, x_265], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_9.run(buf23, arg9_1, 9633792, grid=grid(9633792), stream=stream0)
        del arg9_1
        # Topologically Sorted Source Nodes: [group_norm_73, x_264, x_265, x_267], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf24 = extern_kernels.convolution(buf23, arg10_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del arg10_1
        del buf23
        buf25 = buf5; del buf5  # reuse
        buf26 = buf4; del buf4  # reuse
        buf27 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [group_norm_74], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_10.run(buf14, buf24, arg11_1, arg12_1, buf25, buf26, buf27, 18944, 128, grid=grid(18944), stream=stream0)
        buf28 = buf17; del buf17  # reuse
        buf29 = buf16; del buf16  # reuse
        buf30 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [group_norm_74], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf25, buf26, buf27, buf28, buf29, buf30, 296, 64, grid=grid(296), stream=stream0)
        buf31 = buf19; del buf19  # reuse
        buf32 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [group_norm_74], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf28, buf29, buf30, buf31, buf32, 8, 37, grid=grid(8), stream=stream0)
        buf34 = reinterpret_tensor(buf21, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [group_norm_74], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_11.run(buf14, buf24, arg11_1, arg12_1, buf31, buf32, arg13_1, arg14_1, buf34, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg13_1
        del arg14_1
        buf36 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [group_norm_73, x_264, x_265, x_267, mul_73, x_269, y_37, sub_37, mul_74, x_270], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_12.run(buf36, buf34, buf24, arg11_1, arg12_1, arg15_1, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg11_1
        del arg12_1
        del arg15_1
        del buf24
        buf37 = buf30; del buf30  # reuse
        buf38 = buf29; del buf29  # reuse
        buf39 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [group_norm_75], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_7.run(buf36, buf37, buf38, buf39, 296, 8137, grid=grid(296), stream=stream0)
        buf40 = buf32; del buf32  # reuse
        buf41 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [group_norm_75], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf37, buf38, buf39, buf40, buf41, 8, 37, grid=grid(8), stream=stream0)
        buf43 = reinterpret_tensor(buf34, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [group_norm_75], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_8.run(buf36, buf40, buf41, arg16_1, arg17_1, buf43, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg16_1
        del arg17_1
        # Topologically Sorted Source Nodes: [group_norm_75, x_271], Original ATen: [aten.native_group_norm, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 384, 56, 56), (1204224, 1, 21504, 384))
        del arg18_1
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [group_norm_75, x_271, x_272], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_9.run(buf45, arg19_1, 9633792, grid=grid(9633792), stream=stream0)
        del arg19_1
        # Topologically Sorted Source Nodes: [group_norm_75, x_271, x_272, x_274], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf46 = extern_kernels.convolution(buf45, arg20_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del arg20_1
        del buf45
        buf47 = buf27; del buf27  # reuse
        buf48 = buf26; del buf26  # reuse
        buf49 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [group_norm_76], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_10.run(buf36, buf46, arg21_1, arg22_1, buf47, buf48, buf49, 18944, 128, grid=grid(18944), stream=stream0)
        buf50 = buf39; del buf39  # reuse
        buf51 = buf38; del buf38  # reuse
        buf52 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [group_norm_76], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf47, buf48, buf49, buf50, buf51, buf52, 296, 64, grid=grid(296), stream=stream0)
        buf53 = buf41; del buf41  # reuse
        buf54 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [group_norm_76], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf50, buf51, buf52, buf53, buf54, 8, 37, grid=grid(8), stream=stream0)
        buf56 = reinterpret_tensor(buf43, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [group_norm_76], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_11.run(buf36, buf46, arg21_1, arg22_1, buf53, buf54, arg23_1, arg24_1, buf56, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg23_1
        del arg24_1
        buf58 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [group_norm_75, x_271, x_272, x_274, mul_75, x_276, y_38, sub_38, mul_76, x_277], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_12.run(buf58, buf56, buf46, arg21_1, arg22_1, arg25_1, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg21_1
        del arg22_1
        del arg25_1
        del buf46
        buf59 = buf52; del buf52  # reuse
        buf60 = buf51; del buf51  # reuse
        buf61 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [group_norm_77], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_7.run(buf58, buf59, buf60, buf61, 296, 8137, grid=grid(296), stream=stream0)
        buf62 = buf54; del buf54  # reuse
        buf63 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [group_norm_77], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf59, buf60, buf61, buf62, buf63, 8, 37, grid=grid(8), stream=stream0)
        buf65 = reinterpret_tensor(buf56, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [group_norm_77], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_8.run(buf58, buf62, buf63, arg26_1, arg27_1, buf65, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg26_1
        del arg27_1
        # Topologically Sorted Source Nodes: [group_norm_77, x_278], Original ATen: [aten.native_group_norm, aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 384, 56, 56), (1204224, 1, 21504, 384))
        del arg28_1
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [group_norm_77, x_278, x_279], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_9.run(buf67, arg29_1, 9633792, grid=grid(9633792), stream=stream0)
        del arg29_1
        # Topologically Sorted Source Nodes: [group_norm_77, x_278, x_279, x_281], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf68 = extern_kernels.convolution(buf67, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del arg30_1
        del buf67
        buf69 = buf49; del buf49  # reuse
        buf70 = buf48; del buf48  # reuse
        buf71 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [group_norm_78], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_10.run(buf58, buf68, arg31_1, arg32_1, buf69, buf70, buf71, 18944, 128, grid=grid(18944), stream=stream0)
        buf72 = buf61; del buf61  # reuse
        buf73 = buf60; del buf60  # reuse
        buf74 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [group_norm_78], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf69, buf70, buf71, buf72, buf73, buf74, 296, 64, grid=grid(296), stream=stream0)
        buf75 = buf63; del buf63  # reuse
        buf76 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [group_norm_78], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf72, buf73, buf74, buf75, buf76, 8, 37, grid=grid(8), stream=stream0)
        buf78 = reinterpret_tensor(buf65, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [group_norm_78], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_11.run(buf58, buf68, arg31_1, arg32_1, buf75, buf76, arg33_1, arg34_1, buf78, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg33_1
        del arg34_1
        buf80 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [group_norm_77, x_278, x_279, x_281, mul_77, x_283, y_39, sub_39, mul_78, x_284], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_12.run(buf80, buf78, buf68, arg31_1, arg32_1, arg35_1, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg31_1
        del arg32_1
        del arg35_1
        del buf68
        buf81 = buf74; del buf74  # reuse
        buf82 = buf73; del buf73  # reuse
        buf83 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [group_norm_79], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_7.run(buf80, buf81, buf82, buf83, 296, 8137, grid=grid(296), stream=stream0)
        buf84 = buf76; del buf76  # reuse
        buf85 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [group_norm_79], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf81, buf82, buf83, buf84, buf85, 8, 37, grid=grid(8), stream=stream0)
        buf87 = reinterpret_tensor(buf78, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [group_norm_79], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_8.run(buf80, buf84, buf85, arg36_1, arg37_1, buf87, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg36_1
        del arg37_1
        # Topologically Sorted Source Nodes: [group_norm_79, x_285], Original ATen: [aten.native_group_norm, aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg38_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 384, 56, 56), (1204224, 1, 21504, 384))
        del arg38_1
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [group_norm_79, x_285, x_286], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_9.run(buf89, arg39_1, 9633792, grid=grid(9633792), stream=stream0)
        del arg39_1
        # Topologically Sorted Source Nodes: [group_norm_79, x_285, x_286, x_288], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf90 = extern_kernels.convolution(buf89, arg40_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del arg40_1
        del buf89
        buf91 = buf71; del buf71  # reuse
        buf92 = buf70; del buf70  # reuse
        buf93 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [group_norm_80], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_10.run(buf80, buf90, arg41_1, arg42_1, buf91, buf92, buf93, 18944, 128, grid=grid(18944), stream=stream0)
        buf94 = buf83; del buf83  # reuse
        buf95 = buf82; del buf82  # reuse
        buf96 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [group_norm_80], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf91, buf92, buf93, buf94, buf95, buf96, 296, 64, grid=grid(296), stream=stream0)
        buf97 = buf85; del buf85  # reuse
        buf98 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [group_norm_80], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf94, buf95, buf96, buf97, buf98, 8, 37, grid=grid(8), stream=stream0)
        buf100 = reinterpret_tensor(buf87, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [group_norm_80], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_11.run(buf80, buf90, arg41_1, arg42_1, buf97, buf98, arg43_1, arg44_1, buf100, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg43_1
        del arg44_1
        buf102 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [group_norm_79, x_285, x_286, x_288, mul_79, x_290, y_40, sub_40, mul_80, x_291], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_12.run(buf102, buf100, buf90, arg41_1, arg42_1, arg45_1, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg41_1
        del arg42_1
        del arg45_1
        del buf100
        buf103 = buf96; del buf96  # reuse
        buf104 = buf95; del buf95  # reuse
        buf105 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [group_norm_81], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_7.run(buf102, buf103, buf104, buf105, 296, 8137, grid=grid(296), stream=stream0)
        buf106 = buf98; del buf98  # reuse
        buf107 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [group_norm_81], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf103, buf104, buf105, buf106, buf107, 8, 37, grid=grid(8), stream=stream0)
        buf109 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [group_norm_81], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_8.run(buf102, buf106, buf107, arg46_1, arg47_1, buf109, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg46_1
        del arg47_1
        # Topologically Sorted Source Nodes: [group_norm_81, x_292], Original ATen: [aten.native_group_norm, aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 384, 56, 56), (1204224, 1, 21504, 384))
        del arg48_1
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [group_norm_81, x_292, x_293], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_9.run(buf111, arg49_1, 9633792, grid=grid(9633792), stream=stream0)
        del arg49_1
        # Topologically Sorted Source Nodes: [group_norm_81, x_292, x_293, x_295], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf112 = extern_kernels.convolution(buf111, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del arg50_1
        del buf111
        buf113 = buf93; del buf93  # reuse
        buf114 = buf92; del buf92  # reuse
        buf115 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [group_norm_82], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_10.run(buf102, buf112, arg51_1, arg52_1, buf113, buf114, buf115, 18944, 128, grid=grid(18944), stream=stream0)
        buf116 = buf105; del buf105  # reuse
        buf117 = buf104; del buf104  # reuse
        buf118 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [group_norm_82], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf113, buf114, buf115, buf116, buf117, buf118, 296, 64, grid=grid(296), stream=stream0)
        del buf113
        del buf114
        del buf115
        buf119 = buf107; del buf107  # reuse
        buf120 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [group_norm_82], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf116, buf117, buf118, buf119, buf120, 8, 37, grid=grid(8), stream=stream0)
        buf122 = reinterpret_tensor(buf109, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [group_norm_82], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_11.run(buf102, buf112, arg51_1, arg52_1, buf119, buf120, arg53_1, arg54_1, buf122, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg53_1
        del arg54_1
        buf124 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [group_norm_81, x_292, x_293, x_295, mul_81, x_297, y_41, sub_41, mul_82, x_298], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_12.run(buf124, buf122, buf112, arg51_1, arg52_1, arg55_1, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg51_1
        del arg52_1
        del arg55_1
        del buf112
        buf125 = buf118; del buf118  # reuse
        buf126 = buf117; del buf117  # reuse
        buf127 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [group_norm_83], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_7.run(buf124, buf125, buf126, buf127, 296, 8137, grid=grid(296), stream=stream0)
        buf128 = buf120; del buf120  # reuse
        buf129 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [group_norm_83], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_4.run(buf125, buf126, buf127, buf128, buf129, 8, 37, grid=grid(8), stream=stream0)
        del buf125
        del buf126
        del buf127
        buf131 = reinterpret_tensor(buf122, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [group_norm_83], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_8.run(buf124, buf128, buf129, arg56_1, arg57_1, buf131, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg56_1
        del arg57_1
        # Topologically Sorted Source Nodes: [group_norm_83, x_299], Original ATen: [aten.native_group_norm, aten.convolution]
        buf132 = extern_kernels.convolution(buf131, arg58_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 384, 56, 56), (1204224, 1, 21504, 384))
        del arg58_1
        del buf131
        buf133 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [group_norm_83, x_299, x_300], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_9.run(buf133, arg59_1, 9633792, grid=grid(9633792), stream=stream0)
        del arg59_1
        # Topologically Sorted Source Nodes: [group_norm_83, x_299, x_300, x_302], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf134 = extern_kernels.convolution(buf133, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del arg60_1
        del buf133
        buf135 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [group_norm_83, x_299, x_300, x_302, mul_83, x_304], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
        triton_poi_fused_add_convolution_gelu_mul_native_group_norm_13.run(buf135, buf124, arg61_1, arg62_1, 25088, 96, grid=grid(25088, 96), stream=stream0)
        del arg61_1
        del arg62_1
        del buf124
        buf136 = empty_strided_cuda((192, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_83, x_299, x_300, x_302, mul_83, x_304, x_305], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
        triton_poi_fused_add_convolution_gelu_mul_native_group_norm_14.run(arg63_1, buf136, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg63_1
        # Topologically Sorted Source Nodes: [group_norm_83, x_299, x_300, x_302, mul_83, x_304, x_305], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
        buf137 = extern_kernels.convolution(buf135, buf136, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del buf135
        del buf136
        buf138 = empty_strided_cuda((8, 1, 1, 1, 19, 62), (1184, 9472, 9472, 9472, 1, 19), torch.float32)
        buf139 = empty_strided_cuda((8, 1, 1, 1, 19, 62), (1184, 9472, 9472, 9472, 1, 19), torch.float32)
        buf140 = empty_strided_cuda((8, 1, 1, 1, 19, 62), (1184, 9472, 9472, 9472, 1, 19), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_84], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_15.run(buf137, arg64_1, buf138, buf139, buf140, 9424, 128, grid=grid(9424), stream=stream0)
        buf141 = empty_strided_cuda((8, 1, 1, 1, 19), (19, 152, 152, 152, 1), torch.float32)
        buf142 = empty_strided_cuda((8, 1, 1, 1, 19), (19, 152, 152, 152, 1), torch.float32)
        buf143 = empty_strided_cuda((8, 1, 1, 1, 19), (19, 152, 152, 152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_84], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf138, buf139, buf140, buf141, buf142, buf143, 152, 62, grid=grid(152), stream=stream0)
        buf144 = buf129; del buf129  # reuse
        buf145 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [group_norm_84], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_17.run(buf141, buf142, buf143, buf144, buf145, 8, 19, grid=grid(8), stream=stream0)
        buf147 = reinterpret_tensor(buf0, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [group_norm_84], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_18.run(buf137, arg64_1, buf144, buf145, arg65_1, arg66_1, buf147, 1204224, grid=grid(1204224), stream=stream0)
        del arg65_1
        del arg66_1
        buf149 = empty_strided_cuda((8, 192, 28, 28), (150528, 784, 28, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_83, x_299, x_300, x_302, mul_83, x_304, x_305, y_42, sub_42, mul_84, x_306], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_19.run(buf147, buf137, arg64_1, arg67_1, buf149, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg64_1
        del arg67_1
        del buf137
        buf150 = buf143; del buf143  # reuse
        buf151 = buf142; del buf142  # reuse
        buf152 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [group_norm_85], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_20.run(buf149, buf150, buf151, buf152, 152, 7923, grid=grid(152), stream=stream0)
        buf153 = buf145; del buf145  # reuse
        buf154 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [group_norm_85], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_17.run(buf150, buf151, buf152, buf153, buf154, 8, 19, grid=grid(8), stream=stream0)
        buf156 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [group_norm_85], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_21.run(buf149, buf153, buf154, arg68_1, arg69_1, buf156, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg68_1
        del arg69_1
        # Topologically Sorted Source Nodes: [group_norm_85, x_307], Original ATen: [aten.native_group_norm, aten.convolution]
        buf157 = extern_kernels.convolution(buf156, arg70_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg70_1
        buf158 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [group_norm_85, x_307, x_308], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_22.run(buf158, arg71_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg71_1
        # Topologically Sorted Source Nodes: [group_norm_85, x_307, x_308, x_310], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf159 = extern_kernels.convolution(buf158, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg72_1
        del buf158
        buf160 = reinterpret_tensor(buf140, (8, 1, 1, 1, 19, 62), (1184, 9472, 9472, 9472, 62, 1), 0); del buf140  # reuse
        buf161 = reinterpret_tensor(buf139, (8, 1, 1, 1, 19, 62), (1184, 9472, 9472, 9472, 62, 1), 0); del buf139  # reuse
        buf162 = reinterpret_tensor(buf138, (8, 1, 1, 1, 19, 62), (1184, 9472, 9472, 9472, 62, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [group_norm_86], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_23.run(buf149, buf159, arg73_1, arg74_1, buf160, buf161, buf162, 9424, 128, grid=grid(9424), stream=stream0)
        buf163 = buf152; del buf152  # reuse
        buf164 = buf151; del buf151  # reuse
        buf165 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [group_norm_86], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf160, buf161, buf162, buf163, buf164, buf165, 152, 62, grid=grid(152), stream=stream0)
        buf166 = buf154; del buf154  # reuse
        buf167 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [group_norm_86], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_17.run(buf163, buf164, buf165, buf166, buf167, 8, 19, grid=grid(8), stream=stream0)
        buf169 = reinterpret_tensor(buf156, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [group_norm_86], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_25.run(buf149, buf159, arg73_1, arg74_1, buf166, buf167, arg75_1, arg76_1, buf169, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg75_1
        del arg76_1
        buf171 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [group_norm_85, x_307, x_308, x_310, mul_85, x_312, y_43, sub_43, mul_86, x_313], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_26.run(buf171, buf169, buf159, arg73_1, arg74_1, arg77_1, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg73_1
        del arg74_1
        del arg77_1
        del buf159
        buf172 = buf165; del buf165  # reuse
        buf173 = buf164; del buf164  # reuse
        buf174 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [group_norm_87], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_20.run(buf171, buf172, buf173, buf174, 152, 7923, grid=grid(152), stream=stream0)
        buf175 = buf167; del buf167  # reuse
        buf176 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [group_norm_87], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_17.run(buf172, buf173, buf174, buf175, buf176, 8, 19, grid=grid(8), stream=stream0)
        buf178 = reinterpret_tensor(buf169, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [group_norm_87], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_21.run(buf171, buf175, buf176, arg78_1, arg79_1, buf178, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg78_1
        del arg79_1
        # Topologically Sorted Source Nodes: [group_norm_87, x_314], Original ATen: [aten.native_group_norm, aten.convolution]
        buf179 = extern_kernels.convolution(buf178, arg80_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg80_1
        buf180 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [group_norm_87, x_314, x_315], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_22.run(buf180, arg81_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg81_1
        # Topologically Sorted Source Nodes: [group_norm_87, x_314, x_315, x_317], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf181 = extern_kernels.convolution(buf180, arg82_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg82_1
        del buf180
        buf182 = buf162; del buf162  # reuse
        buf183 = buf161; del buf161  # reuse
        buf184 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [group_norm_88], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_23.run(buf171, buf181, arg83_1, arg84_1, buf182, buf183, buf184, 9424, 128, grid=grid(9424), stream=stream0)
        buf185 = buf174; del buf174  # reuse
        buf186 = buf173; del buf173  # reuse
        buf187 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [group_norm_88], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf182, buf183, buf184, buf185, buf186, buf187, 152, 62, grid=grid(152), stream=stream0)
        buf188 = buf176; del buf176  # reuse
        buf189 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [group_norm_88], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_17.run(buf185, buf186, buf187, buf188, buf189, 8, 19, grid=grid(8), stream=stream0)
        buf191 = reinterpret_tensor(buf178, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [group_norm_88], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_25.run(buf171, buf181, arg83_1, arg84_1, buf188, buf189, arg85_1, arg86_1, buf191, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg85_1
        del arg86_1
        buf193 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [group_norm_87, x_314, x_315, x_317, mul_87, x_319, y_44, sub_44, mul_88, x_320], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_26.run(buf193, buf191, buf181, arg83_1, arg84_1, arg87_1, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg83_1
        del arg84_1
        del arg87_1
        del buf181
        buf194 = buf187; del buf187  # reuse
        buf195 = buf186; del buf186  # reuse
        buf196 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [group_norm_89], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_20.run(buf193, buf194, buf195, buf196, 152, 7923, grid=grid(152), stream=stream0)
        buf197 = buf189; del buf189  # reuse
        buf198 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [group_norm_89], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_17.run(buf194, buf195, buf196, buf197, buf198, 8, 19, grid=grid(8), stream=stream0)
        buf200 = reinterpret_tensor(buf191, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [group_norm_89], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_21.run(buf193, buf197, buf198, arg88_1, arg89_1, buf200, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg88_1
        del arg89_1
        # Topologically Sorted Source Nodes: [group_norm_89, x_321], Original ATen: [aten.native_group_norm, aten.convolution]
        buf201 = extern_kernels.convolution(buf200, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg90_1
        buf202 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [group_norm_89, x_321, x_322], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_22.run(buf202, arg91_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg91_1
        # Topologically Sorted Source Nodes: [group_norm_89, x_321, x_322, x_324], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf203 = extern_kernels.convolution(buf202, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg92_1
        del buf202
        buf204 = buf184; del buf184  # reuse
        buf205 = buf183; del buf183  # reuse
        buf206 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [group_norm_90], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_23.run(buf193, buf203, arg93_1, arg94_1, buf204, buf205, buf206, 9424, 128, grid=grid(9424), stream=stream0)
        buf207 = buf196; del buf196  # reuse
        buf208 = buf195; del buf195  # reuse
        buf209 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [group_norm_90], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf204, buf205, buf206, buf207, buf208, buf209, 152, 62, grid=grid(152), stream=stream0)
        buf210 = buf198; del buf198  # reuse
        buf211 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [group_norm_90], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_17.run(buf207, buf208, buf209, buf210, buf211, 8, 19, grid=grid(8), stream=stream0)
        buf213 = reinterpret_tensor(buf200, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [group_norm_90], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_25.run(buf193, buf203, arg93_1, arg94_1, buf210, buf211, arg95_1, arg96_1, buf213, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg95_1
        del arg96_1
        buf215 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [group_norm_89, x_321, x_322, x_324, mul_89, x_326, y_45, sub_45, mul_90, x_327], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_26.run(buf215, buf213, buf203, arg93_1, arg94_1, arg97_1, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg93_1
        del arg94_1
        del arg97_1
        del buf203
        buf216 = buf209; del buf209  # reuse
        buf217 = buf208; del buf208  # reuse
        buf218 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [group_norm_91], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_20.run(buf215, buf216, buf217, buf218, 152, 7923, grid=grid(152), stream=stream0)
        buf219 = buf211; del buf211  # reuse
        buf220 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [group_norm_91], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_17.run(buf216, buf217, buf218, buf219, buf220, 8, 19, grid=grid(8), stream=stream0)
        buf222 = reinterpret_tensor(buf213, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [group_norm_91], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_21.run(buf215, buf219, buf220, arg98_1, arg99_1, buf222, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg98_1
        del arg99_1
        # Topologically Sorted Source Nodes: [group_norm_91, x_328], Original ATen: [aten.native_group_norm, aten.convolution]
        buf223 = extern_kernels.convolution(buf222, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg100_1
        buf224 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [group_norm_91, x_328, x_329], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_22.run(buf224, arg101_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg101_1
        # Topologically Sorted Source Nodes: [group_norm_91, x_328, x_329, x_331], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf225 = extern_kernels.convolution(buf224, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg102_1
        del buf224
        buf226 = buf206; del buf206  # reuse
        buf227 = buf205; del buf205  # reuse
        buf228 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [group_norm_92], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_23.run(buf215, buf225, arg103_1, arg104_1, buf226, buf227, buf228, 9424, 128, grid=grid(9424), stream=stream0)
        buf229 = buf218; del buf218  # reuse
        buf230 = buf217; del buf217  # reuse
        buf231 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [group_norm_92], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf226, buf227, buf228, buf229, buf230, buf231, 152, 62, grid=grid(152), stream=stream0)
        buf232 = buf220; del buf220  # reuse
        buf233 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [group_norm_92], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_17.run(buf229, buf230, buf231, buf232, buf233, 8, 19, grid=grid(8), stream=stream0)
        buf235 = reinterpret_tensor(buf222, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [group_norm_92], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_25.run(buf215, buf225, arg103_1, arg104_1, buf232, buf233, arg105_1, arg106_1, buf235, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg105_1
        del arg106_1
        buf237 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [group_norm_91, x_328, x_329, x_331, mul_91, x_333, y_46, sub_46, mul_92, x_334], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_26.run(buf237, buf235, buf225, arg103_1, arg104_1, arg107_1, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg103_1
        del arg104_1
        del arg107_1
        del buf225
        buf238 = buf231; del buf231  # reuse
        buf239 = buf230; del buf230  # reuse
        buf240 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [group_norm_93], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_20.run(buf237, buf238, buf239, buf240, 152, 7923, grid=grid(152), stream=stream0)
        buf241 = buf233; del buf233  # reuse
        buf242 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [group_norm_93], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_17.run(buf238, buf239, buf240, buf241, buf242, 8, 19, grid=grid(8), stream=stream0)
        buf244 = reinterpret_tensor(buf235, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [group_norm_93], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_21.run(buf237, buf241, buf242, arg108_1, arg109_1, buf244, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg108_1
        del arg109_1
        # Topologically Sorted Source Nodes: [group_norm_93, x_335], Original ATen: [aten.native_group_norm, aten.convolution]
        buf245 = extern_kernels.convolution(buf244, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg110_1
        buf246 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [group_norm_93, x_335, x_336], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_22.run(buf246, arg111_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg111_1
        # Topologically Sorted Source Nodes: [group_norm_93, x_335, x_336, x_338], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf247 = extern_kernels.convolution(buf246, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg112_1
        del buf246
        buf248 = buf228; del buf228  # reuse
        buf249 = buf227; del buf227  # reuse
        buf250 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [group_norm_94], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_23.run(buf237, buf247, arg113_1, arg114_1, buf248, buf249, buf250, 9424, 128, grid=grid(9424), stream=stream0)
        buf251 = buf240; del buf240  # reuse
        buf252 = buf239; del buf239  # reuse
        buf253 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [group_norm_94], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_24.run(buf248, buf249, buf250, buf251, buf252, buf253, 152, 62, grid=grid(152), stream=stream0)
        del buf248
        del buf249
        del buf250
        buf254 = buf242; del buf242  # reuse
        buf255 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [group_norm_94], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_17.run(buf251, buf252, buf253, buf254, buf255, 8, 19, grid=grid(8), stream=stream0)
        buf257 = reinterpret_tensor(buf244, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [group_norm_94], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_25.run(buf237, buf247, arg113_1, arg114_1, buf254, buf255, arg115_1, arg116_1, buf257, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg115_1
        del arg116_1
        buf259 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [group_norm_93, x_335, x_336, x_338, mul_93, x_340, y_47, sub_47, mul_94, x_341], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_26.run(buf259, buf257, buf247, arg113_1, arg114_1, arg117_1, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg113_1
        del arg114_1
        del arg117_1
        del buf247
        buf260 = buf253; del buf253  # reuse
        buf261 = buf252; del buf252  # reuse
        buf262 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [group_norm_95], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_20.run(buf259, buf260, buf261, buf262, 152, 7923, grid=grid(152), stream=stream0)
        buf263 = buf255; del buf255  # reuse
        buf264 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [group_norm_95], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_17.run(buf260, buf261, buf262, buf263, buf264, 8, 19, grid=grid(8), stream=stream0)
        del buf260
        del buf261
        del buf262
        buf266 = reinterpret_tensor(buf257, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [group_norm_95], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_21.run(buf259, buf263, buf264, arg118_1, arg119_1, buf266, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg118_1
        del arg119_1
        # Topologically Sorted Source Nodes: [group_norm_95, x_342], Original ATen: [aten.native_group_norm, aten.convolution]
        buf267 = extern_kernels.convolution(buf266, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg120_1
        del buf266
        buf268 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [group_norm_95, x_342, x_343], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_22.run(buf268, arg121_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg121_1
        # Topologically Sorted Source Nodes: [group_norm_95, x_342, x_343, x_345], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf269 = extern_kernels.convolution(buf268, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg122_1
        del buf268
        buf270 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [group_norm_95, x_342, x_343, x_345, mul_95, x_347], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
        triton_poi_fused_add_convolution_gelu_mul_native_group_norm_27.run(buf270, buf259, arg123_1, arg124_1, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg123_1
        del arg124_1
        del buf259
        buf271 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_95, x_342, x_343, x_345, mul_95, x_347, x_348], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
        triton_poi_fused_add_convolution_gelu_mul_native_group_norm_28.run(arg125_1, buf271, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del arg125_1
        # Topologically Sorted Source Nodes: [group_norm_95, x_342, x_343, x_345, mul_95, x_347, x_348], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
        buf272 = extern_kernels.convolution(buf270, buf271, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del buf270
        del buf271
        buf273 = empty_strided_cuda((8, 1, 1, 1, 10, 59), (590, 4736, 4736, 4736, 1, 10), torch.float32)
        buf274 = empty_strided_cuda((8, 1, 1, 1, 10, 59), (590, 4736, 4736, 4736, 1, 10), torch.float32)
        buf275 = empty_strided_cuda((8, 1, 1, 1, 10, 59), (590, 4736, 4736, 4736, 1, 10), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_96], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_29.run(buf272, arg126_1, buf273, buf274, buf275, 4720, 128, grid=grid(4720), stream=stream0)
        buf276 = empty_strided_cuda((8, 1, 1, 1, 10), (10, 80, 80, 80, 1), torch.float32)
        buf277 = empty_strided_cuda((8, 1, 1, 1, 10), (10, 80, 80, 80, 1), torch.float32)
        buf278 = empty_strided_cuda((8, 1, 1, 1, 10), (10, 80, 80, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_96], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_30.run(buf273, buf274, buf275, buf276, buf277, buf278, 80, 59, grid=grid(80), stream=stream0)
        buf279 = buf264; del buf264  # reuse
        buf280 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [group_norm_96], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf276, buf277, buf278, buf279, buf280, 8, 10, grid=grid(8), stream=stream0)
        buf282 = empty_strided_cuda((8, 384, 14, 14), (75264, 1, 5376, 384), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_96], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf272, arg126_1, buf279, buf280, arg127_1, arg128_1, buf282, 602112, grid=grid(602112), stream=stream0)
        del arg127_1
        del arg128_1
        buf284 = empty_strided_cuda((8, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_95, x_342, x_343, x_345, mul_95, x_347, x_348, y_48, sub_48, mul_96, x_349], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_33.run(buf282, buf272, arg126_1, arg129_1, buf284, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg126_1
        del arg129_1
        del buf272
        buf285 = buf278; del buf278  # reuse
        buf286 = buf277; del buf277  # reuse
        buf287 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [group_norm_97], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf284, buf285, buf286, buf287, 80, 7527, grid=grid(80), stream=stream0)
        buf288 = buf280; del buf280  # reuse
        buf289 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [group_norm_97], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf285, buf286, buf287, buf288, buf289, 8, 10, grid=grid(8), stream=stream0)
        buf291 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [group_norm_97], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf284, buf288, buf289, arg130_1, arg131_1, buf291, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg130_1
        del arg131_1
        # Topologically Sorted Source Nodes: [group_norm_97, x_350], Original ATen: [aten.native_group_norm, aten.convolution]
        buf292 = extern_kernels.convolution(buf291, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg132_1
        buf293 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [group_norm_97, x_350, x_351], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf293, arg133_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg133_1
        # Topologically Sorted Source Nodes: [group_norm_97, x_350, x_351, x_353], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf294 = extern_kernels.convolution(buf293, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg134_1
        del buf293
        buf295 = reinterpret_tensor(buf275, (8, 1, 1, 1, 10, 59), (590, 4736, 4736, 4736, 59, 1), 0); del buf275  # reuse
        buf296 = reinterpret_tensor(buf274, (8, 1, 1, 1, 10, 59), (590, 4736, 4736, 4736, 59, 1), 0); del buf274  # reuse
        buf297 = reinterpret_tensor(buf273, (8, 1, 1, 1, 10, 59), (590, 4736, 4736, 4736, 59, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [group_norm_98], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf284, buf294, arg135_1, arg136_1, buf295, buf296, buf297, 4720, 128, grid=grid(4720), stream=stream0)
        buf298 = buf287; del buf287  # reuse
        buf299 = buf286; del buf286  # reuse
        buf300 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [group_norm_98], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf295, buf296, buf297, buf298, buf299, buf300, 80, 59, grid=grid(80), stream=stream0)
        buf301 = buf289; del buf289  # reuse
        buf302 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [group_norm_98], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf298, buf299, buf300, buf301, buf302, 8, 10, grid=grid(8), stream=stream0)
        buf304 = reinterpret_tensor(buf291, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [group_norm_98], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf284, buf294, arg135_1, arg136_1, buf301, buf302, arg137_1, arg138_1, buf304, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg137_1
        del arg138_1
        buf306 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [group_norm_97, x_350, x_351, x_353, mul_97, x_355, y_49, sub_49, mul_98, x_356], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf306, buf304, buf294, arg135_1, arg136_1, arg139_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg135_1
        del arg136_1
        del arg139_1
        del buf294
        buf307 = buf300; del buf300  # reuse
        buf308 = buf299; del buf299  # reuse
        buf309 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [group_norm_99], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf306, buf307, buf308, buf309, 80, 7527, grid=grid(80), stream=stream0)
        buf310 = buf302; del buf302  # reuse
        buf311 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [group_norm_99], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf307, buf308, buf309, buf310, buf311, 8, 10, grid=grid(8), stream=stream0)
        buf313 = reinterpret_tensor(buf304, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [group_norm_99], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf306, buf310, buf311, arg140_1, arg141_1, buf313, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg140_1
        del arg141_1
        # Topologically Sorted Source Nodes: [group_norm_99, x_357], Original ATen: [aten.native_group_norm, aten.convolution]
        buf314 = extern_kernels.convolution(buf313, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg142_1
        buf315 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [group_norm_99, x_357, x_358], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf315, arg143_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg143_1
        # Topologically Sorted Source Nodes: [group_norm_99, x_357, x_358, x_360], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf316 = extern_kernels.convolution(buf315, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg144_1
        del buf315
        buf317 = buf297; del buf297  # reuse
        buf318 = buf296; del buf296  # reuse
        buf319 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [group_norm_100], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf306, buf316, arg145_1, arg146_1, buf317, buf318, buf319, 4720, 128, grid=grid(4720), stream=stream0)
        buf320 = buf309; del buf309  # reuse
        buf321 = buf308; del buf308  # reuse
        buf322 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [group_norm_100], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf317, buf318, buf319, buf320, buf321, buf322, 80, 59, grid=grid(80), stream=stream0)
        buf323 = buf311; del buf311  # reuse
        buf324 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [group_norm_100], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf320, buf321, buf322, buf323, buf324, 8, 10, grid=grid(8), stream=stream0)
        buf326 = reinterpret_tensor(buf313, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [group_norm_100], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf306, buf316, arg145_1, arg146_1, buf323, buf324, arg147_1, arg148_1, buf326, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg147_1
        del arg148_1
        buf328 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [group_norm_99, x_357, x_358, x_360, mul_99, x_362, y_50, sub_50, mul_100, x_363], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf328, buf326, buf316, arg145_1, arg146_1, arg149_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg145_1
        del arg146_1
        del arg149_1
        del buf316
        buf329 = buf322; del buf322  # reuse
        buf330 = buf321; del buf321  # reuse
        buf331 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [group_norm_101], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf328, buf329, buf330, buf331, 80, 7527, grid=grid(80), stream=stream0)
        buf332 = buf324; del buf324  # reuse
        buf333 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [group_norm_101], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf329, buf330, buf331, buf332, buf333, 8, 10, grid=grid(8), stream=stream0)
        buf335 = reinterpret_tensor(buf326, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [group_norm_101], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf328, buf332, buf333, arg150_1, arg151_1, buf335, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg150_1
        del arg151_1
        # Topologically Sorted Source Nodes: [group_norm_101, x_364], Original ATen: [aten.native_group_norm, aten.convolution]
        buf336 = extern_kernels.convolution(buf335, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg152_1
        buf337 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [group_norm_101, x_364, x_365], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf337, arg153_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg153_1
        # Topologically Sorted Source Nodes: [group_norm_101, x_364, x_365, x_367], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf338 = extern_kernels.convolution(buf337, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg154_1
        del buf337
        buf339 = buf319; del buf319  # reuse
        buf340 = buf318; del buf318  # reuse
        buf341 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [group_norm_102], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf328, buf338, arg155_1, arg156_1, buf339, buf340, buf341, 4720, 128, grid=grid(4720), stream=stream0)
        buf342 = buf331; del buf331  # reuse
        buf343 = buf330; del buf330  # reuse
        buf344 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [group_norm_102], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf339, buf340, buf341, buf342, buf343, buf344, 80, 59, grid=grid(80), stream=stream0)
        buf345 = buf333; del buf333  # reuse
        buf346 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [group_norm_102], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf342, buf343, buf344, buf345, buf346, 8, 10, grid=grid(8), stream=stream0)
        buf348 = reinterpret_tensor(buf335, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [group_norm_102], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf328, buf338, arg155_1, arg156_1, buf345, buf346, arg157_1, arg158_1, buf348, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg157_1
        del arg158_1
        buf350 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [group_norm_101, x_364, x_365, x_367, mul_101, x_369, y_51, sub_51, mul_102, x_370], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf350, buf348, buf338, arg155_1, arg156_1, arg159_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg155_1
        del arg156_1
        del arg159_1
        del buf338
        buf351 = buf344; del buf344  # reuse
        buf352 = buf343; del buf343  # reuse
        buf353 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [group_norm_103], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf350, buf351, buf352, buf353, 80, 7527, grid=grid(80), stream=stream0)
        buf354 = buf346; del buf346  # reuse
        buf355 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [group_norm_103], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf351, buf352, buf353, buf354, buf355, 8, 10, grid=grid(8), stream=stream0)
        buf357 = reinterpret_tensor(buf348, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf348  # reuse
        # Topologically Sorted Source Nodes: [group_norm_103], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf350, buf354, buf355, arg160_1, arg161_1, buf357, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg160_1
        del arg161_1
        # Topologically Sorted Source Nodes: [group_norm_103, x_371], Original ATen: [aten.native_group_norm, aten.convolution]
        buf358 = extern_kernels.convolution(buf357, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg162_1
        buf359 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [group_norm_103, x_371, x_372], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf359, arg163_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg163_1
        # Topologically Sorted Source Nodes: [group_norm_103, x_371, x_372, x_374], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf360 = extern_kernels.convolution(buf359, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg164_1
        del buf359
        buf361 = buf341; del buf341  # reuse
        buf362 = buf340; del buf340  # reuse
        buf363 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [group_norm_104], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf350, buf360, arg165_1, arg166_1, buf361, buf362, buf363, 4720, 128, grid=grid(4720), stream=stream0)
        buf364 = buf353; del buf353  # reuse
        buf365 = buf352; del buf352  # reuse
        buf366 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [group_norm_104], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf361, buf362, buf363, buf364, buf365, buf366, 80, 59, grid=grid(80), stream=stream0)
        buf367 = buf355; del buf355  # reuse
        buf368 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [group_norm_104], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf364, buf365, buf366, buf367, buf368, 8, 10, grid=grid(8), stream=stream0)
        buf370 = reinterpret_tensor(buf357, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [group_norm_104], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf350, buf360, arg165_1, arg166_1, buf367, buf368, arg167_1, arg168_1, buf370, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg167_1
        del arg168_1
        buf372 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [group_norm_103, x_371, x_372, x_374, mul_103, x_376, y_52, sub_52, mul_104, x_377], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf372, buf370, buf360, arg165_1, arg166_1, arg169_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg165_1
        del arg166_1
        del arg169_1
        del buf360
        buf373 = buf366; del buf366  # reuse
        buf374 = buf365; del buf365  # reuse
        buf375 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [group_norm_105], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf372, buf373, buf374, buf375, 80, 7527, grid=grid(80), stream=stream0)
        buf376 = buf368; del buf368  # reuse
        buf377 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [group_norm_105], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf373, buf374, buf375, buf376, buf377, 8, 10, grid=grid(8), stream=stream0)
        buf379 = reinterpret_tensor(buf370, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [group_norm_105], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf372, buf376, buf377, arg170_1, arg171_1, buf379, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg170_1
        del arg171_1
        # Topologically Sorted Source Nodes: [group_norm_105, x_378], Original ATen: [aten.native_group_norm, aten.convolution]
        buf380 = extern_kernels.convolution(buf379, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg172_1
        buf381 = buf380; del buf380  # reuse
        # Topologically Sorted Source Nodes: [group_norm_105, x_378, x_379], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf381, arg173_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg173_1
        # Topologically Sorted Source Nodes: [group_norm_105, x_378, x_379, x_381], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf382 = extern_kernels.convolution(buf381, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg174_1
        del buf381
        buf383 = buf363; del buf363  # reuse
        buf384 = buf362; del buf362  # reuse
        buf385 = buf361; del buf361  # reuse
        # Topologically Sorted Source Nodes: [group_norm_106], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf372, buf382, arg175_1, arg176_1, buf383, buf384, buf385, 4720, 128, grid=grid(4720), stream=stream0)
        buf386 = buf375; del buf375  # reuse
        buf387 = buf374; del buf374  # reuse
        buf388 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [group_norm_106], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf383, buf384, buf385, buf386, buf387, buf388, 80, 59, grid=grid(80), stream=stream0)
        buf389 = buf377; del buf377  # reuse
        buf390 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [group_norm_106], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf386, buf387, buf388, buf389, buf390, 8, 10, grid=grid(8), stream=stream0)
        buf392 = reinterpret_tensor(buf379, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [group_norm_106], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf372, buf382, arg175_1, arg176_1, buf389, buf390, arg177_1, arg178_1, buf392, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg177_1
        del arg178_1
        buf394 = buf372; del buf372  # reuse
        # Topologically Sorted Source Nodes: [group_norm_105, x_378, x_379, x_381, mul_105, x_383, y_53, sub_53, mul_106, x_384], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf394, buf392, buf382, arg175_1, arg176_1, arg179_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg175_1
        del arg176_1
        del arg179_1
        del buf382
        buf395 = buf388; del buf388  # reuse
        buf396 = buf387; del buf387  # reuse
        buf397 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [group_norm_107], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf394, buf395, buf396, buf397, 80, 7527, grid=grid(80), stream=stream0)
        buf398 = buf390; del buf390  # reuse
        buf399 = buf389; del buf389  # reuse
        # Topologically Sorted Source Nodes: [group_norm_107], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf395, buf396, buf397, buf398, buf399, 8, 10, grid=grid(8), stream=stream0)
        buf401 = reinterpret_tensor(buf392, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [group_norm_107], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf394, buf398, buf399, arg180_1, arg181_1, buf401, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg180_1
        del arg181_1
        # Topologically Sorted Source Nodes: [group_norm_107, x_385], Original ATen: [aten.native_group_norm, aten.convolution]
        buf402 = extern_kernels.convolution(buf401, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg182_1
        buf403 = buf402; del buf402  # reuse
        # Topologically Sorted Source Nodes: [group_norm_107, x_385, x_386], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf403, arg183_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg183_1
        # Topologically Sorted Source Nodes: [group_norm_107, x_385, x_386, x_388], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf404 = extern_kernels.convolution(buf403, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg184_1
        del buf403
        buf405 = buf385; del buf385  # reuse
        buf406 = buf384; del buf384  # reuse
        buf407 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [group_norm_108], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf394, buf404, arg185_1, arg186_1, buf405, buf406, buf407, 4720, 128, grid=grid(4720), stream=stream0)
        buf408 = buf397; del buf397  # reuse
        buf409 = buf396; del buf396  # reuse
        buf410 = buf395; del buf395  # reuse
        # Topologically Sorted Source Nodes: [group_norm_108], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf405, buf406, buf407, buf408, buf409, buf410, 80, 59, grid=grid(80), stream=stream0)
        buf411 = buf399; del buf399  # reuse
        buf412 = buf398; del buf398  # reuse
        # Topologically Sorted Source Nodes: [group_norm_108], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf408, buf409, buf410, buf411, buf412, 8, 10, grid=grid(8), stream=stream0)
        buf414 = reinterpret_tensor(buf401, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [group_norm_108], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf394, buf404, arg185_1, arg186_1, buf411, buf412, arg187_1, arg188_1, buf414, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg187_1
        del arg188_1
        buf416 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [group_norm_107, x_385, x_386, x_388, mul_107, x_390, y_54, sub_54, mul_108, x_391], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf416, buf414, buf404, arg185_1, arg186_1, arg189_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg185_1
        del arg186_1
        del arg189_1
        del buf404
        buf417 = buf410; del buf410  # reuse
        buf418 = buf409; del buf409  # reuse
        buf419 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [group_norm_109], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf416, buf417, buf418, buf419, 80, 7527, grid=grid(80), stream=stream0)
        buf420 = buf412; del buf412  # reuse
        buf421 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [group_norm_109], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf417, buf418, buf419, buf420, buf421, 8, 10, grid=grid(8), stream=stream0)
        buf423 = reinterpret_tensor(buf414, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [group_norm_109], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf416, buf420, buf421, arg190_1, arg191_1, buf423, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg190_1
        del arg191_1
        # Topologically Sorted Source Nodes: [group_norm_109, x_392], Original ATen: [aten.native_group_norm, aten.convolution]
        buf424 = extern_kernels.convolution(buf423, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg192_1
        buf425 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [group_norm_109, x_392, x_393], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf425, arg193_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg193_1
        # Topologically Sorted Source Nodes: [group_norm_109, x_392, x_393, x_395], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf426 = extern_kernels.convolution(buf425, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg194_1
        del buf425
        buf427 = buf407; del buf407  # reuse
        buf428 = buf406; del buf406  # reuse
        buf429 = buf405; del buf405  # reuse
        # Topologically Sorted Source Nodes: [group_norm_110], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf416, buf426, arg195_1, arg196_1, buf427, buf428, buf429, 4720, 128, grid=grid(4720), stream=stream0)
        buf430 = buf419; del buf419  # reuse
        buf431 = buf418; del buf418  # reuse
        buf432 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [group_norm_110], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf427, buf428, buf429, buf430, buf431, buf432, 80, 59, grid=grid(80), stream=stream0)
        buf433 = buf421; del buf421  # reuse
        buf434 = buf420; del buf420  # reuse
        # Topologically Sorted Source Nodes: [group_norm_110], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf430, buf431, buf432, buf433, buf434, 8, 10, grid=grid(8), stream=stream0)
        buf436 = reinterpret_tensor(buf423, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [group_norm_110], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf416, buf426, arg195_1, arg196_1, buf433, buf434, arg197_1, arg198_1, buf436, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg197_1
        del arg198_1
        buf438 = buf416; del buf416  # reuse
        # Topologically Sorted Source Nodes: [group_norm_109, x_392, x_393, x_395, mul_109, x_397, y_55, sub_55, mul_110, x_398], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf438, buf436, buf426, arg195_1, arg196_1, arg199_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg195_1
        del arg196_1
        del arg199_1
        del buf426
        buf439 = buf432; del buf432  # reuse
        buf440 = buf431; del buf431  # reuse
        buf441 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [group_norm_111], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf438, buf439, buf440, buf441, 80, 7527, grid=grid(80), stream=stream0)
        buf442 = buf434; del buf434  # reuse
        buf443 = buf433; del buf433  # reuse
        # Topologically Sorted Source Nodes: [group_norm_111], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf439, buf440, buf441, buf442, buf443, 8, 10, grid=grid(8), stream=stream0)
        buf445 = reinterpret_tensor(buf436, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [group_norm_111], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf438, buf442, buf443, arg200_1, arg201_1, buf445, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg200_1
        del arg201_1
        # Topologically Sorted Source Nodes: [group_norm_111, x_399], Original ATen: [aten.native_group_norm, aten.convolution]
        buf446 = extern_kernels.convolution(buf445, arg202_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg202_1
        buf447 = buf446; del buf446  # reuse
        # Topologically Sorted Source Nodes: [group_norm_111, x_399, x_400], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf447, arg203_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg203_1
        # Topologically Sorted Source Nodes: [group_norm_111, x_399, x_400, x_402], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf448 = extern_kernels.convolution(buf447, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg204_1
        del buf447
        buf449 = buf429; del buf429  # reuse
        buf450 = buf428; del buf428  # reuse
        buf451 = buf427; del buf427  # reuse
        # Topologically Sorted Source Nodes: [group_norm_112], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf438, buf448, arg205_1, arg206_1, buf449, buf450, buf451, 4720, 128, grid=grid(4720), stream=stream0)
        buf452 = buf441; del buf441  # reuse
        buf453 = buf440; del buf440  # reuse
        buf454 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [group_norm_112], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf449, buf450, buf451, buf452, buf453, buf454, 80, 59, grid=grid(80), stream=stream0)
        buf455 = buf443; del buf443  # reuse
        buf456 = buf442; del buf442  # reuse
        # Topologically Sorted Source Nodes: [group_norm_112], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf452, buf453, buf454, buf455, buf456, 8, 10, grid=grid(8), stream=stream0)
        buf458 = reinterpret_tensor(buf445, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf445  # reuse
        # Topologically Sorted Source Nodes: [group_norm_112], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf438, buf448, arg205_1, arg206_1, buf455, buf456, arg207_1, arg208_1, buf458, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg207_1
        del arg208_1
        buf460 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [group_norm_111, x_399, x_400, x_402, mul_111, x_404, y_56, sub_56, mul_112, x_405], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf460, buf458, buf448, arg205_1, arg206_1, arg209_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg205_1
        del arg206_1
        del arg209_1
        del buf448
        buf461 = buf454; del buf454  # reuse
        buf462 = buf453; del buf453  # reuse
        buf463 = buf452; del buf452  # reuse
        # Topologically Sorted Source Nodes: [group_norm_113], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf460, buf461, buf462, buf463, 80, 7527, grid=grid(80), stream=stream0)
        buf464 = buf456; del buf456  # reuse
        buf465 = buf455; del buf455  # reuse
        # Topologically Sorted Source Nodes: [group_norm_113], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf461, buf462, buf463, buf464, buf465, 8, 10, grid=grid(8), stream=stream0)
        buf467 = reinterpret_tensor(buf458, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf458  # reuse
        # Topologically Sorted Source Nodes: [group_norm_113], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf460, buf464, buf465, arg210_1, arg211_1, buf467, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg210_1
        del arg211_1
        # Topologically Sorted Source Nodes: [group_norm_113, x_406], Original ATen: [aten.native_group_norm, aten.convolution]
        buf468 = extern_kernels.convolution(buf467, arg212_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf468, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg212_1
        buf469 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [group_norm_113, x_406, x_407], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf469, arg213_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg213_1
        # Topologically Sorted Source Nodes: [group_norm_113, x_406, x_407, x_409], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf470 = extern_kernels.convolution(buf469, arg214_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg214_1
        del buf469
        buf471 = buf451; del buf451  # reuse
        buf472 = buf450; del buf450  # reuse
        buf473 = buf449; del buf449  # reuse
        # Topologically Sorted Source Nodes: [group_norm_114], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf460, buf470, arg215_1, arg216_1, buf471, buf472, buf473, 4720, 128, grid=grid(4720), stream=stream0)
        buf474 = buf463; del buf463  # reuse
        buf475 = buf462; del buf462  # reuse
        buf476 = buf461; del buf461  # reuse
        # Topologically Sorted Source Nodes: [group_norm_114], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf471, buf472, buf473, buf474, buf475, buf476, 80, 59, grid=grid(80), stream=stream0)
        buf477 = buf465; del buf465  # reuse
        buf478 = buf464; del buf464  # reuse
        # Topologically Sorted Source Nodes: [group_norm_114], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf474, buf475, buf476, buf477, buf478, 8, 10, grid=grid(8), stream=stream0)
        buf480 = reinterpret_tensor(buf467, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [group_norm_114], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf460, buf470, arg215_1, arg216_1, buf477, buf478, arg217_1, arg218_1, buf480, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg217_1
        del arg218_1
        buf482 = buf460; del buf460  # reuse
        # Topologically Sorted Source Nodes: [group_norm_113, x_406, x_407, x_409, mul_113, x_411, y_57, sub_57, mul_114, x_412], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf482, buf480, buf470, arg215_1, arg216_1, arg219_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg215_1
        del arg216_1
        del arg219_1
        del buf470
        buf483 = buf476; del buf476  # reuse
        buf484 = buf475; del buf475  # reuse
        buf485 = buf474; del buf474  # reuse
        # Topologically Sorted Source Nodes: [group_norm_115], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf482, buf483, buf484, buf485, 80, 7527, grid=grid(80), stream=stream0)
        buf486 = buf478; del buf478  # reuse
        buf487 = buf477; del buf477  # reuse
        # Topologically Sorted Source Nodes: [group_norm_115], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf483, buf484, buf485, buf486, buf487, 8, 10, grid=grid(8), stream=stream0)
        buf489 = reinterpret_tensor(buf480, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [group_norm_115], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf482, buf486, buf487, arg220_1, arg221_1, buf489, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg220_1
        del arg221_1
        # Topologically Sorted Source Nodes: [group_norm_115, x_413], Original ATen: [aten.native_group_norm, aten.convolution]
        buf490 = extern_kernels.convolution(buf489, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg222_1
        buf491 = buf490; del buf490  # reuse
        # Topologically Sorted Source Nodes: [group_norm_115, x_413, x_414], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf491, arg223_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg223_1
        # Topologically Sorted Source Nodes: [group_norm_115, x_413, x_414, x_416], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf492 = extern_kernels.convolution(buf491, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf492, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg224_1
        del buf491
        buf493 = buf473; del buf473  # reuse
        buf494 = buf472; del buf472  # reuse
        buf495 = buf471; del buf471  # reuse
        # Topologically Sorted Source Nodes: [group_norm_116], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf482, buf492, arg225_1, arg226_1, buf493, buf494, buf495, 4720, 128, grid=grid(4720), stream=stream0)
        buf496 = buf485; del buf485  # reuse
        buf497 = buf484; del buf484  # reuse
        buf498 = buf483; del buf483  # reuse
        # Topologically Sorted Source Nodes: [group_norm_116], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf493, buf494, buf495, buf496, buf497, buf498, 80, 59, grid=grid(80), stream=stream0)
        buf499 = buf487; del buf487  # reuse
        buf500 = buf486; del buf486  # reuse
        # Topologically Sorted Source Nodes: [group_norm_116], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf496, buf497, buf498, buf499, buf500, 8, 10, grid=grid(8), stream=stream0)
        buf502 = reinterpret_tensor(buf489, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [group_norm_116], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf482, buf492, arg225_1, arg226_1, buf499, buf500, arg227_1, arg228_1, buf502, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg227_1
        del arg228_1
        buf504 = buf482; del buf482  # reuse
        # Topologically Sorted Source Nodes: [group_norm_115, x_413, x_414, x_416, mul_115, x_418, y_58, sub_58, mul_116, x_419], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf504, buf502, buf492, arg225_1, arg226_1, arg229_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg225_1
        del arg226_1
        del arg229_1
        del buf492
        buf505 = buf498; del buf498  # reuse
        buf506 = buf497; del buf497  # reuse
        buf507 = buf496; del buf496  # reuse
        # Topologically Sorted Source Nodes: [group_norm_117], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf504, buf505, buf506, buf507, 80, 7527, grid=grid(80), stream=stream0)
        buf508 = buf500; del buf500  # reuse
        buf509 = buf499; del buf499  # reuse
        # Topologically Sorted Source Nodes: [group_norm_117], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf505, buf506, buf507, buf508, buf509, 8, 10, grid=grid(8), stream=stream0)
        buf511 = reinterpret_tensor(buf502, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [group_norm_117], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf504, buf508, buf509, arg230_1, arg231_1, buf511, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg230_1
        del arg231_1
        # Topologically Sorted Source Nodes: [group_norm_117, x_420], Original ATen: [aten.native_group_norm, aten.convolution]
        buf512 = extern_kernels.convolution(buf511, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf512, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg232_1
        buf513 = buf512; del buf512  # reuse
        # Topologically Sorted Source Nodes: [group_norm_117, x_420, x_421], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf513, arg233_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg233_1
        # Topologically Sorted Source Nodes: [group_norm_117, x_420, x_421, x_423], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf514 = extern_kernels.convolution(buf513, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf514, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg234_1
        del buf513
        buf515 = buf495; del buf495  # reuse
        buf516 = buf494; del buf494  # reuse
        buf517 = buf493; del buf493  # reuse
        # Topologically Sorted Source Nodes: [group_norm_118], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf504, buf514, arg235_1, arg236_1, buf515, buf516, buf517, 4720, 128, grid=grid(4720), stream=stream0)
        buf518 = buf507; del buf507  # reuse
        buf519 = buf506; del buf506  # reuse
        buf520 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [group_norm_118], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf515, buf516, buf517, buf518, buf519, buf520, 80, 59, grid=grid(80), stream=stream0)
        buf521 = buf509; del buf509  # reuse
        buf522 = buf508; del buf508  # reuse
        # Topologically Sorted Source Nodes: [group_norm_118], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf518, buf519, buf520, buf521, buf522, 8, 10, grid=grid(8), stream=stream0)
        buf524 = reinterpret_tensor(buf511, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [group_norm_118], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf504, buf514, arg235_1, arg236_1, buf521, buf522, arg237_1, arg238_1, buf524, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg237_1
        del arg238_1
        buf526 = buf504; del buf504  # reuse
        # Topologically Sorted Source Nodes: [group_norm_117, x_420, x_421, x_423, mul_117, x_425, y_59, sub_59, mul_118, x_426], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf526, buf524, buf514, arg235_1, arg236_1, arg239_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg235_1
        del arg236_1
        del arg239_1
        del buf514
        buf527 = buf520; del buf520  # reuse
        buf528 = buf519; del buf519  # reuse
        buf529 = buf518; del buf518  # reuse
        # Topologically Sorted Source Nodes: [group_norm_119], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf526, buf527, buf528, buf529, 80, 7527, grid=grid(80), stream=stream0)
        buf530 = buf522; del buf522  # reuse
        buf531 = buf521; del buf521  # reuse
        # Topologically Sorted Source Nodes: [group_norm_119], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf527, buf528, buf529, buf530, buf531, 8, 10, grid=grid(8), stream=stream0)
        buf533 = reinterpret_tensor(buf524, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [group_norm_119], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf526, buf530, buf531, arg240_1, arg241_1, buf533, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg240_1
        del arg241_1
        # Topologically Sorted Source Nodes: [group_norm_119, x_427], Original ATen: [aten.native_group_norm, aten.convolution]
        buf534 = extern_kernels.convolution(buf533, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg242_1
        buf535 = buf534; del buf534  # reuse
        # Topologically Sorted Source Nodes: [group_norm_119, x_427, x_428], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf535, arg243_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg243_1
        # Topologically Sorted Source Nodes: [group_norm_119, x_427, x_428, x_430], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf536 = extern_kernels.convolution(buf535, arg244_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg244_1
        del buf535
        buf537 = buf517; del buf517  # reuse
        buf538 = buf516; del buf516  # reuse
        buf539 = buf515; del buf515  # reuse
        # Topologically Sorted Source Nodes: [group_norm_120], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf526, buf536, arg245_1, arg246_1, buf537, buf538, buf539, 4720, 128, grid=grid(4720), stream=stream0)
        buf540 = buf529; del buf529  # reuse
        buf541 = buf528; del buf528  # reuse
        buf542 = buf527; del buf527  # reuse
        # Topologically Sorted Source Nodes: [group_norm_120], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf537, buf538, buf539, buf540, buf541, buf542, 80, 59, grid=grid(80), stream=stream0)
        buf543 = buf531; del buf531  # reuse
        buf544 = buf530; del buf530  # reuse
        # Topologically Sorted Source Nodes: [group_norm_120], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf540, buf541, buf542, buf543, buf544, 8, 10, grid=grid(8), stream=stream0)
        buf546 = reinterpret_tensor(buf533, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf533  # reuse
        # Topologically Sorted Source Nodes: [group_norm_120], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf526, buf536, arg245_1, arg246_1, buf543, buf544, arg247_1, arg248_1, buf546, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg247_1
        del arg248_1
        buf548 = buf526; del buf526  # reuse
        # Topologically Sorted Source Nodes: [group_norm_119, x_427, x_428, x_430, mul_119, x_432, y_60, sub_60, mul_120, x_433], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf548, buf546, buf536, arg245_1, arg246_1, arg249_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg245_1
        del arg246_1
        del arg249_1
        del buf536
        buf549 = buf542; del buf542  # reuse
        buf550 = buf541; del buf541  # reuse
        buf551 = buf540; del buf540  # reuse
        # Topologically Sorted Source Nodes: [group_norm_121], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf548, buf549, buf550, buf551, 80, 7527, grid=grid(80), stream=stream0)
        buf552 = buf544; del buf544  # reuse
        buf553 = buf543; del buf543  # reuse
        # Topologically Sorted Source Nodes: [group_norm_121], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf549, buf550, buf551, buf552, buf553, 8, 10, grid=grid(8), stream=stream0)
        buf555 = reinterpret_tensor(buf546, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf546  # reuse
        # Topologically Sorted Source Nodes: [group_norm_121], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf548, buf552, buf553, arg250_1, arg251_1, buf555, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg250_1
        del arg251_1
        # Topologically Sorted Source Nodes: [group_norm_121, x_434], Original ATen: [aten.native_group_norm, aten.convolution]
        buf556 = extern_kernels.convolution(buf555, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf556, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg252_1
        buf557 = buf556; del buf556  # reuse
        # Topologically Sorted Source Nodes: [group_norm_121, x_434, x_435], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf557, arg253_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg253_1
        # Topologically Sorted Source Nodes: [group_norm_121, x_434, x_435, x_437], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf558 = extern_kernels.convolution(buf557, arg254_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf558, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg254_1
        del buf557
        buf559 = buf539; del buf539  # reuse
        buf560 = buf538; del buf538  # reuse
        buf561 = buf537; del buf537  # reuse
        # Topologically Sorted Source Nodes: [group_norm_122], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf548, buf558, arg255_1, arg256_1, buf559, buf560, buf561, 4720, 128, grid=grid(4720), stream=stream0)
        buf562 = buf551; del buf551  # reuse
        buf563 = buf550; del buf550  # reuse
        buf564 = buf549; del buf549  # reuse
        # Topologically Sorted Source Nodes: [group_norm_122], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf559, buf560, buf561, buf562, buf563, buf564, 80, 59, grid=grid(80), stream=stream0)
        buf565 = buf553; del buf553  # reuse
        buf566 = buf552; del buf552  # reuse
        # Topologically Sorted Source Nodes: [group_norm_122], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf562, buf563, buf564, buf565, buf566, 8, 10, grid=grid(8), stream=stream0)
        buf568 = reinterpret_tensor(buf555, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf555  # reuse
        # Topologically Sorted Source Nodes: [group_norm_122], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf548, buf558, arg255_1, arg256_1, buf565, buf566, arg257_1, arg258_1, buf568, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg257_1
        del arg258_1
        buf570 = buf548; del buf548  # reuse
        # Topologically Sorted Source Nodes: [group_norm_121, x_434, x_435, x_437, mul_121, x_439, y_61, sub_61, mul_122, x_440], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf570, buf568, buf558, arg255_1, arg256_1, arg259_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg255_1
        del arg256_1
        del arg259_1
        del buf558
        buf571 = buf564; del buf564  # reuse
        buf572 = buf563; del buf563  # reuse
        buf573 = buf562; del buf562  # reuse
        # Topologically Sorted Source Nodes: [group_norm_123], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf570, buf571, buf572, buf573, 80, 7527, grid=grid(80), stream=stream0)
        buf574 = buf566; del buf566  # reuse
        buf575 = buf565; del buf565  # reuse
        # Topologically Sorted Source Nodes: [group_norm_123], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf571, buf572, buf573, buf574, buf575, 8, 10, grid=grid(8), stream=stream0)
        buf577 = reinterpret_tensor(buf568, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf568  # reuse
        # Topologically Sorted Source Nodes: [group_norm_123], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf570, buf574, buf575, arg260_1, arg261_1, buf577, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg260_1
        del arg261_1
        # Topologically Sorted Source Nodes: [group_norm_123, x_441], Original ATen: [aten.native_group_norm, aten.convolution]
        buf578 = extern_kernels.convolution(buf577, arg262_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf578, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg262_1
        buf579 = buf578; del buf578  # reuse
        # Topologically Sorted Source Nodes: [group_norm_123, x_441, x_442], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf579, arg263_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg263_1
        # Topologically Sorted Source Nodes: [group_norm_123, x_441, x_442, x_444], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf580 = extern_kernels.convolution(buf579, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf580, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg264_1
        del buf579
        buf581 = buf561; del buf561  # reuse
        buf582 = buf560; del buf560  # reuse
        buf583 = buf559; del buf559  # reuse
        # Topologically Sorted Source Nodes: [group_norm_124], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf570, buf580, arg265_1, arg266_1, buf581, buf582, buf583, 4720, 128, grid=grid(4720), stream=stream0)
        buf584 = buf573; del buf573  # reuse
        buf585 = buf572; del buf572  # reuse
        buf586 = buf571; del buf571  # reuse
        # Topologically Sorted Source Nodes: [group_norm_124], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf581, buf582, buf583, buf584, buf585, buf586, 80, 59, grid=grid(80), stream=stream0)
        buf587 = buf575; del buf575  # reuse
        buf588 = buf574; del buf574  # reuse
        # Topologically Sorted Source Nodes: [group_norm_124], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf584, buf585, buf586, buf587, buf588, 8, 10, grid=grid(8), stream=stream0)
        buf590 = reinterpret_tensor(buf577, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [group_norm_124], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf570, buf580, arg265_1, arg266_1, buf587, buf588, arg267_1, arg268_1, buf590, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg267_1
        del arg268_1
        buf592 = buf570; del buf570  # reuse
        # Topologically Sorted Source Nodes: [group_norm_123, x_441, x_442, x_444, mul_123, x_446, y_62, sub_62, mul_124, x_447], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf592, buf590, buf580, arg265_1, arg266_1, arg269_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg265_1
        del arg266_1
        del arg269_1
        del buf580
        buf593 = buf586; del buf586  # reuse
        buf594 = buf585; del buf585  # reuse
        buf595 = buf584; del buf584  # reuse
        # Topologically Sorted Source Nodes: [group_norm_125], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf592, buf593, buf594, buf595, 80, 7527, grid=grid(80), stream=stream0)
        buf596 = buf588; del buf588  # reuse
        buf597 = buf587; del buf587  # reuse
        # Topologically Sorted Source Nodes: [group_norm_125], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf593, buf594, buf595, buf596, buf597, 8, 10, grid=grid(8), stream=stream0)
        buf599 = reinterpret_tensor(buf590, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf590  # reuse
        # Topologically Sorted Source Nodes: [group_norm_125], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf592, buf596, buf597, arg270_1, arg271_1, buf599, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg270_1
        del arg271_1
        # Topologically Sorted Source Nodes: [group_norm_125, x_448], Original ATen: [aten.native_group_norm, aten.convolution]
        buf600 = extern_kernels.convolution(buf599, arg272_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf600, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg272_1
        buf601 = buf600; del buf600  # reuse
        # Topologically Sorted Source Nodes: [group_norm_125, x_448, x_449], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf601, arg273_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg273_1
        # Topologically Sorted Source Nodes: [group_norm_125, x_448, x_449, x_451], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf602 = extern_kernels.convolution(buf601, arg274_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf602, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg274_1
        del buf601
        buf603 = buf583; del buf583  # reuse
        buf604 = buf582; del buf582  # reuse
        buf605 = buf581; del buf581  # reuse
        # Topologically Sorted Source Nodes: [group_norm_126], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf592, buf602, arg275_1, arg276_1, buf603, buf604, buf605, 4720, 128, grid=grid(4720), stream=stream0)
        buf606 = buf595; del buf595  # reuse
        buf607 = buf594; del buf594  # reuse
        buf608 = buf593; del buf593  # reuse
        # Topologically Sorted Source Nodes: [group_norm_126], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf603, buf604, buf605, buf606, buf607, buf608, 80, 59, grid=grid(80), stream=stream0)
        buf609 = buf597; del buf597  # reuse
        buf610 = buf596; del buf596  # reuse
        # Topologically Sorted Source Nodes: [group_norm_126], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf606, buf607, buf608, buf609, buf610, 8, 10, grid=grid(8), stream=stream0)
        buf612 = reinterpret_tensor(buf599, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf599  # reuse
        # Topologically Sorted Source Nodes: [group_norm_126], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf592, buf602, arg275_1, arg276_1, buf609, buf610, arg277_1, arg278_1, buf612, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg277_1
        del arg278_1
        buf614 = buf592; del buf592  # reuse
        # Topologically Sorted Source Nodes: [group_norm_125, x_448, x_449, x_451, mul_125, x_453, y_63, sub_63, mul_126, x_454], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf614, buf612, buf602, arg275_1, arg276_1, arg279_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg275_1
        del arg276_1
        del arg279_1
        del buf602
        buf615 = buf608; del buf608  # reuse
        buf616 = buf607; del buf607  # reuse
        buf617 = buf606; del buf606  # reuse
        # Topologically Sorted Source Nodes: [group_norm_127], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf614, buf615, buf616, buf617, 80, 7527, grid=grid(80), stream=stream0)
        buf618 = buf610; del buf610  # reuse
        buf619 = buf609; del buf609  # reuse
        # Topologically Sorted Source Nodes: [group_norm_127], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf615, buf616, buf617, buf618, buf619, 8, 10, grid=grid(8), stream=stream0)
        buf621 = reinterpret_tensor(buf612, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf612  # reuse
        # Topologically Sorted Source Nodes: [group_norm_127], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf614, buf618, buf619, arg280_1, arg281_1, buf621, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg280_1
        del arg281_1
        # Topologically Sorted Source Nodes: [group_norm_127, x_455], Original ATen: [aten.native_group_norm, aten.convolution]
        buf622 = extern_kernels.convolution(buf621, arg282_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf622, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg282_1
        buf623 = buf622; del buf622  # reuse
        # Topologically Sorted Source Nodes: [group_norm_127, x_455, x_456], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf623, arg283_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg283_1
        # Topologically Sorted Source Nodes: [group_norm_127, x_455, x_456, x_458], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf624 = extern_kernels.convolution(buf623, arg284_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf624, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg284_1
        del buf623
        buf625 = buf605; del buf605  # reuse
        buf626 = buf604; del buf604  # reuse
        buf627 = buf603; del buf603  # reuse
        # Topologically Sorted Source Nodes: [group_norm_128], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf614, buf624, arg285_1, arg286_1, buf625, buf626, buf627, 4720, 128, grid=grid(4720), stream=stream0)
        buf628 = buf617; del buf617  # reuse
        buf629 = buf616; del buf616  # reuse
        buf630 = buf615; del buf615  # reuse
        # Topologically Sorted Source Nodes: [group_norm_128], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf625, buf626, buf627, buf628, buf629, buf630, 80, 59, grid=grid(80), stream=stream0)
        buf631 = buf619; del buf619  # reuse
        buf632 = buf618; del buf618  # reuse
        # Topologically Sorted Source Nodes: [group_norm_128], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf628, buf629, buf630, buf631, buf632, 8, 10, grid=grid(8), stream=stream0)
        buf634 = reinterpret_tensor(buf621, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf621  # reuse
        # Topologically Sorted Source Nodes: [group_norm_128], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf614, buf624, arg285_1, arg286_1, buf631, buf632, arg287_1, arg288_1, buf634, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg287_1
        del arg288_1
        buf636 = buf614; del buf614  # reuse
        # Topologically Sorted Source Nodes: [group_norm_127, x_455, x_456, x_458, mul_127, x_460, y_64, sub_64, mul_128, x_461], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf636, buf634, buf624, arg285_1, arg286_1, arg289_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg285_1
        del arg286_1
        del arg289_1
        del buf624
        buf637 = buf630; del buf630  # reuse
        buf638 = buf629; del buf629  # reuse
        buf639 = buf628; del buf628  # reuse
        # Topologically Sorted Source Nodes: [group_norm_129], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf636, buf637, buf638, buf639, 80, 7527, grid=grid(80), stream=stream0)
        buf640 = buf632; del buf632  # reuse
        buf641 = buf631; del buf631  # reuse
        # Topologically Sorted Source Nodes: [group_norm_129], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf637, buf638, buf639, buf640, buf641, 8, 10, grid=grid(8), stream=stream0)
        buf643 = reinterpret_tensor(buf634, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf634  # reuse
        # Topologically Sorted Source Nodes: [group_norm_129], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf636, buf640, buf641, arg290_1, arg291_1, buf643, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg290_1
        del arg291_1
        # Topologically Sorted Source Nodes: [group_norm_129, x_462], Original ATen: [aten.native_group_norm, aten.convolution]
        buf644 = extern_kernels.convolution(buf643, arg292_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf644, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg292_1
        buf645 = buf644; del buf644  # reuse
        # Topologically Sorted Source Nodes: [group_norm_129, x_462, x_463], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf645, arg293_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg293_1
        # Topologically Sorted Source Nodes: [group_norm_129, x_462, x_463, x_465], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf646 = extern_kernels.convolution(buf645, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf646, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg294_1
        del buf645
        buf647 = buf627; del buf627  # reuse
        buf648 = buf626; del buf626  # reuse
        buf649 = buf625; del buf625  # reuse
        # Topologically Sorted Source Nodes: [group_norm_130], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_37.run(buf636, buf646, arg295_1, arg296_1, buf647, buf648, buf649, 4720, 128, grid=grid(4720), stream=stream0)
        buf650 = buf639; del buf639  # reuse
        buf651 = buf638; del buf638  # reuse
        buf652 = buf637; del buf637  # reuse
        # Topologically Sorted Source Nodes: [group_norm_130], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_38.run(buf647, buf648, buf649, buf650, buf651, buf652, 80, 59, grid=grid(80), stream=stream0)
        del buf647
        del buf648
        del buf649
        buf653 = buf641; del buf641  # reuse
        buf654 = buf640; del buf640  # reuse
        # Topologically Sorted Source Nodes: [group_norm_130], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf650, buf651, buf652, buf653, buf654, 8, 10, grid=grid(8), stream=stream0)
        buf656 = reinterpret_tensor(buf643, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf643  # reuse
        # Topologically Sorted Source Nodes: [group_norm_130], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_39.run(buf636, buf646, arg295_1, arg296_1, buf653, buf654, arg297_1, arg298_1, buf656, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg297_1
        del arg298_1
        buf658 = buf636; del buf636  # reuse
        # Topologically Sorted Source Nodes: [group_norm_129, x_462, x_463, x_465, mul_129, x_467, y_65, sub_65, mul_130, x_468], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_40.run(buf658, buf656, buf646, arg295_1, arg296_1, arg299_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg295_1
        del arg296_1
        del arg299_1
        del buf646
        buf659 = buf652; del buf652  # reuse
        buf660 = buf651; del buf651  # reuse
        buf661 = buf650; del buf650  # reuse
        # Topologically Sorted Source Nodes: [group_norm_131], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf658, buf659, buf660, buf661, 80, 7527, grid=grid(80), stream=stream0)
        buf662 = buf654; del buf654  # reuse
        buf663 = buf653; del buf653  # reuse
        # Topologically Sorted Source Nodes: [group_norm_131], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_31.run(buf659, buf660, buf661, buf662, buf663, 8, 10, grid=grid(8), stream=stream0)
        del buf659
        del buf660
        del buf661
        buf665 = reinterpret_tensor(buf656, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf656  # reuse
        # Topologically Sorted Source Nodes: [group_norm_131], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf658, buf662, buf663, arg300_1, arg301_1, buf665, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg300_1
        del arg301_1
        # Topologically Sorted Source Nodes: [group_norm_131, x_469], Original ATen: [aten.native_group_norm, aten.convolution]
        buf666 = extern_kernels.convolution(buf665, arg302_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf666, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg302_1
        del buf665
        buf667 = buf666; del buf666  # reuse
        # Topologically Sorted Source Nodes: [group_norm_131, x_469, x_470], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_36.run(buf667, arg303_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg303_1
        # Topologically Sorted Source Nodes: [group_norm_131, x_469, x_470, x_472], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf668 = extern_kernels.convolution(buf667, arg304_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf668, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg304_1
        del buf667
        buf669 = buf668; del buf668  # reuse
        # Topologically Sorted Source Nodes: [group_norm_131, x_469, x_470, x_472, mul_131, x_474], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
        triton_poi_fused_add_convolution_gelu_mul_native_group_norm_41.run(buf669, buf658, arg305_1, arg306_1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg305_1
        del arg306_1
        del buf658
        buf670 = empty_strided_cuda((768, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_131, x_469, x_470, x_472, mul_131, x_474, x_475], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
        triton_poi_fused_add_convolution_gelu_mul_native_group_norm_42.run(arg307_1, buf670, 294912, 9, grid=grid(294912, 9), stream=stream0)
        del arg307_1
        # Topologically Sorted Source Nodes: [group_norm_131, x_469, x_470, x_472, mul_131, x_474, x_475], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add]
        buf671 = extern_kernels.convolution(buf669, buf670, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf671, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del buf669
        del buf670
        buf672 = empty_strided_cuda((8, 1, 1, 1, 5, 59), (295, 2368, 2368, 2368, 59, 1), torch.float32)
        buf673 = empty_strided_cuda((8, 1, 1, 1, 5, 59), (295, 2368, 2368, 2368, 59, 1), torch.float32)
        buf674 = empty_strided_cuda((8, 1, 1, 1, 5, 59), (295, 2368, 2368, 2368, 59, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_132], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_43.run(buf671, arg308_1, buf672, buf673, buf674, 2360, 128, grid=grid(2360), stream=stream0)
        buf675 = empty_strided_cuda((8, 1, 1, 1, 5), (5, 40, 40, 40, 1), torch.float32)
        buf676 = empty_strided_cuda((8, 1, 1, 1, 5), (5, 40, 40, 40, 1), torch.float32)
        buf677 = empty_strided_cuda((8, 1, 1, 1, 5), (5, 40, 40, 40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_132], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_44.run(buf672, buf673, buf674, buf675, buf676, buf677, 40, 59, grid=grid(40), stream=stream0)
        buf678 = buf663; del buf663  # reuse
        buf679 = buf662; del buf662  # reuse
        # Topologically Sorted Source Nodes: [group_norm_132], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf675, buf676, buf677, buf678, buf679, 8, 5, grid=grid(8), stream=stream0)
        buf681 = empty_strided_cuda((8, 768, 7, 7), (37632, 1, 5376, 768), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_132], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_46.run(buf671, arg308_1, buf678, buf679, arg309_1, arg310_1, buf681, 301056, grid=grid(301056), stream=stream0)
        del arg309_1
        del arg310_1
        buf683 = empty_strided_cuda((8, 768, 7, 7), (37632, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_131, x_469, x_470, x_472, mul_131, x_474, x_475, y_66, sub_66, mul_132, x_476], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_47.run(buf681, buf671, arg308_1, arg311_1, buf683, 392, 768, grid=grid(392, 768), stream=stream0)
        del arg308_1
        del arg311_1
        del buf671
        buf684 = buf677; del buf677  # reuse
        buf685 = buf676; del buf676  # reuse
        buf686 = buf675; del buf675  # reuse
        # Topologically Sorted Source Nodes: [group_norm_133], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_48.run(buf683, buf684, buf685, buf686, 40, 7527, grid=grid(40), stream=stream0)
        buf687 = buf679; del buf679  # reuse
        buf688 = buf678; del buf678  # reuse
        # Topologically Sorted Source Nodes: [group_norm_133], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf684, buf685, buf686, buf687, buf688, 8, 5, grid=grid(8), stream=stream0)
        buf690 = buf681; del buf681  # reuse
        # Topologically Sorted Source Nodes: [group_norm_133], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_49.run(buf683, buf687, buf688, arg312_1, arg313_1, buf690, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg312_1
        del arg313_1
        # Topologically Sorted Source Nodes: [group_norm_133, x_477], Original ATen: [aten.native_group_norm, aten.convolution]
        buf691 = extern_kernels.convolution(buf690, arg314_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf691, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
        del arg314_1
        buf692 = buf691; del buf691  # reuse
        # Topologically Sorted Source Nodes: [group_norm_133, x_477, x_478], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_50.run(buf692, arg315_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg315_1
        # Topologically Sorted Source Nodes: [group_norm_133, x_477, x_478, x_480], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf693 = extern_kernels.convolution(buf692, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf693, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg316_1
        del buf692
        buf694 = buf674; del buf674  # reuse
        buf695 = buf673; del buf673  # reuse
        buf696 = buf672; del buf672  # reuse
        # Topologically Sorted Source Nodes: [group_norm_134], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_51.run(buf683, buf693, arg317_1, arg318_1, buf694, buf695, buf696, 2360, 128, grid=grid(2360), stream=stream0)
        buf697 = buf686; del buf686  # reuse
        buf698 = buf685; del buf685  # reuse
        buf699 = buf684; del buf684  # reuse
        # Topologically Sorted Source Nodes: [group_norm_134], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_44.run(buf694, buf695, buf696, buf697, buf698, buf699, 40, 59, grid=grid(40), stream=stream0)
        buf700 = buf688; del buf688  # reuse
        buf701 = buf687; del buf687  # reuse
        # Topologically Sorted Source Nodes: [group_norm_134], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf697, buf698, buf699, buf700, buf701, 8, 5, grid=grid(8), stream=stream0)
        buf703 = reinterpret_tensor(buf690, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf690  # reuse
        # Topologically Sorted Source Nodes: [group_norm_134], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_52.run(buf683, buf693, arg317_1, arg318_1, buf700, buf701, arg319_1, arg320_1, buf703, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg319_1
        del arg320_1
        buf705 = buf683; del buf683  # reuse
        # Topologically Sorted Source Nodes: [group_norm_133, x_477, x_478, x_480, mul_133, x_482, y_67, sub_67, mul_134, x_483], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_53.run(buf705, buf703, buf693, arg317_1, arg318_1, arg321_1, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg317_1
        del arg318_1
        del arg321_1
        del buf693
        buf706 = buf699; del buf699  # reuse
        buf707 = buf698; del buf698  # reuse
        buf708 = buf697; del buf697  # reuse
        # Topologically Sorted Source Nodes: [group_norm_135], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_48.run(buf705, buf706, buf707, buf708, 40, 7527, grid=grid(40), stream=stream0)
        buf709 = buf701; del buf701  # reuse
        buf710 = buf700; del buf700  # reuse
        # Topologically Sorted Source Nodes: [group_norm_135], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf706, buf707, buf708, buf709, buf710, 8, 5, grid=grid(8), stream=stream0)
        buf712 = reinterpret_tensor(buf703, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf703  # reuse
        # Topologically Sorted Source Nodes: [group_norm_135], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_49.run(buf705, buf709, buf710, arg322_1, arg323_1, buf712, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg322_1
        del arg323_1
        # Topologically Sorted Source Nodes: [group_norm_135, x_484], Original ATen: [aten.native_group_norm, aten.convolution]
        buf713 = extern_kernels.convolution(buf712, arg324_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf713, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
        del arg324_1
        buf714 = buf713; del buf713  # reuse
        # Topologically Sorted Source Nodes: [group_norm_135, x_484, x_485], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_50.run(buf714, arg325_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg325_1
        # Topologically Sorted Source Nodes: [group_norm_135, x_484, x_485, x_487], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf715 = extern_kernels.convolution(buf714, arg326_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf715, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg326_1
        del buf714
        buf716 = buf696; del buf696  # reuse
        buf717 = buf695; del buf695  # reuse
        buf718 = buf694; del buf694  # reuse
        # Topologically Sorted Source Nodes: [group_norm_136], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_51.run(buf705, buf715, arg327_1, arg328_1, buf716, buf717, buf718, 2360, 128, grid=grid(2360), stream=stream0)
        buf719 = buf708; del buf708  # reuse
        buf720 = buf707; del buf707  # reuse
        buf721 = buf706; del buf706  # reuse
        # Topologically Sorted Source Nodes: [group_norm_136], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_44.run(buf716, buf717, buf718, buf719, buf720, buf721, 40, 59, grid=grid(40), stream=stream0)
        buf722 = buf710; del buf710  # reuse
        buf723 = buf709; del buf709  # reuse
        # Topologically Sorted Source Nodes: [group_norm_136], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf719, buf720, buf721, buf722, buf723, 8, 5, grid=grid(8), stream=stream0)
        buf725 = reinterpret_tensor(buf712, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf712  # reuse
        # Topologically Sorted Source Nodes: [group_norm_136], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_52.run(buf705, buf715, arg327_1, arg328_1, buf722, buf723, arg329_1, arg330_1, buf725, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg329_1
        del arg330_1
        buf727 = buf705; del buf705  # reuse
        # Topologically Sorted Source Nodes: [group_norm_135, x_484, x_485, x_487, mul_135, x_489, y_68, sub_68, mul_136, x_490], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_53.run(buf727, buf725, buf715, arg327_1, arg328_1, arg331_1, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg327_1
        del arg328_1
        del arg331_1
        del buf715
        buf728 = buf721; del buf721  # reuse
        buf729 = buf720; del buf720  # reuse
        buf730 = buf719; del buf719  # reuse
        # Topologically Sorted Source Nodes: [group_norm_137], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_48.run(buf727, buf728, buf729, buf730, 40, 7527, grid=grid(40), stream=stream0)
        buf731 = buf723; del buf723  # reuse
        buf732 = buf722; del buf722  # reuse
        # Topologically Sorted Source Nodes: [group_norm_137], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf728, buf729, buf730, buf731, buf732, 8, 5, grid=grid(8), stream=stream0)
        buf734 = reinterpret_tensor(buf725, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf725  # reuse
        # Topologically Sorted Source Nodes: [group_norm_137], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_49.run(buf727, buf731, buf732, arg332_1, arg333_1, buf734, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg332_1
        del arg333_1
        # Topologically Sorted Source Nodes: [group_norm_137, x_491], Original ATen: [aten.native_group_norm, aten.convolution]
        buf735 = extern_kernels.convolution(buf734, arg334_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf735, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
        del arg334_1
        buf736 = buf735; del buf735  # reuse
        # Topologically Sorted Source Nodes: [group_norm_137, x_491, x_492], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_50.run(buf736, arg335_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg335_1
        # Topologically Sorted Source Nodes: [group_norm_137, x_491, x_492, x_494], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf737 = extern_kernels.convolution(buf736, arg336_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf737, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg336_1
        del buf736
        buf738 = buf718; del buf718  # reuse
        buf739 = buf717; del buf717  # reuse
        buf740 = buf716; del buf716  # reuse
        # Topologically Sorted Source Nodes: [group_norm_138], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_51.run(buf727, buf737, arg337_1, arg338_1, buf738, buf739, buf740, 2360, 128, grid=grid(2360), stream=stream0)
        buf741 = buf730; del buf730  # reuse
        buf742 = buf729; del buf729  # reuse
        buf743 = buf728; del buf728  # reuse
        # Topologically Sorted Source Nodes: [group_norm_138], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_44.run(buf738, buf739, buf740, buf741, buf742, buf743, 40, 59, grid=grid(40), stream=stream0)
        buf744 = buf732; del buf732  # reuse
        buf745 = buf731; del buf731  # reuse
        # Topologically Sorted Source Nodes: [group_norm_138], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf741, buf742, buf743, buf744, buf745, 8, 5, grid=grid(8), stream=stream0)
        buf747 = reinterpret_tensor(buf734, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf734  # reuse
        # Topologically Sorted Source Nodes: [group_norm_138], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_52.run(buf727, buf737, arg337_1, arg338_1, buf744, buf745, arg339_1, arg340_1, buf747, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg339_1
        del arg340_1
        buf749 = buf727; del buf727  # reuse
        # Topologically Sorted Source Nodes: [group_norm_137, x_491, x_492, x_494, mul_137, x_496, y_69, sub_69, mul_138, x_497], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_53.run(buf749, buf747, buf737, arg337_1, arg338_1, arg341_1, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg337_1
        del arg338_1
        del arg341_1
        del buf737
        buf750 = buf743; del buf743  # reuse
        buf751 = buf742; del buf742  # reuse
        buf752 = buf741; del buf741  # reuse
        # Topologically Sorted Source Nodes: [group_norm_139], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_48.run(buf749, buf750, buf751, buf752, 40, 7527, grid=grid(40), stream=stream0)
        buf753 = buf745; del buf745  # reuse
        buf754 = buf744; del buf744  # reuse
        # Topologically Sorted Source Nodes: [group_norm_139], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf750, buf751, buf752, buf753, buf754, 8, 5, grid=grid(8), stream=stream0)
        buf756 = reinterpret_tensor(buf747, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf747  # reuse
        # Topologically Sorted Source Nodes: [group_norm_139], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_49.run(buf749, buf753, buf754, arg342_1, arg343_1, buf756, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg342_1
        del arg343_1
        # Topologically Sorted Source Nodes: [group_norm_139, x_498], Original ATen: [aten.native_group_norm, aten.convolution]
        buf757 = extern_kernels.convolution(buf756, arg344_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf757, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
        del arg344_1
        buf758 = buf757; del buf757  # reuse
        # Topologically Sorted Source Nodes: [group_norm_139, x_498, x_499], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_50.run(buf758, arg345_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg345_1
        # Topologically Sorted Source Nodes: [group_norm_139, x_498, x_499, x_501], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf759 = extern_kernels.convolution(buf758, arg346_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf759, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg346_1
        del buf758
        buf760 = buf740; del buf740  # reuse
        buf761 = buf739; del buf739  # reuse
        buf762 = buf738; del buf738  # reuse
        # Topologically Sorted Source Nodes: [group_norm_140], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_51.run(buf749, buf759, arg347_1, arg348_1, buf760, buf761, buf762, 2360, 128, grid=grid(2360), stream=stream0)
        buf763 = buf752; del buf752  # reuse
        buf764 = buf751; del buf751  # reuse
        buf765 = buf750; del buf750  # reuse
        # Topologically Sorted Source Nodes: [group_norm_140], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_44.run(buf760, buf761, buf762, buf763, buf764, buf765, 40, 59, grid=grid(40), stream=stream0)
        buf766 = buf754; del buf754  # reuse
        buf767 = buf753; del buf753  # reuse
        # Topologically Sorted Source Nodes: [group_norm_140], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf763, buf764, buf765, buf766, buf767, 8, 5, grid=grid(8), stream=stream0)
        buf769 = reinterpret_tensor(buf756, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf756  # reuse
        # Topologically Sorted Source Nodes: [group_norm_140], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_52.run(buf749, buf759, arg347_1, arg348_1, buf766, buf767, arg349_1, arg350_1, buf769, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg349_1
        del arg350_1
        buf771 = buf749; del buf749  # reuse
        # Topologically Sorted Source Nodes: [group_norm_139, x_498, x_499, x_501, mul_139, x_503, y_70, sub_70, mul_140, x_504], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_53.run(buf771, buf769, buf759, arg347_1, arg348_1, arg351_1, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg347_1
        del arg348_1
        del arg351_1
        del buf759
        buf772 = buf765; del buf765  # reuse
        buf773 = buf764; del buf764  # reuse
        buf774 = buf763; del buf763  # reuse
        # Topologically Sorted Source Nodes: [group_norm_141], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_48.run(buf771, buf772, buf773, buf774, 40, 7527, grid=grid(40), stream=stream0)
        buf775 = buf767; del buf767  # reuse
        buf776 = buf766; del buf766  # reuse
        # Topologically Sorted Source Nodes: [group_norm_141], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf772, buf773, buf774, buf775, buf776, 8, 5, grid=grid(8), stream=stream0)
        buf778 = reinterpret_tensor(buf769, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf769  # reuse
        # Topologically Sorted Source Nodes: [group_norm_141], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_49.run(buf771, buf775, buf776, arg352_1, arg353_1, buf778, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg352_1
        del arg353_1
        # Topologically Sorted Source Nodes: [group_norm_141, x_505], Original ATen: [aten.native_group_norm, aten.convolution]
        buf779 = extern_kernels.convolution(buf778, arg354_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf779, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
        del arg354_1
        buf780 = buf779; del buf779  # reuse
        # Topologically Sorted Source Nodes: [group_norm_141, x_505, x_506], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_50.run(buf780, arg355_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg355_1
        # Topologically Sorted Source Nodes: [group_norm_141, x_505, x_506, x_508], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf781 = extern_kernels.convolution(buf780, arg356_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf781, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg356_1
        del buf780
        buf782 = buf762; del buf762  # reuse
        buf783 = buf761; del buf761  # reuse
        buf784 = buf760; del buf760  # reuse
        # Topologically Sorted Source Nodes: [group_norm_142], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_51.run(buf771, buf781, arg357_1, arg358_1, buf782, buf783, buf784, 2360, 128, grid=grid(2360), stream=stream0)
        buf785 = buf774; del buf774  # reuse
        buf786 = buf773; del buf773  # reuse
        buf787 = buf772; del buf772  # reuse
        # Topologically Sorted Source Nodes: [group_norm_142], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_44.run(buf782, buf783, buf784, buf785, buf786, buf787, 40, 59, grid=grid(40), stream=stream0)
        del buf782
        del buf783
        del buf784
        buf788 = buf776; del buf776  # reuse
        buf789 = buf775; del buf775  # reuse
        # Topologically Sorted Source Nodes: [group_norm_142], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf785, buf786, buf787, buf788, buf789, 8, 5, grid=grid(8), stream=stream0)
        buf791 = reinterpret_tensor(buf778, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf778  # reuse
        # Topologically Sorted Source Nodes: [group_norm_142], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_52.run(buf771, buf781, arg357_1, arg358_1, buf788, buf789, arg359_1, arg360_1, buf791, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg359_1
        del arg360_1
        buf793 = buf771; del buf771  # reuse
        # Topologically Sorted Source Nodes: [group_norm_141, x_505, x_506, x_508, mul_141, x_510, y_71, sub_71, mul_142, x_511], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.avg_pool2d, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_53.run(buf793, buf791, buf781, arg357_1, arg358_1, arg361_1, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg357_1
        del arg358_1
        del arg361_1
        del buf781
        buf794 = buf787; del buf787  # reuse
        buf795 = buf786; del buf786  # reuse
        buf796 = buf785; del buf785  # reuse
        # Topologically Sorted Source Nodes: [group_norm_143], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_48.run(buf793, buf794, buf795, buf796, 40, 7527, grid=grid(40), stream=stream0)
        buf797 = buf789; del buf789  # reuse
        buf798 = buf788; del buf788  # reuse
        # Topologically Sorted Source Nodes: [group_norm_143], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_45.run(buf794, buf795, buf796, buf797, buf798, 8, 5, grid=grid(8), stream=stream0)
        del buf794
        del buf795
        del buf796
        buf800 = reinterpret_tensor(buf791, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf791  # reuse
        # Topologically Sorted Source Nodes: [group_norm_143], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_49.run(buf793, buf797, buf798, arg362_1, arg363_1, buf800, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg362_1
        del arg363_1
        del buf797
        del buf798
        # Topologically Sorted Source Nodes: [group_norm_143, x_512], Original ATen: [aten.native_group_norm, aten.convolution]
        buf801 = extern_kernels.convolution(buf800, arg364_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf801, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
        del arg364_1
        del buf800
        buf802 = buf801; del buf801  # reuse
        # Topologically Sorted Source Nodes: [group_norm_143, x_512, x_513], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_native_group_norm_50.run(buf802, arg365_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg365_1
        # Topologically Sorted Source Nodes: [group_norm_143, x_512, x_513, x_515], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu]
        buf803 = extern_kernels.convolution(buf802, arg366_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf803, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg366_1
        del buf802
        buf804 = empty_strided_cuda((8, 768, 1, 1), (768, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_143, x_512, x_513, x_515, mul_143, x_517, x_518], Original ATen: [aten.native_group_norm, aten.convolution, aten.gelu, aten.mul, aten.add, aten.mean]
        triton_per_fused_add_convolution_gelu_mean_mul_native_group_norm_54.run(buf793, buf803, arg367_1, arg368_1, buf804, 6144, 49, grid=grid(6144), stream=stream0)
        del arg367_1
        del arg368_1
        del buf793
        del buf803
        buf808 = empty_strided_cuda((8, 1, 1, 768), (768, 1, 6144, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_520], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_55.run(buf804, arg369_1, arg370_1, buf808, 8, 768, grid=grid(8), stream=stream0)
        del arg369_1
        del arg370_1
        del buf804
        buf809 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_523], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg372_1, reinterpret_tensor(buf808, (8, 768), (768, 1), 0), reinterpret_tensor(arg371_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf809)
        del arg371_1
        del arg372_1
        del buf808
    return (buf809, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((96, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((192, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((768, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('poolformer_m36', benchmark_compiled_module)
