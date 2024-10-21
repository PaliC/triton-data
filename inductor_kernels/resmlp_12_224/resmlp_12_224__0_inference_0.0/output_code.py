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


# kernel path: /tmp/torchinductor_sahanp/ra/crahkwohwknyhxey7qd6hmuuniel4fxxklkw63ubva73ma6mqs5o.py
# Topologically Sorted Source Nodes: [addcmul_25], Original ATen: [aten.addcmul]
# Source node to ATen node mapping:
#   addcmul_25 => add_73, mul_110, mul_111
# Graph fragment:
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg5_1, 1), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_110, %permute_62), kwargs = {})
#   %add_73 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, %mul_111), kwargs = {})
triton_poi_fused_addcmul_0 = async_compile.triton('triton_poi_fused_addcmul_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addcmul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addcmul_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp8, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wl/cwl576w4ebkcar4hl4om3bbtj2e2vaq4mlsa23rcn62smtow5vjv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_default_23 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_74, %permute_64), kwargs = {})
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 3072
    x1 = (xindex // 3072)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((384*x1) + (75264*(x0 // 384)) + (x0 % 384)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/z6/cz6tydytodktenk3r4cuktu76n5z5k5f7uqoivqr2rdcebcisirk.py
# Topologically Sorted Source Nodes: [mul_24, x_92], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_24 => mul_112
#   x_92 => add_74
# Graph fragment:
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, %permute_65), kwargs = {})
#   %add_74 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_62, %mul_112), kwargs = {})
triton_poi_fused_add_mul_2 = async_compile.triton('triton_poi_fused_add_mul_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 196) % 384
    x0 = xindex % 196
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp2 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vu/cvuqrw4nhdnh5sbscvbhm34brt5nbawy6lpovmyppext2twess5y.py
# Topologically Sorted Source Nodes: [addcmul_26, x_93], Original ATen: [aten.addcmul, aten.clone]
# Source node to ATen node mapping:
#   addcmul_26 => add_75, mul_113, mul_114
#   x_93 => clone_37
# Graph fragment:
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg10_1, 1), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_113, %add_74), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg9_1, %mul_114), kwargs = {})
#   %clone_37 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_75,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_addcmul_clone_3 = async_compile.triton('triton_poi_fused_addcmul_clone_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addcmul_clone_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addcmul_clone_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sn/csns3tescziuicxdmb4s7oxhn7sud44w4gf7uma3sijnhfshgpw6.py
# Topologically Sorted Source Nodes: [x_93, x_94], Original ATen: [aten.add, aten.gelu]
# Source node to ATen node mapping:
#   x_93 => add_76
#   x_94 => add_77, erf_12, mul_115, mul_116, mul_117
# Graph fragment:
#   %add_76 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_77, %arg12_1), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, 0.5), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, 0.7071067811865476), kwargs = {})
#   %erf_12 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_116,), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_12, 1), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %add_77), kwargs = {})
triton_poi_fused_add_gelu_4 = async_compile.triton('triton_poi_fused_add_gelu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_gelu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/x2/cx2hq3zpe43mgtdq42iu5vz655w5lcyyjsdwbk4jjgjfohagl72u.py
# Topologically Sorted Source Nodes: [addcmul_27, mul_25, x_98], Original ATen: [aten.addcmul, aten.mul, aten.add]
# Source node to ATen node mapping:
#   addcmul_27 => add_79, mul_119, mul_120
#   mul_25 => mul_118
#   x_98 => add_78
# Graph fragment:
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg17_1, 1), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg8_1, %view_79), kwargs = {})
#   %add_78 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_74, %mul_118), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_119, %add_78), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg16_1, %mul_120), kwargs = {})
triton_poi_fused_add_addcmul_mul_5 = async_compile.triton('triton_poi_fused_add_addcmul_mul_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addcmul_mul_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addcmul_mul_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tmp4 + tmp9
    tmp11 = tmp3 * tmp10
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp12, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fy/cfy4zugp4o5zhtpmvjgaaurlvbjyjpqlncbqygjlnmq5v2fuhrwh.py
# Topologically Sorted Source Nodes: [mul_25, x_98, mul_26, x_99], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_25 => mul_118
#   mul_26 => mul_121
#   x_98 => add_78
#   x_99 => add_80
# Graph fragment:
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg8_1, %view_79), kwargs = {})
#   %add_78 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_74, %mul_118), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg15_1, %permute_70), kwargs = {})
#   %add_80 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_78, %mul_121), kwargs = {})
triton_poi_fused_add_mul_6 = async_compile.triton('triton_poi_fused_add_mul_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr2 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/el/celtpnmlamdujtyo3ogp26rfyo335d3z3jxmu7bhtj64oaaxqtgv.py
# Topologically Sorted Source Nodes: [mul_27, x_105, mul_28, x_106], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_27 => mul_127
#   mul_28 => mul_130
#   x_105 => add_84
#   x_106 => add_86
# Graph fragment:
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg20_1, %view_85), kwargs = {})
#   %add_84 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_80, %mul_127), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg27_1, %permute_75), kwargs = {})
#   %add_86 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_84, %mul_130), kwargs = {})
triton_poi_fused_add_mul_7 = async_compile.triton('triton_poi_fused_add_mul_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fj/cfjqf7kuiqcxlq2je7cmrivtk3mo65bgvsuns45wulyhwkrmp3fb.py
# Topologically Sorted Source Nodes: [addcmul_48, mul_45, x_168, mul_46, x_169, x_170], Original ATen: [aten.addcmul, aten.mul, aten.add, aten.clone]
# Source node to ATen node mapping:
#   addcmul_48 => add_141, mul_212, mul_213
#   mul_45 => mul_208
#   mul_46 => mul_211
#   x_168 => add_138
#   x_169 => add_140
#   x_170 => clone_70
# Graph fragment:
#   %mul_212 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg142_1, 1), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg128_1, %view_139), kwargs = {})
#   %add_138 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_134, %mul_208), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg135_1, %permute_120), kwargs = {})
#   %add_140 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_138, %mul_211), kwargs = {})
#   %mul_213 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_212, %add_140), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg141_1, %mul_213), kwargs = {})
#   %clone_70 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_141,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_addcmul_clone_mul_8 = async_compile.triton('triton_poi_fused_add_addcmul_clone_mul_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addcmul_clone_mul_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addcmul_clone_mul_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_out_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16 * tmp12
    tmp18 = tmp13 + tmp17
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (384*y3)), tmp12, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp18, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/td/ctdwlio43ocitxnfqgrflctcwlsqoan4sgdccoz6ha3lhxz4tzs7.py
# Topologically Sorted Source Nodes: [x_176, mul_47, x_175, x_177], Original ATen: [aten.addcmul, aten.mul, aten.add, aten.mean]
# Source node to ATen node mapping:
#   mul_47 => mul_217
#   x_175 => add_144
#   x_176 => add_145, mul_218, mul_219
#   x_177 => mean_1
# Graph fragment:
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg148_1, 1), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg140_1, %view_145), kwargs = {})
#   %add_144 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_140, %mul_217), kwargs = {})
#   %mul_219 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_218, %add_144), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg147_1, %mul_219), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_145, [1]), kwargs = {})
triton_red_fused_add_addcmul_mean_mul_9 = async_compile.triton('triton_red_fused_add_addcmul_mean_mul_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addcmul_mean_mul_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_addcmul_mean_mul_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    x1 = (xindex // 384)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp4 = tl.load(in_ptr2 + (x0 + (384*r2) + (37632*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr4 + (x0 + (384*r2) + (37632*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 1.0
        tmp3 = tmp1 * tmp2
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tmp4 + tmp9
        tmp11 = tmp3 * tmp10
        tmp12 = tmp0 + tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/so/csovgpfe6dktckekmba4wuecgodofif3ukzlh5ijcjhkw5q7b7wc.py
# Topologically Sorted Source Nodes: [x_176, mul_47, x_175, x_177], Original ATen: [aten.addcmul, aten.mul, aten.add, aten.mean]
# Source node to ATen node mapping:
#   mul_47 => mul_217
#   x_175 => add_144
#   x_176 => add_145, mul_218, mul_219
#   x_177 => mean_1
# Graph fragment:
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg148_1, 1), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg140_1, %view_145), kwargs = {})
#   %add_144 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_140, %mul_217), kwargs = {})
#   %mul_219 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_218, %add_144), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg147_1, %mul_219), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_145, [1]), kwargs = {})
triton_per_fused_add_addcmul_mean_mul_10 = async_compile.triton('triton_per_fused_add_addcmul_mean_mul_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addcmul_mean_mul_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_addcmul_mean_mul_10(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 384
    x1 = (xindex // 384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (768*x1)), xmask, other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (384, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg2_1, (384, ), (1, ))
    assert_size_stride(arg3_1, (384, ), (1, ))
    assert_size_stride(arg4_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg5_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg6_1, (196, 196), (196, 1))
    assert_size_stride(arg7_1, (196, ), (1, ))
    assert_size_stride(arg8_1, (384, ), (1, ))
    assert_size_stride(arg9_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg10_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg11_1, (1536, 384), (384, 1))
    assert_size_stride(arg12_1, (1536, ), (1, ))
    assert_size_stride(arg13_1, (384, 1536), (1536, 1))
    assert_size_stride(arg14_1, (384, ), (1, ))
    assert_size_stride(arg15_1, (384, ), (1, ))
    assert_size_stride(arg16_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg17_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg18_1, (196, 196), (196, 1))
    assert_size_stride(arg19_1, (196, ), (1, ))
    assert_size_stride(arg20_1, (384, ), (1, ))
    assert_size_stride(arg21_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg22_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg23_1, (1536, 384), (384, 1))
    assert_size_stride(arg24_1, (1536, ), (1, ))
    assert_size_stride(arg25_1, (384, 1536), (1536, 1))
    assert_size_stride(arg26_1, (384, ), (1, ))
    assert_size_stride(arg27_1, (384, ), (1, ))
    assert_size_stride(arg28_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg29_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg30_1, (196, 196), (196, 1))
    assert_size_stride(arg31_1, (196, ), (1, ))
    assert_size_stride(arg32_1, (384, ), (1, ))
    assert_size_stride(arg33_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg34_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg35_1, (1536, 384), (384, 1))
    assert_size_stride(arg36_1, (1536, ), (1, ))
    assert_size_stride(arg37_1, (384, 1536), (1536, 1))
    assert_size_stride(arg38_1, (384, ), (1, ))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg41_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg42_1, (196, 196), (196, 1))
    assert_size_stride(arg43_1, (196, ), (1, ))
    assert_size_stride(arg44_1, (384, ), (1, ))
    assert_size_stride(arg45_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg46_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg47_1, (1536, 384), (384, 1))
    assert_size_stride(arg48_1, (1536, ), (1, ))
    assert_size_stride(arg49_1, (384, 1536), (1536, 1))
    assert_size_stride(arg50_1, (384, ), (1, ))
    assert_size_stride(arg51_1, (384, ), (1, ))
    assert_size_stride(arg52_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg53_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg54_1, (196, 196), (196, 1))
    assert_size_stride(arg55_1, (196, ), (1, ))
    assert_size_stride(arg56_1, (384, ), (1, ))
    assert_size_stride(arg57_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg58_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg59_1, (1536, 384), (384, 1))
    assert_size_stride(arg60_1, (1536, ), (1, ))
    assert_size_stride(arg61_1, (384, 1536), (1536, 1))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (384, ), (1, ))
    assert_size_stride(arg64_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg65_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg66_1, (196, 196), (196, 1))
    assert_size_stride(arg67_1, (196, ), (1, ))
    assert_size_stride(arg68_1, (384, ), (1, ))
    assert_size_stride(arg69_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg70_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg71_1, (1536, 384), (384, 1))
    assert_size_stride(arg72_1, (1536, ), (1, ))
    assert_size_stride(arg73_1, (384, 1536), (1536, 1))
    assert_size_stride(arg74_1, (384, ), (1, ))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg77_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg78_1, (196, 196), (196, 1))
    assert_size_stride(arg79_1, (196, ), (1, ))
    assert_size_stride(arg80_1, (384, ), (1, ))
    assert_size_stride(arg81_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg82_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg83_1, (1536, 384), (384, 1))
    assert_size_stride(arg84_1, (1536, ), (1, ))
    assert_size_stride(arg85_1, (384, 1536), (1536, 1))
    assert_size_stride(arg86_1, (384, ), (1, ))
    assert_size_stride(arg87_1, (384, ), (1, ))
    assert_size_stride(arg88_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg89_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg90_1, (196, 196), (196, 1))
    assert_size_stride(arg91_1, (196, ), (1, ))
    assert_size_stride(arg92_1, (384, ), (1, ))
    assert_size_stride(arg93_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg94_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg95_1, (1536, 384), (384, 1))
    assert_size_stride(arg96_1, (1536, ), (1, ))
    assert_size_stride(arg97_1, (384, 1536), (1536, 1))
    assert_size_stride(arg98_1, (384, ), (1, ))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg101_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg102_1, (196, 196), (196, 1))
    assert_size_stride(arg103_1, (196, ), (1, ))
    assert_size_stride(arg104_1, (384, ), (1, ))
    assert_size_stride(arg105_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg106_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg107_1, (1536, 384), (384, 1))
    assert_size_stride(arg108_1, (1536, ), (1, ))
    assert_size_stride(arg109_1, (384, 1536), (1536, 1))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg113_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg114_1, (196, 196), (196, 1))
    assert_size_stride(arg115_1, (196, ), (1, ))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg118_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg119_1, (1536, 384), (384, 1))
    assert_size_stride(arg120_1, (1536, ), (1, ))
    assert_size_stride(arg121_1, (384, 1536), (1536, 1))
    assert_size_stride(arg122_1, (384, ), (1, ))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg125_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg126_1, (196, 196), (196, 1))
    assert_size_stride(arg127_1, (196, ), (1, ))
    assert_size_stride(arg128_1, (384, ), (1, ))
    assert_size_stride(arg129_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg130_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg131_1, (1536, 384), (384, 1))
    assert_size_stride(arg132_1, (1536, ), (1, ))
    assert_size_stride(arg133_1, (384, 1536), (1536, 1))
    assert_size_stride(arg134_1, (384, ), (1, ))
    assert_size_stride(arg135_1, (384, ), (1, ))
    assert_size_stride(arg136_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg137_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg138_1, (196, 196), (196, 1))
    assert_size_stride(arg139_1, (196, ), (1, ))
    assert_size_stride(arg140_1, (384, ), (1, ))
    assert_size_stride(arg141_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg142_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg143_1, (1536, 384), (384, 1))
    assert_size_stride(arg144_1, (1536, ), (1, ))
    assert_size_stride(arg145_1, (384, 1536), (1536, 1))
    assert_size_stride(arg146_1, (384, ), (1, ))
    assert_size_stride(arg147_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg148_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg149_1, (1000, 384), (384, 1))
    assert_size_stride(arg150_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((8, 196, 384), (75264, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [addcmul_25], Original ATen: [aten.addcmul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addcmul_0.run(arg4_1, arg5_1, buf0, arg2_1, buf1, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg4_1
        del arg5_1
        buf2 = empty_strided_cuda((3072, 196), (1, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf1, buf2, 602112, grid=grid(602112), stream=stream0)
        buf3 = reinterpret_tensor(buf1, (3072, 196), (196, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf2, reinterpret_tensor(arg6_1, (196, 196), (1, 196), 0), out=buf3)
        del arg6_1
        buf4 = reinterpret_tensor(buf0, (8, 196, 384), (75264, 1, 196), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [mul_24, x_92], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf4, arg2_1, arg3_1, buf3, arg7_1, 602112, grid=grid(602112), stream=stream0)
        del arg2_1
        del arg3_1
        del arg7_1
        buf5 = reinterpret_tensor(buf3, (8, 196, 384), (75264, 384, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [addcmul_26, x_93], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg9_1, arg10_1, buf4, buf5, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg10_1
        del arg9_1
        buf6 = empty_strided_cuda((1568, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_93], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1568, 384), (384, 1), 0), reinterpret_tensor(arg11_1, (384, 1536), (1, 384), 0), out=buf6)
        del arg11_1
        buf7 = reinterpret_tensor(buf6, (8, 196, 1536), (301056, 1536, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_93, x_94], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf7, arg12_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg12_1
        buf8 = reinterpret_tensor(buf5, (1568, 384), (384, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg13_1, (1536, 384), (1, 1536), 0), out=buf8)
        del arg13_1
        buf9 = reinterpret_tensor(buf2, (8, 196, 384), (75264, 384, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [addcmul_27, mul_25, x_98], Original ATen: [aten.addcmul, aten.mul, aten.add]
        triton_poi_fused_add_addcmul_mul_5.run(arg16_1, arg17_1, buf4, arg8_1, buf8, arg14_1, buf9, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg16_1
        del arg17_1
        buf10 = empty_strided_cuda((3072, 196), (1, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf9, buf10, 602112, grid=grid(602112), stream=stream0)
        buf11 = reinterpret_tensor(buf9, (3072, 196), (196, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf10, reinterpret_tensor(arg18_1, (196, 196), (1, 196), 0), out=buf11)
        del arg18_1
        buf12 = reinterpret_tensor(buf11, (8, 196, 384), (75264, 1, 196), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [mul_25, x_98, mul_26, x_99], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_6.run(buf12, buf4, arg8_1, buf8, arg14_1, arg15_1, arg19_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg14_1
        del arg15_1
        del arg19_1
        del arg8_1
        buf13 = reinterpret_tensor(buf8, (8, 196, 384), (75264, 384, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [addcmul_28, x_100], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg21_1, arg22_1, buf12, buf13, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg21_1
        del arg22_1
        buf14 = reinterpret_tensor(buf7, (1568, 1536), (1536, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_100], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (1568, 384), (384, 1), 0), reinterpret_tensor(arg23_1, (384, 1536), (1, 384), 0), out=buf14)
        del arg23_1
        buf15 = reinterpret_tensor(buf14, (8, 196, 1536), (301056, 1536, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_100, x_101], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf15, arg24_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg24_1
        buf16 = reinterpret_tensor(buf13, (1568, 384), (384, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg25_1, (1536, 384), (1, 1536), 0), out=buf16)
        del arg25_1
        buf17 = reinterpret_tensor(buf4, (8, 196, 384), (75264, 384, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [addcmul_29, mul_27, x_105], Original ATen: [aten.addcmul, aten.mul, aten.add]
        triton_poi_fused_add_addcmul_mul_5.run(arg28_1, arg29_1, buf12, arg20_1, buf16, arg26_1, buf17, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg28_1
        del arg29_1
        buf18 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf17, buf18, 602112, grid=grid(602112), stream=stream0)
        buf19 = reinterpret_tensor(buf17, (3072, 196), (196, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf18, reinterpret_tensor(arg30_1, (196, 196), (1, 196), 0), out=buf19)
        del arg30_1
        buf20 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [mul_27, x_105, mul_28, x_106], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_7.run(buf20, arg20_1, buf16, arg26_1, arg27_1, buf19, arg31_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg20_1
        del arg26_1
        del arg27_1
        del arg31_1
        buf21 = reinterpret_tensor(buf19, (8, 196, 384), (75264, 384, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [addcmul_30, x_107], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg33_1, arg34_1, buf20, buf21, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg33_1
        del arg34_1
        buf22 = reinterpret_tensor(buf15, (1568, 1536), (1536, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_107], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (1568, 384), (384, 1), 0), reinterpret_tensor(arg35_1, (384, 1536), (1, 384), 0), out=buf22)
        del arg35_1
        buf23 = reinterpret_tensor(buf22, (8, 196, 1536), (301056, 1536, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_107, x_108], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf23, arg36_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg36_1
        buf24 = reinterpret_tensor(buf21, (1568, 384), (384, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg37_1, (1536, 384), (1, 1536), 0), out=buf24)
        del arg37_1
        buf25 = reinterpret_tensor(buf16, (8, 196, 384), (75264, 384, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [addcmul_31, mul_29, x_112], Original ATen: [aten.addcmul, aten.mul, aten.add]
        triton_poi_fused_add_addcmul_mul_5.run(arg40_1, arg41_1, buf20, arg32_1, buf24, arg38_1, buf25, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg40_1
        del arg41_1
        buf26 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf25, buf26, 602112, grid=grid(602112), stream=stream0)
        buf27 = reinterpret_tensor(buf25, (3072, 196), (196, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf26, reinterpret_tensor(arg42_1, (196, 196), (1, 196), 0), out=buf27)
        del arg42_1
        buf28 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [mul_29, x_112, mul_30, x_113], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_7.run(buf28, arg32_1, buf24, arg38_1, arg39_1, buf27, arg43_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg32_1
        del arg38_1
        del arg39_1
        del arg43_1
        buf29 = reinterpret_tensor(buf27, (8, 196, 384), (75264, 384, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [addcmul_32, x_114], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg45_1, arg46_1, buf28, buf29, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg45_1
        del arg46_1
        buf30 = reinterpret_tensor(buf23, (1568, 1536), (1536, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_114], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (1568, 384), (384, 1), 0), reinterpret_tensor(arg47_1, (384, 1536), (1, 384), 0), out=buf30)
        del arg47_1
        buf31 = reinterpret_tensor(buf30, (8, 196, 1536), (301056, 1536, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_114, x_115], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf31, arg48_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg48_1
        buf32 = reinterpret_tensor(buf29, (1568, 384), (384, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg49_1, (1536, 384), (1, 1536), 0), out=buf32)
        del arg49_1
        buf33 = reinterpret_tensor(buf24, (8, 196, 384), (75264, 384, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [addcmul_33, mul_31, x_119], Original ATen: [aten.addcmul, aten.mul, aten.add]
        triton_poi_fused_add_addcmul_mul_5.run(arg52_1, arg53_1, buf28, arg44_1, buf32, arg50_1, buf33, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg52_1
        del arg53_1
        buf34 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf33, buf34, 602112, grid=grid(602112), stream=stream0)
        buf35 = reinterpret_tensor(buf33, (3072, 196), (196, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf34, reinterpret_tensor(arg54_1, (196, 196), (1, 196), 0), out=buf35)
        del arg54_1
        buf36 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [mul_31, x_119, mul_32, x_120], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_7.run(buf36, arg44_1, buf32, arg50_1, arg51_1, buf35, arg55_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg44_1
        del arg50_1
        del arg51_1
        del arg55_1
        buf37 = reinterpret_tensor(buf35, (8, 196, 384), (75264, 384, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [addcmul_34, x_121], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg57_1, arg58_1, buf36, buf37, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg57_1
        del arg58_1
        buf38 = reinterpret_tensor(buf31, (1568, 1536), (1536, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_121], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (1568, 384), (384, 1), 0), reinterpret_tensor(arg59_1, (384, 1536), (1, 384), 0), out=buf38)
        del arg59_1
        buf39 = reinterpret_tensor(buf38, (8, 196, 1536), (301056, 1536, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_121, x_122], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf39, arg60_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg60_1
        buf40 = reinterpret_tensor(buf37, (1568, 384), (384, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg61_1, (1536, 384), (1, 1536), 0), out=buf40)
        del arg61_1
        buf41 = reinterpret_tensor(buf32, (8, 196, 384), (75264, 384, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [addcmul_35, mul_33, x_126], Original ATen: [aten.addcmul, aten.mul, aten.add]
        triton_poi_fused_add_addcmul_mul_5.run(arg64_1, arg65_1, buf36, arg56_1, buf40, arg62_1, buf41, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg64_1
        del arg65_1
        buf42 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf41, buf42, 602112, grid=grid(602112), stream=stream0)
        buf43 = reinterpret_tensor(buf41, (3072, 196), (196, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf42, reinterpret_tensor(arg66_1, (196, 196), (1, 196), 0), out=buf43)
        del arg66_1
        buf44 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [mul_33, x_126, mul_34, x_127], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_7.run(buf44, arg56_1, buf40, arg62_1, arg63_1, buf43, arg67_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg56_1
        del arg62_1
        del arg63_1
        del arg67_1
        buf45 = reinterpret_tensor(buf43, (8, 196, 384), (75264, 384, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [addcmul_36, x_128], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg69_1, arg70_1, buf44, buf45, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg69_1
        del arg70_1
        buf46 = reinterpret_tensor(buf39, (1568, 1536), (1536, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_128], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (1568, 384), (384, 1), 0), reinterpret_tensor(arg71_1, (384, 1536), (1, 384), 0), out=buf46)
        del arg71_1
        buf47 = reinterpret_tensor(buf46, (8, 196, 1536), (301056, 1536, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_128, x_129], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf47, arg72_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg72_1
        buf48 = reinterpret_tensor(buf45, (1568, 384), (384, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg73_1, (1536, 384), (1, 1536), 0), out=buf48)
        del arg73_1
        buf49 = reinterpret_tensor(buf40, (8, 196, 384), (75264, 384, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [addcmul_37, mul_35, x_133], Original ATen: [aten.addcmul, aten.mul, aten.add]
        triton_poi_fused_add_addcmul_mul_5.run(arg76_1, arg77_1, buf44, arg68_1, buf48, arg74_1, buf49, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg76_1
        del arg77_1
        buf50 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf49, buf50, 602112, grid=grid(602112), stream=stream0)
        buf51 = reinterpret_tensor(buf49, (3072, 196), (196, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf50, reinterpret_tensor(arg78_1, (196, 196), (1, 196), 0), out=buf51)
        del arg78_1
        buf52 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [mul_35, x_133, mul_36, x_134], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_7.run(buf52, arg68_1, buf48, arg74_1, arg75_1, buf51, arg79_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg68_1
        del arg74_1
        del arg75_1
        del arg79_1
        buf53 = reinterpret_tensor(buf51, (8, 196, 384), (75264, 384, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [addcmul_38, x_135], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg81_1, arg82_1, buf52, buf53, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg81_1
        del arg82_1
        buf54 = reinterpret_tensor(buf47, (1568, 1536), (1536, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_135], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (1568, 384), (384, 1), 0), reinterpret_tensor(arg83_1, (384, 1536), (1, 384), 0), out=buf54)
        del arg83_1
        buf55 = reinterpret_tensor(buf54, (8, 196, 1536), (301056, 1536, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_135, x_136], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf55, arg84_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg84_1
        buf56 = reinterpret_tensor(buf53, (1568, 384), (384, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf55, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg85_1, (1536, 384), (1, 1536), 0), out=buf56)
        del arg85_1
        buf57 = reinterpret_tensor(buf48, (8, 196, 384), (75264, 384, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [addcmul_39, mul_37, x_140], Original ATen: [aten.addcmul, aten.mul, aten.add]
        triton_poi_fused_add_addcmul_mul_5.run(arg88_1, arg89_1, buf52, arg80_1, buf56, arg86_1, buf57, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg88_1
        del arg89_1
        buf58 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf57, buf58, 602112, grid=grid(602112), stream=stream0)
        buf59 = reinterpret_tensor(buf57, (3072, 196), (196, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf58, reinterpret_tensor(arg90_1, (196, 196), (1, 196), 0), out=buf59)
        del arg90_1
        buf60 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [mul_37, x_140, mul_38, x_141], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_7.run(buf60, arg80_1, buf56, arg86_1, arg87_1, buf59, arg91_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg80_1
        del arg86_1
        del arg87_1
        del arg91_1
        buf61 = reinterpret_tensor(buf59, (8, 196, 384), (75264, 384, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [addcmul_40, x_142], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg93_1, arg94_1, buf60, buf61, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg93_1
        del arg94_1
        buf62 = reinterpret_tensor(buf55, (1568, 1536), (1536, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_142], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (1568, 384), (384, 1), 0), reinterpret_tensor(arg95_1, (384, 1536), (1, 384), 0), out=buf62)
        del arg95_1
        buf63 = reinterpret_tensor(buf62, (8, 196, 1536), (301056, 1536, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_142, x_143], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf63, arg96_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg96_1
        buf64 = reinterpret_tensor(buf61, (1568, 384), (384, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg97_1, (1536, 384), (1, 1536), 0), out=buf64)
        del arg97_1
        buf65 = reinterpret_tensor(buf56, (8, 196, 384), (75264, 384, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [addcmul_41, mul_39, x_147], Original ATen: [aten.addcmul, aten.mul, aten.add]
        triton_poi_fused_add_addcmul_mul_5.run(arg100_1, arg101_1, buf60, arg92_1, buf64, arg98_1, buf65, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg100_1
        del arg101_1
        buf66 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf65, buf66, 602112, grid=grid(602112), stream=stream0)
        buf67 = reinterpret_tensor(buf65, (3072, 196), (196, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf66, reinterpret_tensor(arg102_1, (196, 196), (1, 196), 0), out=buf67)
        del arg102_1
        buf68 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [mul_39, x_147, mul_40, x_148], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_7.run(buf68, arg92_1, buf64, arg98_1, arg99_1, buf67, arg103_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg103_1
        del arg92_1
        del arg98_1
        del arg99_1
        buf69 = reinterpret_tensor(buf67, (8, 196, 384), (75264, 384, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [addcmul_42, x_149], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg105_1, arg106_1, buf68, buf69, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg105_1
        del arg106_1
        buf70 = reinterpret_tensor(buf63, (1568, 1536), (1536, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [x_149], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (1568, 384), (384, 1), 0), reinterpret_tensor(arg107_1, (384, 1536), (1, 384), 0), out=buf70)
        del arg107_1
        buf71 = reinterpret_tensor(buf70, (8, 196, 1536), (301056, 1536, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_149, x_150], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf71, arg108_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg108_1
        buf72 = reinterpret_tensor(buf69, (1568, 384), (384, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg109_1, (1536, 384), (1, 1536), 0), out=buf72)
        del arg109_1
        buf73 = reinterpret_tensor(buf64, (8, 196, 384), (75264, 384, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [addcmul_43, mul_41, x_154], Original ATen: [aten.addcmul, aten.mul, aten.add]
        triton_poi_fused_add_addcmul_mul_5.run(arg112_1, arg113_1, buf68, arg104_1, buf72, arg110_1, buf73, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg112_1
        del arg113_1
        buf74 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf73, buf74, 602112, grid=grid(602112), stream=stream0)
        buf75 = reinterpret_tensor(buf73, (3072, 196), (196, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf74, reinterpret_tensor(arg114_1, (196, 196), (1, 196), 0), out=buf75)
        del arg114_1
        buf76 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [mul_41, x_154, mul_42, x_155], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_7.run(buf76, arg104_1, buf72, arg110_1, arg111_1, buf75, arg115_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg104_1
        del arg110_1
        del arg111_1
        del arg115_1
        buf77 = reinterpret_tensor(buf75, (8, 196, 384), (75264, 384, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [addcmul_44, x_156], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg117_1, arg118_1, buf76, buf77, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg117_1
        del arg118_1
        buf78 = reinterpret_tensor(buf71, (1568, 1536), (1536, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_156], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (1568, 384), (384, 1), 0), reinterpret_tensor(arg119_1, (384, 1536), (1, 384), 0), out=buf78)
        del arg119_1
        buf79 = reinterpret_tensor(buf78, (8, 196, 1536), (301056, 1536, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf79, arg120_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg120_1
        buf80 = reinterpret_tensor(buf77, (1568, 384), (384, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg121_1, (1536, 384), (1, 1536), 0), out=buf80)
        del arg121_1
        buf81 = reinterpret_tensor(buf72, (8, 196, 384), (75264, 384, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [addcmul_45, mul_43, x_161], Original ATen: [aten.addcmul, aten.mul, aten.add]
        triton_poi_fused_add_addcmul_mul_5.run(arg124_1, arg125_1, buf76, arg116_1, buf80, arg122_1, buf81, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg124_1
        del arg125_1
        buf82 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf81, buf82, 602112, grid=grid(602112), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (3072, 196), (196, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf82, reinterpret_tensor(arg126_1, (196, 196), (1, 196), 0), out=buf83)
        del arg126_1
        buf84 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [mul_43, x_161, mul_44, x_162], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_7.run(buf84, arg116_1, buf80, arg122_1, arg123_1, buf83, arg127_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg116_1
        del arg122_1
        del arg123_1
        del arg127_1
        buf85 = reinterpret_tensor(buf83, (8, 196, 384), (75264, 384, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [addcmul_46, x_163], Original ATen: [aten.addcmul, aten.clone]
        triton_poi_fused_addcmul_clone_3.run(arg129_1, arg130_1, buf84, buf85, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg129_1
        del arg130_1
        buf86 = reinterpret_tensor(buf79, (1568, 1536), (1536, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (1568, 384), (384, 1), 0), reinterpret_tensor(arg131_1, (384, 1536), (1, 384), 0), out=buf86)
        del arg131_1
        buf87 = reinterpret_tensor(buf86, (8, 196, 1536), (301056, 1536, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_163, x_164], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf87, arg132_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg132_1
        buf88 = reinterpret_tensor(buf85, (1568, 384), (384, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg133_1, (1536, 384), (1, 1536), 0), out=buf88)
        del arg133_1
        buf89 = reinterpret_tensor(buf80, (8, 196, 384), (75264, 384, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [addcmul_47, mul_45, x_168], Original ATen: [aten.addcmul, aten.mul, aten.add]
        triton_poi_fused_add_addcmul_mul_5.run(arg136_1, arg137_1, buf84, arg128_1, buf88, arg134_1, buf89, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg136_1
        del arg137_1
        buf90 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf89, buf90, 602112, grid=grid(602112), stream=stream0)
        buf91 = reinterpret_tensor(buf89, (3072, 196), (196, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf90, reinterpret_tensor(arg138_1, (196, 196), (1, 196), 0), out=buf91)
        del arg138_1
        buf92 = reinterpret_tensor(buf88, (8, 196, 384), (75264, 384, 1), 0); del buf88  # reuse
        buf93 = reinterpret_tensor(buf90, (8, 196, 384), (75264, 384, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [addcmul_48, mul_45, x_168, mul_46, x_169, x_170], Original ATen: [aten.addcmul, aten.mul, aten.add, aten.clone]
        triton_poi_fused_add_addcmul_clone_mul_8.run(buf92, buf84, arg128_1, arg134_1, arg135_1, buf91, arg139_1, arg141_1, arg142_1, buf93, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg128_1
        del arg134_1
        del arg135_1
        del arg139_1
        del arg141_1
        del arg142_1
        del buf84
        del buf91
        buf94 = reinterpret_tensor(buf87, (1568, 1536), (1536, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_170], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (1568, 384), (384, 1), 0), reinterpret_tensor(arg143_1, (384, 1536), (1, 384), 0), out=buf94)
        del arg143_1
        buf95 = reinterpret_tensor(buf94, (8, 196, 1536), (301056, 1536, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_170, x_171], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf95, arg144_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg144_1
        buf96 = reinterpret_tensor(buf93, (1568, 384), (384, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg145_1, (1536, 384), (1, 1536), 0), out=buf96)
        del arg145_1
        del buf95
        buf97 = empty_strided_cuda((8, 384, 2), (768, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_176, mul_47, x_175, x_177], Original ATen: [aten.addcmul, aten.mul, aten.add, aten.mean]
        triton_red_fused_add_addcmul_mean_mul_9.run(arg147_1, arg148_1, buf92, arg140_1, buf96, arg146_1, buf97, 6144, 98, grid=grid(6144), stream=stream0)
        del arg140_1
        del arg146_1
        del arg147_1
        del arg148_1
        del buf92
        del buf96
        buf99 = empty_strided_cuda((8, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_176, mul_47, x_175, x_177], Original ATen: [aten.addcmul, aten.mul, aten.add, aten.mean]
        triton_per_fused_add_addcmul_mean_mul_10.run(buf97, buf99, 3072, 2, grid=grid(3072), stream=stream0)
        del buf97
        buf100 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_176, mul_47, x_175, x_177, x_179], Original ATen: [aten.addcmul, aten.mul, aten.add, aten.mean, aten.addmm]
        extern_kernels.addmm(arg150_1, buf99, reinterpret_tensor(arg149_1, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf100)
        del arg149_1
        del arg150_1
        del buf99
    return (buf100, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resmlp_12_224', benchmark_compiled_module)
