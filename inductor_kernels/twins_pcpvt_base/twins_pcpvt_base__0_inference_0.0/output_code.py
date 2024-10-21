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
# Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_33 => convolution_33
# Graph fragment:
#   %convolution_33 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/dk/cdkrxm2oguw66kqtc67klvcxub3xn6mx22aiioll6anadn73qikk.py
# Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_33 => convolution_33
# Graph fragment:
#   %convolution_33 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (48*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nq/cnqh7w3dyesoob6k2cqxx3kfnx2u6c3u3aeq22psltnfysgv2fzh.py
# Topologically Sorted Source Nodes: [x_411, layer_norm_87], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_87 => add_262, add_263, mul_258, mul_259, rsqrt_87, sub_87, var_mean_87
#   x_411 => add_260, add_261, clone_96, mul_256, mul_257, rsqrt_86, sub_86, var_mean_86
# Graph fragment:
#   %clone_96 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_294,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_86 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_96, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_96, %getitem_341), kwargs = {})
#   %add_260 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_340, 1e-05), kwargs = {})
#   %rsqrt_86 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_260,), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_86, %rsqrt_86), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %arg3_1), kwargs = {})
#   %add_261 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %arg4_1), kwargs = {})
#   %var_mean_87 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_261, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_261, %getitem_343), kwargs = {})
#   %add_262 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_342, 1e-06), kwargs = {})
#   %rsqrt_87 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_262,), kwargs = {})
#   %mul_258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %rsqrt_87), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_258, %arg5_1), kwargs = {})
#   %add_263 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_259, %arg6_1), kwargs = {})
triton_per_fused_native_layer_norm_2 = async_compile.triton('triton_per_fused_native_layer_norm_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 8, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 64.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(xmask, tmp30, 0)
    tmp33 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp35 = tl.where(xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp36 / tmp11
    tmp38 = tmp30 - tmp37
    tmp39 = tmp38 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.where(xmask, tmp40, 0)
    tmp43 = tl.sum(tmp42, 1)[:, None]
    tmp44 = tmp29 - tmp37
    tmp45 = tmp43 / tmp20
    tmp46 = 1e-06
    tmp47 = tmp45 + tmp46
    tmp48 = libdevice.rsqrt(tmp47)
    tmp49 = tmp44 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp29, xmask)
    tl.store(out_ptr5 + (r1 + (64*x0)), tmp53, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bd/cbdxrp4kvdjcrdksiyjd7lez6ffmyzz2p4giy3qadaictdadpork.py
# Topologically Sorted Source Nodes: [conv2d_34], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_34 => convolution_34
# Graph fragment:
#   %convolution_34 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_437, %arg9_1, %arg10_1, [8, 8], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (4096*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xn/cxn4mw37bdnt3qi7pkuuu3ljzb4la73qlwzqsa3xt5trvmf37u5c.py
# Topologically Sorted Source Nodes: [x_415], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_415 => add_264, add_265, mul_260, mul_261, rsqrt_88, sub_88, var_mean_88
# Graph fragment:
#   %var_mean_88 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_298, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_298, %getitem_345), kwargs = {})
#   %add_264 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_344, 1e-05), kwargs = {})
#   %rsqrt_88 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_264,), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %rsqrt_88), kwargs = {})
#   %mul_261 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_260, %arg11_1), kwargs = {})
#   %add_265 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_261, %arg12_1), kwargs = {})
triton_per_fused_native_layer_norm_4 = async_compile.triton('triton_per_fused_native_layer_norm_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 64.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zm/czmiqr2mea2n4heds7ca7m5kjsqm4hewmbsd7pugy3iofn3mbyot.py
# Topologically Sorted Source Nodes: [x_420, layer_norm_89], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_89 => add_267, add_268, mul_262, mul_263, rsqrt_89, sub_89, var_mean_89
#   x_420 => add_266
# Graph fragment:
#   %add_266 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_261, %view_444), kwargs = {})
#   %var_mean_89 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_266, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_266, %getitem_353), kwargs = {})
#   %add_267 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_352, 1e-06), kwargs = {})
#   %rsqrt_89 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_267,), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %rsqrt_89), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %arg17_1), kwargs = {})
#   %add_268 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %arg18_1), kwargs = {})
triton_per_fused_add_native_layer_norm_5 = async_compile.triton('triton_per_fused_add_native_layer_norm_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp12 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 64.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wb/cwbe6whmypghndxcgnfmgeud2gypdsx6fqpsfc4hz3pmgt4ql2qa.py
# Topologically Sorted Source Nodes: [x_422], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_422 => add_269, erf_28, mul_264, mul_265, mul_266
# Graph fragment:
#   %mul_264 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_446, 0.5), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_446, 0.7071067811865476), kwargs = {})
#   %erf_28 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_265,), kwargs = {})
#   %add_269 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_28, 1), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_264, %add_269), kwargs = {})
triton_poi_fused_gelu_6 = async_compile.triton('triton_poi_fused_gelu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/4i/c4iz4mpjgjzyrzqqukdpsrgay2sildvd5jcn7fsk54nktclzoc4q.py
# Topologically Sorted Source Nodes: [x_420, x_426], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_420 => add_266
#   x_426 => add_270
# Graph fragment:
#   %add_266 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_261, %view_444), kwargs = {})
#   %add_270 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_266, %view_448), kwargs = {})
triton_poi_fused_add_7 = async_compile.triton('triton_poi_fused_add_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zj/czji3p24jlgvbxdlp4y32cyn6gv46jql2q37yrpgvyg4tqs2jfyi.py
# Topologically Sorted Source Nodes: [layer_norm_90], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_90 => add_272, add_273, mul_267, mul_268, rsqrt_90, sub_90, var_mean_90
# Graph fragment:
#   %var_mean_90 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_307, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_307, %getitem_355), kwargs = {})
#   %add_272 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_354, 1e-06), kwargs = {})
#   %rsqrt_90 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_272,), kwargs = {})
#   %mul_267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %rsqrt_90), kwargs = {})
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_267, %arg25_1), kwargs = {})
#   %add_273 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_268, %arg26_1), kwargs = {})
triton_per_fused_native_layer_norm_8 = async_compile.triton('triton_per_fused_native_layer_norm_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (64*x0)), xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 64.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/do/cdoaif77qp3b4hznjteidzueaucehtveoeg5akbthyc6q5ld5t6n.py
# Topologically Sorted Source Nodes: [x_436, layer_norm_92], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_92 => add_277, add_278, mul_271, mul_272, rsqrt_92, sub_92, var_mean_92
#   x_436 => add_276
# Graph fragment:
#   %add_276 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_307, %view_462), kwargs = {})
#   %var_mean_92 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_276, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_276, %getitem_365), kwargs = {})
#   %add_277 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_364, 1e-06), kwargs = {})
#   %rsqrt_92 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_277,), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %rsqrt_92), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %arg37_1), kwargs = {})
#   %add_278 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_272, %arg38_1), kwargs = {})
triton_per_fused_add_native_layer_norm_9 = async_compile.triton('triton_per_fused_add_native_layer_norm_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (64*x0)), xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (64*x0)), xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 64.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (64*x0)), tmp8, xmask)
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp35, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5j/c5jekbkvcqw63ud634kmol2li6tuvojlqoh3xsjdgnmcpr7wgawn.py
# Topologically Sorted Source Nodes: [x_442, x_450, layer_norm_95], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_95 => add_286, add_287, mul_280, mul_281, rsqrt_95, sub_95, var_mean_95
#   x_442 => add_280
#   x_450 => add_285
# Graph fragment:
#   %add_280 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_276, %view_466), kwargs = {})
#   %add_285 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_280, %view_477), kwargs = {})
#   %var_mean_95 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_285, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_285, %getitem_377), kwargs = {})
#   %add_286 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_376, 1e-06), kwargs = {})
#   %rsqrt_95 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_286,), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %rsqrt_95), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_280, %arg55_1), kwargs = {})
#   %add_287 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_281, %arg56_1), kwargs = {})
triton_per_fused_add_native_layer_norm_10 = async_compile.triton('triton_per_fused_add_native_layer_norm_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r1 + (64*x0)), xmask, other=0.0)
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
    tmp16 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 64.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (64*x0)), tmp8, xmask)
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp35, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sp/cspsawbd4wwspvwcg6zizg67swu24sp5urpeefvl3ibznmvrh3wf.py
# Topologically Sorted Source Nodes: [x_457], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_457 => clone_107
# Graph fragment:
#   %clone_107 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_328,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_11 = async_compile.triton('triton_poi_fused_clone_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_11(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h4/ch4uqt353n6nqft4rfas4gcfagam4ip2hxqeq5xdovhch5calu7d.py
# Topologically Sorted Source Nodes: [x_457, conv2d_38], Original ATen: [aten.clone, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_38 => convolution_38
#   x_457 => clone_107
# Graph fragment:
#   %clone_107 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_328,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_38 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_107, %arg61_1, %arg62_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_clone_convolution_12 = async_compile.triton('triton_poi_fused_clone_convolution_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (256*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2s/c2s4bviaz63rdb5fez4hr4d5samqhzjny7a7arl3lrxxupiyopoq.py
# Topologically Sorted Source Nodes: [x_459, layer_norm_97], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_97 => add_292, add_293, mul_287, mul_288, rsqrt_97, sub_97, var_mean_97
#   x_459 => add_290, add_291, clone_108, mul_285, mul_286, rsqrt_96, sub_96, var_mean_96
# Graph fragment:
#   %clone_108 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_329,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_96 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_108, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_108, %getitem_379), kwargs = {})
#   %add_290 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_378, 1e-05), kwargs = {})
#   %rsqrt_96 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_290,), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %rsqrt_96), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_285, %arg63_1), kwargs = {})
#   %add_291 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_286, %arg64_1), kwargs = {})
#   %var_mean_97 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_291, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_97 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_291, %getitem_381), kwargs = {})
#   %add_292 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_380, 1e-06), kwargs = {})
#   %rsqrt_97 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_292,), kwargs = {})
#   %mul_287 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_97, %rsqrt_97), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_287, %arg65_1), kwargs = {})
#   %add_293 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_288, %arg66_1), kwargs = {})
triton_per_fused_native_layer_norm_13 = async_compile.triton('triton_per_fused_native_layer_norm_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 8, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 128.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(xmask, tmp30, 0)
    tmp33 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp35 = tl.where(xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp36 / tmp11
    tmp38 = tmp30 - tmp37
    tmp39 = tmp38 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.where(xmask, tmp40, 0)
    tmp43 = tl.sum(tmp42, 1)[:, None]
    tmp44 = tmp29 - tmp37
    tmp45 = tmp43 / tmp20
    tmp46 = 1e-06
    tmp47 = tmp45 + tmp46
    tmp48 = libdevice.rsqrt(tmp47)
    tmp49 = tmp44 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp29, xmask)
    tl.store(out_ptr5 + (r1 + (128*x0)), tmp53, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kp/ckp34gisi243snu464ii7ccq5j3nlola4i3hkikwnniohan4yxk2.py
# Topologically Sorted Source Nodes: [conv2d_39], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_39 => convolution_39
# Graph fragment:
#   %convolution_39 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_487, %arg69_1, %arg70_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_14 = async_compile.triton('triton_poi_fused_convolution_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (2048*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fo/cfouhkaacdn6lwoxgdohlsq6huotd24i3r424k2qytwys5dam2xo.py
# Topologically Sorted Source Nodes: [x_463], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_463 => add_294, add_295, mul_289, mul_290, rsqrt_98, sub_98, var_mean_98
# Graph fragment:
#   %var_mean_98 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_333, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_333, %getitem_383), kwargs = {})
#   %add_294 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_382, 1e-05), kwargs = {})
#   %rsqrt_98 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_294,), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_98, %rsqrt_98), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %arg71_1), kwargs = {})
#   %add_295 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %arg72_1), kwargs = {})
triton_per_fused_native_layer_norm_15 = async_compile.triton('triton_per_fused_native_layer_norm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 128.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mf/cmfl2mjcgrky3lcgbma5eod4pytpkqtan7qz5zpjbfmjyvxqczf6.py
# Topologically Sorted Source Nodes: [x_468, layer_norm_99], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_99 => add_297, add_298, mul_291, mul_292, rsqrt_99, sub_99, var_mean_99
#   x_468 => add_296
# Graph fragment:
#   %add_296 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_291, %view_494), kwargs = {})
#   %var_mean_99 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_296, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_99 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_296, %getitem_391), kwargs = {})
#   %add_297 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_390, 1e-06), kwargs = {})
#   %rsqrt_99 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_297,), kwargs = {})
#   %mul_291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_99, %rsqrt_99), kwargs = {})
#   %mul_292 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_291, %arg77_1), kwargs = {})
#   %add_298 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_292, %arg78_1), kwargs = {})
triton_per_fused_add_native_layer_norm_16 = async_compile.triton('triton_per_fused_add_native_layer_norm_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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


# kernel path: /tmp/torchinductor_sahanp/sw/cswa6j26jod6j7vvkvrpwdp5d36xwkt2yxndkmh4iefzrbbxc3or.py
# Topologically Sorted Source Nodes: [x_470], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_470 => add_299, erf_31, mul_293, mul_294, mul_295
# Graph fragment:
#   %mul_293 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_496, 0.5), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_496, 0.7071067811865476), kwargs = {})
#   %erf_31 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_294,), kwargs = {})
#   %add_299 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_31, 1), kwargs = {})
#   %mul_295 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_293, %add_299), kwargs = {})
triton_poi_fused_gelu_17 = async_compile.triton('triton_poi_fused_gelu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/kc/ckc6ljy4djqgwvjozaqdu5bindvdy3jgdf3ikghu6eeliwtkc7o6.py
# Topologically Sorted Source Nodes: [x_468, x_474], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_468 => add_296
#   x_474 => add_300
# Graph fragment:
#   %add_296 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_291, %view_494), kwargs = {})
#   %add_300 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_296, %view_498), kwargs = {})
triton_poi_fused_add_18 = async_compile.triton('triton_poi_fused_add_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pu/cpuswmek4gpmmvx7s77wys7o6qxf7ncplq4wxcgt5ahmnlctpqb4.py
# Topologically Sorted Source Nodes: [layer_norm_100], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_100 => add_302, add_303, mul_296, mul_297, rsqrt_100, sub_100, var_mean_100
# Graph fragment:
#   %var_mean_100 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_342, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_342, %getitem_393), kwargs = {})
#   %add_302 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_392, 1e-06), kwargs = {})
#   %rsqrt_100 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_302,), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %rsqrt_100), kwargs = {})
#   %mul_297 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_296, %arg85_1), kwargs = {})
#   %add_303 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_297, %arg86_1), kwargs = {})
triton_per_fused_native_layer_norm_19 = async_compile.triton('triton_per_fused_native_layer_norm_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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


# kernel path: /tmp/torchinductor_sahanp/i6/ci6b42f5bbx3gww56o7bkog3hhvixkihxy47bz2d3uogccnrgood.py
# Topologically Sorted Source Nodes: [x_484, layer_norm_102], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_102 => add_307, add_308, mul_300, mul_301, rsqrt_102, sub_102, var_mean_102
#   x_484 => add_306
# Graph fragment:
#   %add_306 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_342, %view_512), kwargs = {})
#   %var_mean_102 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_306, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_306, %getitem_403), kwargs = {})
#   %add_307 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_402, 1e-06), kwargs = {})
#   %rsqrt_102 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_307,), kwargs = {})
#   %mul_300 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %rsqrt_102), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_300, %arg97_1), kwargs = {})
#   %add_308 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_301, %arg98_1), kwargs = {})
triton_per_fused_add_native_layer_norm_20 = async_compile.triton('triton_per_fused_add_native_layer_norm_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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


# kernel path: /tmp/torchinductor_sahanp/lb/clbbzcf3zrzpxeehyranicugsoxkktd5vgto3yuridlkmtk62jdk.py
# Topologically Sorted Source Nodes: [x_490, x_498, layer_norm_105], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_105 => add_316, add_317, mul_309, mul_310, rsqrt_105, sub_105, var_mean_105
#   x_490 => add_310
#   x_498 => add_315
# Graph fragment:
#   %add_310 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_306, %view_516), kwargs = {})
#   %add_315 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_310, %view_527), kwargs = {})
#   %var_mean_105 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_315, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_315, %getitem_415), kwargs = {})
#   %add_316 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_414, 1e-06), kwargs = {})
#   %rsqrt_105 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_316,), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %rsqrt_105), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_309, %arg115_1), kwargs = {})
#   %add_317 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_310, %arg116_1), kwargs = {})
triton_per_fused_add_native_layer_norm_21 = async_compile.triton('triton_per_fused_add_native_layer_norm_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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


# kernel path: /tmp/torchinductor_sahanp/vq/cvqsnzhou3miabqlc5qtbydjel5tmhk5nkhfs3ytetyqp6nw6fl4.py
# Topologically Sorted Source Nodes: [x_519], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_519 => clone_122
# Graph fragment:
#   %clone_122 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_373,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_22 = async_compile.triton('triton_poi_fused_clone_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_22(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/56/c56smsi22kbsswheilil4sp3sewlthrp6uunhsek2mjob2mohpk4.py
# Topologically Sorted Source Nodes: [x_519, conv2d_44], Original ATen: [aten.clone, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_44 => convolution_44
#   x_519 => clone_122
# Graph fragment:
#   %clone_122 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_373,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_122, %arg139_1, %arg140_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_clone_convolution_23 = async_compile.triton('triton_poi_fused_clone_convolution_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (512*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ld/cldgefrshardgvqqwbzcwkoui734oipackjwrf6kxwnmyjowukr5.py
# Topologically Sorted Source Nodes: [x_521, layer_norm_110], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_110 => add_331, add_332, mul_325, mul_326, rsqrt_110, sub_110, var_mean_110
#   x_521 => add_329, add_330, clone_123, mul_323, mul_324, rsqrt_109, sub_109, var_mean_109
# Graph fragment:
#   %clone_123 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_374,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_109 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_123, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_109 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_123, %getitem_429), kwargs = {})
#   %add_329 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_428, 1e-05), kwargs = {})
#   %rsqrt_109 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_329,), kwargs = {})
#   %mul_323 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_109, %rsqrt_109), kwargs = {})
#   %mul_324 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_323, %arg141_1), kwargs = {})
#   %add_330 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_324, %arg142_1), kwargs = {})
#   %var_mean_110 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_330, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_110 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_330, %getitem_431), kwargs = {})
#   %add_331 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_430, 1e-06), kwargs = {})
#   %rsqrt_110 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_331,), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_110, %rsqrt_110), kwargs = {})
#   %mul_326 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_325, %arg143_1), kwargs = {})
#   %add_332 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_326, %arg144_1), kwargs = {})
triton_per_fused_native_layer_norm_24 = async_compile.triton('triton_per_fused_native_layer_norm_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 8, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr5, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 320, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 320.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.broadcast_to(tmp29, [RBLOCK])
    tmp32 = tl.where(rmask, tmp30, 0)
    tmp33 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = tmp36 / tmp11
    tmp38 = tmp30 - tmp37
    tmp39 = tmp38 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [RBLOCK])
    tmp42 = tl.where(rmask, tmp40, 0)
    tmp43 = triton_helpers.promote_to_tensor(tl.sum(tmp42, 0))
    tmp44 = tmp29 - tmp37
    tmp45 = tmp43 / tmp20
    tmp46 = 1e-06
    tmp47 = tmp45 + tmp46
    tmp48 = libdevice.rsqrt(tmp47)
    tmp49 = tmp44 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp29, rmask)
    tl.store(out_ptr5 + (r1 + (320*x0)), tmp53, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gs/cgsfgomqdciaqajebabg3xzsrnkwqbjdwebpjgwtdmrizo66b7dz.py
# Topologically Sorted Source Nodes: [conv2d_45], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_45 => convolution_45
# Graph fragment:
#   %convolution_45 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_552, %arg147_1, %arg148_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_25 = async_compile.triton('triton_poi_fused_convolution_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_25(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 102400
    xnumel = 4
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (1280*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6k/c6kvshns2fvfascm33ayjrggrfhnyolyzquknm7ijhqv7uawq6bp.py
# Topologically Sorted Source Nodes: [x_525], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_525 => add_333, add_334, mul_327, mul_328, rsqrt_111, sub_111, var_mean_111
# Graph fragment:
#   %var_mean_111 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_378, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_378, %getitem_433), kwargs = {})
#   %add_333 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_432, 1e-05), kwargs = {})
#   %rsqrt_111 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_333,), kwargs = {})
#   %mul_327 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_111, %rsqrt_111), kwargs = {})
#   %mul_328 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_327, %arg149_1), kwargs = {})
#   %add_334 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_328, %arg150_1), kwargs = {})
triton_per_fused_native_layer_norm_26 = async_compile.triton('triton_per_fused_native_layer_norm_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 320, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 320.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp29, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vy/cvycezqyc4w23nzkhafgnxu4gbyplpkzgne6sjvztrqg5ewepruy.py
# Topologically Sorted Source Nodes: [x_530, layer_norm_112], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_112 => add_336, add_337, mul_329, mul_330, rsqrt_112, sub_112, var_mean_112
#   x_530 => add_335
# Graph fragment:
#   %add_335 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_330, %view_559), kwargs = {})
#   %var_mean_112 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_335, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_112 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_335, %getitem_441), kwargs = {})
#   %add_336 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_440, 1e-06), kwargs = {})
#   %rsqrt_112 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_336,), kwargs = {})
#   %mul_329 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_112, %rsqrt_112), kwargs = {})
#   %mul_330 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_329, %arg155_1), kwargs = {})
#   %add_337 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_330, %arg156_1), kwargs = {})
triton_per_fused_add_native_layer_norm_27 = async_compile.triton('triton_per_fused_add_native_layer_norm_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (320*x0)), rmask, other=0.0)
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
    tmp12 = tl.full([1], 320, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 320.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/q7/cq7cwh4as6h7zuaofw3ns4aikhmxwgccqvfvyeys5irayrptjlc7.py
# Topologically Sorted Source Nodes: [x_532], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_532 => add_338, erf_35, mul_331, mul_332, mul_333
# Graph fragment:
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_561, 0.5), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_561, 0.7071067811865476), kwargs = {})
#   %erf_35 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_332,), kwargs = {})
#   %add_338 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_35, 1), kwargs = {})
#   %mul_333 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_331, %add_338), kwargs = {})
triton_poi_fused_gelu_28 = async_compile.triton('triton_poi_fused_gelu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_28(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1280
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


# kernel path: /tmp/torchinductor_sahanp/lr/clrj2udatocmv7flli7vfzjq3cmwffrr2dlj556nfmcnssd3ek24.py
# Topologically Sorted Source Nodes: [x_530, x_536], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_530 => add_335
#   x_536 => add_339
# Graph fragment:
#   %add_335 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_330, %view_559), kwargs = {})
#   %add_339 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_335, %view_563), kwargs = {})
triton_poi_fused_add_29 = async_compile.triton('triton_poi_fused_add_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
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


# kernel path: /tmp/torchinductor_sahanp/ur/cur5ict555us77fuv46ggsqh22bnbfsroxet3rkexdcdf3e2lnii.py
# Topologically Sorted Source Nodes: [layer_norm_113], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_113 => add_341, add_342, mul_334, mul_335, rsqrt_113, sub_113, var_mean_113
# Graph fragment:
#   %var_mean_113 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_387, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_387, %getitem_443), kwargs = {})
#   %add_341 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_442, 1e-06), kwargs = {})
#   %rsqrt_113 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_341,), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %rsqrt_113), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_334, %arg163_1), kwargs = {})
#   %add_342 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_335, %arg164_1), kwargs = {})
triton_per_fused_native_layer_norm_30 = async_compile.triton('triton_per_fused_native_layer_norm_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (320*x0)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 320, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 320.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/m5/cm55h6ojtorcrskzhgzmruisv2lxcdre3ktpk54na7onzigb3gik.py
# Topologically Sorted Source Nodes: [x_546, layer_norm_115], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_115 => add_346, add_347, mul_338, mul_339, rsqrt_115, sub_115, var_mean_115
#   x_546 => add_345
# Graph fragment:
#   %add_345 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_387, %view_577), kwargs = {})
#   %var_mean_115 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_345, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_345, %getitem_453), kwargs = {})
#   %add_346 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_452, 1e-06), kwargs = {})
#   %rsqrt_115 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_346,), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %rsqrt_115), kwargs = {})
#   %mul_339 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_338, %arg175_1), kwargs = {})
#   %add_347 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_339, %arg176_1), kwargs = {})
triton_per_fused_add_native_layer_norm_31 = async_compile.triton('triton_per_fused_add_native_layer_norm_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (320*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (320*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 320, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 320.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (320*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6c/c6czo6eqfs2z4bdxge5fqqi2fvq3qutw7zxg6ujsirtu3a7mvp5m.py
# Topologically Sorted Source Nodes: [x_552, x_560, layer_norm_118], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_118 => add_355, add_356, mul_347, mul_348, rsqrt_118, sub_118, var_mean_118
#   x_552 => add_349
#   x_560 => add_354
# Graph fragment:
#   %add_349 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_345, %view_581), kwargs = {})
#   %add_354 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_349, %view_592), kwargs = {})
#   %var_mean_118 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_354, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_118 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_354, %getitem_465), kwargs = {})
#   %add_355 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_464, 1e-06), kwargs = {})
#   %rsqrt_118 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_355,), kwargs = {})
#   %mul_347 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_118, %rsqrt_118), kwargs = {})
#   %mul_348 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_347, %arg193_1), kwargs = {})
#   %add_356 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_348, %arg194_1), kwargs = {})
triton_per_fused_add_native_layer_norm_32 = async_compile.triton('triton_per_fused_add_native_layer_norm_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (320*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (320*x0)), rmask, other=0.0)
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
    tmp16 = tl.full([1], 320, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 320.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (320*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4r/c4rag6haddiis66rtstit7r4mqdlwnoxqk6nyqrqr6f6li5hmlgo.py
# Topologically Sorted Source Nodes: [x_777], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_777 => clone_179
# Graph fragment:
#   %clone_179 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_558,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_33 = async_compile.triton('triton_poi_fused_clone_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_33(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/li/cliesu4zgbmfjwwzc5vvso6ts3vjf5vgmfvz3ykrbny2k3zxtsir.py
# Topologically Sorted Source Nodes: [x_777, conv2d_64], Original ATen: [aten.clone, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_64 => convolution_64
#   x_777 => clone_179
# Graph fragment:
#   %clone_179 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_558,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_64 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_179, %arg469_1, %arg470_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_clone_convolution_34 = async_compile.triton('triton_poi_fused_clone_convolution_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[262144, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 163840
    xnumel = 4
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (1280*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4j/c4jv7cfecc7advindlh2rkdkzowksqj5mnawiz3kfrdcea2irk4f.py
# Topologically Sorted Source Nodes: [x_779, layer_norm_165], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_165 => add_496, add_497, mul_489, mul_490, rsqrt_165, sub_165, var_mean_165
#   x_779 => add_494, add_495, clone_180, mul_487, mul_488, rsqrt_164, sub_164, var_mean_164
# Graph fragment:
#   %clone_180 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_559,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_164 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_180, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_164 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_180, %getitem_647), kwargs = {})
#   %add_494 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_646, 1e-05), kwargs = {})
#   %rsqrt_164 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_494,), kwargs = {})
#   %mul_487 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_164, %rsqrt_164), kwargs = {})
#   %mul_488 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_487, %arg471_1), kwargs = {})
#   %add_495 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_488, %arg472_1), kwargs = {})
#   %var_mean_165 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_495, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_165 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_495, %getitem_649), kwargs = {})
#   %add_496 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_648, 1e-06), kwargs = {})
#   %rsqrt_165 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_496,), kwargs = {})
#   %mul_489 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_165, %rsqrt_165), kwargs = {})
#   %mul_490 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_489, %arg473_1), kwargs = {})
#   %add_497 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_490, %arg474_1), kwargs = {})
triton_per_fused_native_layer_norm_35 = async_compile.triton('triton_per_fused_native_layer_norm_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 8, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr5, xnumel, rnumel):
    xnumel = 392
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
    tmp23 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
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
    tmp16 = tmp2 - tmp10
    tmp17 = 512.0
    tmp18 = tmp15 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = tmp31 / tmp9
    tmp33 = tmp27 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp38 = tmp26 - tmp32
    tmp39 = tmp37 / tmp17
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp38 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp26, None)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp47, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dz/cdzrvotg7faiqfysg32phqelovrc6kzljkej4kqq3ygatdbozuvy.py
# Topologically Sorted Source Nodes: [x_785, layer_norm_166], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_166 => add_499, add_500, mul_491, mul_492, rsqrt_166, sub_166, var_mean_166
#   x_785 => add_498
# Graph fragment:
#   %add_498 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_495, %view_832), kwargs = {})
#   %var_mean_166 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_498, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_166 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_498, %getitem_657), kwargs = {})
#   %add_499 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_656, 1e-06), kwargs = {})
#   %rsqrt_166 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_499,), kwargs = {})
#   %mul_491 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_166, %rsqrt_166), kwargs = {})
#   %mul_492 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_491, %arg481_1), kwargs = {})
#   %add_500 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_492, %arg482_1), kwargs = {})
triton_per_fused_add_native_layer_norm_36 = async_compile.triton('triton_per_fused_add_native_layer_norm_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 392
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


# kernel path: /tmp/torchinductor_sahanp/5y/c5yr7fs33kevh5vjgc4v5ms4hx67mrnnrkl2pi7kiod4gmeycgq2.py
# Topologically Sorted Source Nodes: [x_787], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_787 => add_501, erf_53, mul_493, mul_494, mul_495
# Graph fragment:
#   %mul_493 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_834, 0.5), kwargs = {})
#   %mul_494 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_834, 0.7071067811865476), kwargs = {})
#   %erf_53 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_494,), kwargs = {})
#   %add_501 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_53, 1), kwargs = {})
#   %mul_495 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_493, %add_501), kwargs = {})
triton_poi_fused_gelu_37 = async_compile.triton('triton_poi_fused_gelu_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_37(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
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


# kernel path: /tmp/torchinductor_sahanp/cw/ccwoose24ya4wzy276i3mn3n7tflvqhdmsk2fez54upncfkb57wn.py
# Topologically Sorted Source Nodes: [x_785, x_791], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_785 => add_498
#   x_791 => add_502
# Graph fragment:
#   %add_498 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_495, %view_832), kwargs = {})
#   %add_502 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_498, %view_836), kwargs = {})
triton_poi_fused_add_38 = async_compile.triton('triton_poi_fused_add_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ol/colq5yepf2wd4giw3ipvflov34cwkzxxqyyoszdnbfuo5pqci2rm.py
# Topologically Sorted Source Nodes: [layer_norm_167], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_167 => add_504, add_505, mul_496, mul_497, rsqrt_167, sub_167, var_mean_167
# Graph fragment:
#   %var_mean_167 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_570, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_167 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_570, %getitem_659), kwargs = {})
#   %add_504 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_658, 1e-06), kwargs = {})
#   %rsqrt_167 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_504,), kwargs = {})
#   %mul_496 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_167, %rsqrt_167), kwargs = {})
#   %mul_497 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_496, %arg489_1), kwargs = {})
#   %add_505 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_497, %arg490_1), kwargs = {})
triton_per_fused_native_layer_norm_39 = async_compile.triton('triton_per_fused_native_layer_norm_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 392
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
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), None)
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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


# kernel path: /tmp/torchinductor_sahanp/4g/c4g33ftabvh74kgwzkgcqcyreiotmzmwfu435kvluu44xrhp5yum.py
# Topologically Sorted Source Nodes: [x_798, layer_norm_168], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_168 => add_507, add_508, mul_498, mul_499, rsqrt_168, sub_168, var_mean_168
#   x_798 => add_506
# Graph fragment:
#   %add_506 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_570, %view_848), kwargs = {})
#   %var_mean_168 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_506, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_168 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_506, %getitem_667), kwargs = {})
#   %add_507 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_666, 1e-06), kwargs = {})
#   %rsqrt_168 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_507,), kwargs = {})
#   %mul_498 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_168, %rsqrt_168), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_498, %arg497_1), kwargs = {})
#   %add_508 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_499, %arg498_1), kwargs = {})
triton_per_fused_add_native_layer_norm_40 = async_compile.triton('triton_per_fused_add_native_layer_norm_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 392
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
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), None)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), None)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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


# kernel path: /tmp/torchinductor_sahanp/te/cteefk7sbtc66wvvuitufloblr4mfhmphrmtrmcqnnjzyiullkoj.py
# Topologically Sorted Source Nodes: [x_804, x_809, layer_norm_170], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_170 => add_514, add_515, mul_505, mul_506, rsqrt_170, sub_170, var_mean_170
#   x_804 => add_510
#   x_809 => add_513
# Graph fragment:
#   %add_510 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_506, %view_852), kwargs = {})
#   %add_513 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_510, %view_861), kwargs = {})
#   %var_mean_170 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_513, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_170 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_513, %getitem_677), kwargs = {})
#   %add_514 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_676, 1e-06), kwargs = {})
#   %rsqrt_170 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_514,), kwargs = {})
#   %mul_505 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_170, %rsqrt_170), kwargs = {})
#   %mul_506 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_505, %arg511_1), kwargs = {})
#   %add_515 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_506, %arg512_1), kwargs = {})
triton_per_fused_add_native_layer_norm_41 = async_compile.triton('triton_per_fused_add_native_layer_norm_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 392
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


# kernel path: /tmp/torchinductor_sahanp/ra/cray5pcxnnkhrf53qsszvfadnaatw5mebcb7qf2kb3n6tlsmzk53.py
# Topologically Sorted Source Nodes: [x_815, x_816], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_815 => add_517
#   x_816 => var_mean_171
# Graph fragment:
#   %add_517 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_513, %view_865), kwargs = {})
#   %var_mean_171 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_517, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_42 = async_compile.triton('triton_per_fused_add_native_layer_norm_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_42(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 392
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
    tl.store(out_ptr0 + (x0), tmp12, None)
    tl.store(out_ptr1 + (x0), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/je/cjex6jivgdxx4zncvzboyvcsfzqxivqwgl2pbmcmkixb5zs7ozkc.py
# Topologically Sorted Source Nodes: [x_815, x_816, x_817], Original ATen: [aten.add, aten.native_layer_norm, aten.mean]
# Source node to ATen node mapping:
#   x_815 => add_517
#   x_816 => add_518, add_519, mul_510, mul_511, rsqrt_171, sub_171, var_mean_171
#   x_817 => mean_1
# Graph fragment:
#   %add_517 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_513, %view_865), kwargs = {})
#   %var_mean_171 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_517, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_171 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_517, %getitem_679), kwargs = {})
#   %add_518 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_678, 1e-06), kwargs = {})
#   %rsqrt_171 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_518,), kwargs = {})
#   %mul_510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_171, %rsqrt_171), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_510, %arg517_1), kwargs = {})
#   %add_519 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_511, %arg518_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_519, [1]), kwargs = {})
triton_per_fused_add_mean_native_layer_norm_43 = async_compile.triton('triton_per_fused_add_mean_native_layer_norm_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_native_layer_norm_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_native_layer_norm_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (25088*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (25088*x1)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (r2 + (49*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r2 + (49*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 512.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 49.0
    tmp23 = tmp21 / tmp22
    tl.store(out_ptr1 + (x3), tmp23, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, 64), (64, 1))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (128, 64), (64, 1))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (64, 64), (64, 1))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg19_1, (512, 64), (64, 1))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (64, 512), (512, 1))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, ), (1, ))
    assert_size_stride(arg27_1, (64, 64), (64, 1))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(arg30_1, (64, ), (1, ))
    assert_size_stride(arg31_1, (64, ), (1, ))
    assert_size_stride(arg32_1, (64, ), (1, ))
    assert_size_stride(arg33_1, (128, 64), (64, 1))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (64, 64), (64, 1))
    assert_size_stride(arg36_1, (64, ), (1, ))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (64, ), (1, ))
    assert_size_stride(arg39_1, (512, 64), (64, 1))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (64, 512), (512, 1))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, 64), (64, 1))
    assert_size_stride(arg46_1, (64, ), (1, ))
    assert_size_stride(arg47_1, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(arg48_1, (64, ), (1, ))
    assert_size_stride(arg49_1, (64, ), (1, ))
    assert_size_stride(arg50_1, (64, ), (1, ))
    assert_size_stride(arg51_1, (128, 64), (64, 1))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (64, 64), (64, 1))
    assert_size_stride(arg54_1, (64, ), (1, ))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (64, ), (1, ))
    assert_size_stride(arg57_1, (512, 64), (64, 1))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (64, 512), (512, 1))
    assert_size_stride(arg60_1, (64, ), (1, ))
    assert_size_stride(arg61_1, (128, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (128, ), (1, ))
    assert_size_stride(arg67_1, (128, 128), (128, 1))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, ), (1, ))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (256, 128), (128, 1))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (128, 128), (128, 1))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (1024, 128), (128, 1))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (128, 1024), (1024, 1))
    assert_size_stride(arg82_1, (128, ), (1, ))
    assert_size_stride(arg83_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, ), (1, ))
    assert_size_stride(arg87_1, (128, 128), (128, 1))
    assert_size_stride(arg88_1, (128, ), (1, ))
    assert_size_stride(arg89_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (256, 128), (128, 1))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (128, 128), (128, 1))
    assert_size_stride(arg96_1, (128, ), (1, ))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (1024, 128), (128, 1))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (128, 1024), (1024, 1))
    assert_size_stride(arg102_1, (128, ), (1, ))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, ), (1, ))
    assert_size_stride(arg105_1, (128, 128), (128, 1))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (256, 128), (128, 1))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (128, 128), (128, 1))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (128, ), (1, ))
    assert_size_stride(arg117_1, (1024, 128), (128, 1))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (128, 1024), (1024, 1))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (128, 128), (128, 1))
    assert_size_stride(arg124_1, (128, ), (1, ))
    assert_size_stride(arg125_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (128, ), (1, ))
    assert_size_stride(arg129_1, (256, 128), (128, 1))
    assert_size_stride(arg130_1, (256, ), (1, ))
    assert_size_stride(arg131_1, (128, 128), (128, 1))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (128, ), (1, ))
    assert_size_stride(arg135_1, (1024, 128), (128, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (128, 1024), (1024, 1))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (320, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(arg140_1, (320, ), (1, ))
    assert_size_stride(arg141_1, (320, ), (1, ))
    assert_size_stride(arg142_1, (320, ), (1, ))
    assert_size_stride(arg143_1, (320, ), (1, ))
    assert_size_stride(arg144_1, (320, ), (1, ))
    assert_size_stride(arg145_1, (320, 320), (320, 1))
    assert_size_stride(arg146_1, (320, ), (1, ))
    assert_size_stride(arg147_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg148_1, (320, ), (1, ))
    assert_size_stride(arg149_1, (320, ), (1, ))
    assert_size_stride(arg150_1, (320, ), (1, ))
    assert_size_stride(arg151_1, (640, 320), (320, 1))
    assert_size_stride(arg152_1, (640, ), (1, ))
    assert_size_stride(arg153_1, (320, 320), (320, 1))
    assert_size_stride(arg154_1, (320, ), (1, ))
    assert_size_stride(arg155_1, (320, ), (1, ))
    assert_size_stride(arg156_1, (320, ), (1, ))
    assert_size_stride(arg157_1, (1280, 320), (320, 1))
    assert_size_stride(arg158_1, (1280, ), (1, ))
    assert_size_stride(arg159_1, (320, 1280), (1280, 1))
    assert_size_stride(arg160_1, (320, ), (1, ))
    assert_size_stride(arg161_1, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg162_1, (320, ), (1, ))
    assert_size_stride(arg163_1, (320, ), (1, ))
    assert_size_stride(arg164_1, (320, ), (1, ))
    assert_size_stride(arg165_1, (320, 320), (320, 1))
    assert_size_stride(arg166_1, (320, ), (1, ))
    assert_size_stride(arg167_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg168_1, (320, ), (1, ))
    assert_size_stride(arg169_1, (320, ), (1, ))
    assert_size_stride(arg170_1, (320, ), (1, ))
    assert_size_stride(arg171_1, (640, 320), (320, 1))
    assert_size_stride(arg172_1, (640, ), (1, ))
    assert_size_stride(arg173_1, (320, 320), (320, 1))
    assert_size_stride(arg174_1, (320, ), (1, ))
    assert_size_stride(arg175_1, (320, ), (1, ))
    assert_size_stride(arg176_1, (320, ), (1, ))
    assert_size_stride(arg177_1, (1280, 320), (320, 1))
    assert_size_stride(arg178_1, (1280, ), (1, ))
    assert_size_stride(arg179_1, (320, 1280), (1280, 1))
    assert_size_stride(arg180_1, (320, ), (1, ))
    assert_size_stride(arg181_1, (320, ), (1, ))
    assert_size_stride(arg182_1, (320, ), (1, ))
    assert_size_stride(arg183_1, (320, 320), (320, 1))
    assert_size_stride(arg184_1, (320, ), (1, ))
    assert_size_stride(arg185_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg186_1, (320, ), (1, ))
    assert_size_stride(arg187_1, (320, ), (1, ))
    assert_size_stride(arg188_1, (320, ), (1, ))
    assert_size_stride(arg189_1, (640, 320), (320, 1))
    assert_size_stride(arg190_1, (640, ), (1, ))
    assert_size_stride(arg191_1, (320, 320), (320, 1))
    assert_size_stride(arg192_1, (320, ), (1, ))
    assert_size_stride(arg193_1, (320, ), (1, ))
    assert_size_stride(arg194_1, (320, ), (1, ))
    assert_size_stride(arg195_1, (1280, 320), (320, 1))
    assert_size_stride(arg196_1, (1280, ), (1, ))
    assert_size_stride(arg197_1, (320, 1280), (1280, 1))
    assert_size_stride(arg198_1, (320, ), (1, ))
    assert_size_stride(arg199_1, (320, ), (1, ))
    assert_size_stride(arg200_1, (320, ), (1, ))
    assert_size_stride(arg201_1, (320, 320), (320, 1))
    assert_size_stride(arg202_1, (320, ), (1, ))
    assert_size_stride(arg203_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg204_1, (320, ), (1, ))
    assert_size_stride(arg205_1, (320, ), (1, ))
    assert_size_stride(arg206_1, (320, ), (1, ))
    assert_size_stride(arg207_1, (640, 320), (320, 1))
    assert_size_stride(arg208_1, (640, ), (1, ))
    assert_size_stride(arg209_1, (320, 320), (320, 1))
    assert_size_stride(arg210_1, (320, ), (1, ))
    assert_size_stride(arg211_1, (320, ), (1, ))
    assert_size_stride(arg212_1, (320, ), (1, ))
    assert_size_stride(arg213_1, (1280, 320), (320, 1))
    assert_size_stride(arg214_1, (1280, ), (1, ))
    assert_size_stride(arg215_1, (320, 1280), (1280, 1))
    assert_size_stride(arg216_1, (320, ), (1, ))
    assert_size_stride(arg217_1, (320, ), (1, ))
    assert_size_stride(arg218_1, (320, ), (1, ))
    assert_size_stride(arg219_1, (320, 320), (320, 1))
    assert_size_stride(arg220_1, (320, ), (1, ))
    assert_size_stride(arg221_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg222_1, (320, ), (1, ))
    assert_size_stride(arg223_1, (320, ), (1, ))
    assert_size_stride(arg224_1, (320, ), (1, ))
    assert_size_stride(arg225_1, (640, 320), (320, 1))
    assert_size_stride(arg226_1, (640, ), (1, ))
    assert_size_stride(arg227_1, (320, 320), (320, 1))
    assert_size_stride(arg228_1, (320, ), (1, ))
    assert_size_stride(arg229_1, (320, ), (1, ))
    assert_size_stride(arg230_1, (320, ), (1, ))
    assert_size_stride(arg231_1, (1280, 320), (320, 1))
    assert_size_stride(arg232_1, (1280, ), (1, ))
    assert_size_stride(arg233_1, (320, 1280), (1280, 1))
    assert_size_stride(arg234_1, (320, ), (1, ))
    assert_size_stride(arg235_1, (320, ), (1, ))
    assert_size_stride(arg236_1, (320, ), (1, ))
    assert_size_stride(arg237_1, (320, 320), (320, 1))
    assert_size_stride(arg238_1, (320, ), (1, ))
    assert_size_stride(arg239_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg240_1, (320, ), (1, ))
    assert_size_stride(arg241_1, (320, ), (1, ))
    assert_size_stride(arg242_1, (320, ), (1, ))
    assert_size_stride(arg243_1, (640, 320), (320, 1))
    assert_size_stride(arg244_1, (640, ), (1, ))
    assert_size_stride(arg245_1, (320, 320), (320, 1))
    assert_size_stride(arg246_1, (320, ), (1, ))
    assert_size_stride(arg247_1, (320, ), (1, ))
    assert_size_stride(arg248_1, (320, ), (1, ))
    assert_size_stride(arg249_1, (1280, 320), (320, 1))
    assert_size_stride(arg250_1, (1280, ), (1, ))
    assert_size_stride(arg251_1, (320, 1280), (1280, 1))
    assert_size_stride(arg252_1, (320, ), (1, ))
    assert_size_stride(arg253_1, (320, ), (1, ))
    assert_size_stride(arg254_1, (320, ), (1, ))
    assert_size_stride(arg255_1, (320, 320), (320, 1))
    assert_size_stride(arg256_1, (320, ), (1, ))
    assert_size_stride(arg257_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg258_1, (320, ), (1, ))
    assert_size_stride(arg259_1, (320, ), (1, ))
    assert_size_stride(arg260_1, (320, ), (1, ))
    assert_size_stride(arg261_1, (640, 320), (320, 1))
    assert_size_stride(arg262_1, (640, ), (1, ))
    assert_size_stride(arg263_1, (320, 320), (320, 1))
    assert_size_stride(arg264_1, (320, ), (1, ))
    assert_size_stride(arg265_1, (320, ), (1, ))
    assert_size_stride(arg266_1, (320, ), (1, ))
    assert_size_stride(arg267_1, (1280, 320), (320, 1))
    assert_size_stride(arg268_1, (1280, ), (1, ))
    assert_size_stride(arg269_1, (320, 1280), (1280, 1))
    assert_size_stride(arg270_1, (320, ), (1, ))
    assert_size_stride(arg271_1, (320, ), (1, ))
    assert_size_stride(arg272_1, (320, ), (1, ))
    assert_size_stride(arg273_1, (320, 320), (320, 1))
    assert_size_stride(arg274_1, (320, ), (1, ))
    assert_size_stride(arg275_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg276_1, (320, ), (1, ))
    assert_size_stride(arg277_1, (320, ), (1, ))
    assert_size_stride(arg278_1, (320, ), (1, ))
    assert_size_stride(arg279_1, (640, 320), (320, 1))
    assert_size_stride(arg280_1, (640, ), (1, ))
    assert_size_stride(arg281_1, (320, 320), (320, 1))
    assert_size_stride(arg282_1, (320, ), (1, ))
    assert_size_stride(arg283_1, (320, ), (1, ))
    assert_size_stride(arg284_1, (320, ), (1, ))
    assert_size_stride(arg285_1, (1280, 320), (320, 1))
    assert_size_stride(arg286_1, (1280, ), (1, ))
    assert_size_stride(arg287_1, (320, 1280), (1280, 1))
    assert_size_stride(arg288_1, (320, ), (1, ))
    assert_size_stride(arg289_1, (320, ), (1, ))
    assert_size_stride(arg290_1, (320, ), (1, ))
    assert_size_stride(arg291_1, (320, 320), (320, 1))
    assert_size_stride(arg292_1, (320, ), (1, ))
    assert_size_stride(arg293_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg294_1, (320, ), (1, ))
    assert_size_stride(arg295_1, (320, ), (1, ))
    assert_size_stride(arg296_1, (320, ), (1, ))
    assert_size_stride(arg297_1, (640, 320), (320, 1))
    assert_size_stride(arg298_1, (640, ), (1, ))
    assert_size_stride(arg299_1, (320, 320), (320, 1))
    assert_size_stride(arg300_1, (320, ), (1, ))
    assert_size_stride(arg301_1, (320, ), (1, ))
    assert_size_stride(arg302_1, (320, ), (1, ))
    assert_size_stride(arg303_1, (1280, 320), (320, 1))
    assert_size_stride(arg304_1, (1280, ), (1, ))
    assert_size_stride(arg305_1, (320, 1280), (1280, 1))
    assert_size_stride(arg306_1, (320, ), (1, ))
    assert_size_stride(arg307_1, (320, ), (1, ))
    assert_size_stride(arg308_1, (320, ), (1, ))
    assert_size_stride(arg309_1, (320, 320), (320, 1))
    assert_size_stride(arg310_1, (320, ), (1, ))
    assert_size_stride(arg311_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg312_1, (320, ), (1, ))
    assert_size_stride(arg313_1, (320, ), (1, ))
    assert_size_stride(arg314_1, (320, ), (1, ))
    assert_size_stride(arg315_1, (640, 320), (320, 1))
    assert_size_stride(arg316_1, (640, ), (1, ))
    assert_size_stride(arg317_1, (320, 320), (320, 1))
    assert_size_stride(arg318_1, (320, ), (1, ))
    assert_size_stride(arg319_1, (320, ), (1, ))
    assert_size_stride(arg320_1, (320, ), (1, ))
    assert_size_stride(arg321_1, (1280, 320), (320, 1))
    assert_size_stride(arg322_1, (1280, ), (1, ))
    assert_size_stride(arg323_1, (320, 1280), (1280, 1))
    assert_size_stride(arg324_1, (320, ), (1, ))
    assert_size_stride(arg325_1, (320, ), (1, ))
    assert_size_stride(arg326_1, (320, ), (1, ))
    assert_size_stride(arg327_1, (320, 320), (320, 1))
    assert_size_stride(arg328_1, (320, ), (1, ))
    assert_size_stride(arg329_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg330_1, (320, ), (1, ))
    assert_size_stride(arg331_1, (320, ), (1, ))
    assert_size_stride(arg332_1, (320, ), (1, ))
    assert_size_stride(arg333_1, (640, 320), (320, 1))
    assert_size_stride(arg334_1, (640, ), (1, ))
    assert_size_stride(arg335_1, (320, 320), (320, 1))
    assert_size_stride(arg336_1, (320, ), (1, ))
    assert_size_stride(arg337_1, (320, ), (1, ))
    assert_size_stride(arg338_1, (320, ), (1, ))
    assert_size_stride(arg339_1, (1280, 320), (320, 1))
    assert_size_stride(arg340_1, (1280, ), (1, ))
    assert_size_stride(arg341_1, (320, 1280), (1280, 1))
    assert_size_stride(arg342_1, (320, ), (1, ))
    assert_size_stride(arg343_1, (320, ), (1, ))
    assert_size_stride(arg344_1, (320, ), (1, ))
    assert_size_stride(arg345_1, (320, 320), (320, 1))
    assert_size_stride(arg346_1, (320, ), (1, ))
    assert_size_stride(arg347_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg348_1, (320, ), (1, ))
    assert_size_stride(arg349_1, (320, ), (1, ))
    assert_size_stride(arg350_1, (320, ), (1, ))
    assert_size_stride(arg351_1, (640, 320), (320, 1))
    assert_size_stride(arg352_1, (640, ), (1, ))
    assert_size_stride(arg353_1, (320, 320), (320, 1))
    assert_size_stride(arg354_1, (320, ), (1, ))
    assert_size_stride(arg355_1, (320, ), (1, ))
    assert_size_stride(arg356_1, (320, ), (1, ))
    assert_size_stride(arg357_1, (1280, 320), (320, 1))
    assert_size_stride(arg358_1, (1280, ), (1, ))
    assert_size_stride(arg359_1, (320, 1280), (1280, 1))
    assert_size_stride(arg360_1, (320, ), (1, ))
    assert_size_stride(arg361_1, (320, ), (1, ))
    assert_size_stride(arg362_1, (320, ), (1, ))
    assert_size_stride(arg363_1, (320, 320), (320, 1))
    assert_size_stride(arg364_1, (320, ), (1, ))
    assert_size_stride(arg365_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg366_1, (320, ), (1, ))
    assert_size_stride(arg367_1, (320, ), (1, ))
    assert_size_stride(arg368_1, (320, ), (1, ))
    assert_size_stride(arg369_1, (640, 320), (320, 1))
    assert_size_stride(arg370_1, (640, ), (1, ))
    assert_size_stride(arg371_1, (320, 320), (320, 1))
    assert_size_stride(arg372_1, (320, ), (1, ))
    assert_size_stride(arg373_1, (320, ), (1, ))
    assert_size_stride(arg374_1, (320, ), (1, ))
    assert_size_stride(arg375_1, (1280, 320), (320, 1))
    assert_size_stride(arg376_1, (1280, ), (1, ))
    assert_size_stride(arg377_1, (320, 1280), (1280, 1))
    assert_size_stride(arg378_1, (320, ), (1, ))
    assert_size_stride(arg379_1, (320, ), (1, ))
    assert_size_stride(arg380_1, (320, ), (1, ))
    assert_size_stride(arg381_1, (320, 320), (320, 1))
    assert_size_stride(arg382_1, (320, ), (1, ))
    assert_size_stride(arg383_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg384_1, (320, ), (1, ))
    assert_size_stride(arg385_1, (320, ), (1, ))
    assert_size_stride(arg386_1, (320, ), (1, ))
    assert_size_stride(arg387_1, (640, 320), (320, 1))
    assert_size_stride(arg388_1, (640, ), (1, ))
    assert_size_stride(arg389_1, (320, 320), (320, 1))
    assert_size_stride(arg390_1, (320, ), (1, ))
    assert_size_stride(arg391_1, (320, ), (1, ))
    assert_size_stride(arg392_1, (320, ), (1, ))
    assert_size_stride(arg393_1, (1280, 320), (320, 1))
    assert_size_stride(arg394_1, (1280, ), (1, ))
    assert_size_stride(arg395_1, (320, 1280), (1280, 1))
    assert_size_stride(arg396_1, (320, ), (1, ))
    assert_size_stride(arg397_1, (320, ), (1, ))
    assert_size_stride(arg398_1, (320, ), (1, ))
    assert_size_stride(arg399_1, (320, 320), (320, 1))
    assert_size_stride(arg400_1, (320, ), (1, ))
    assert_size_stride(arg401_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg402_1, (320, ), (1, ))
    assert_size_stride(arg403_1, (320, ), (1, ))
    assert_size_stride(arg404_1, (320, ), (1, ))
    assert_size_stride(arg405_1, (640, 320), (320, 1))
    assert_size_stride(arg406_1, (640, ), (1, ))
    assert_size_stride(arg407_1, (320, 320), (320, 1))
    assert_size_stride(arg408_1, (320, ), (1, ))
    assert_size_stride(arg409_1, (320, ), (1, ))
    assert_size_stride(arg410_1, (320, ), (1, ))
    assert_size_stride(arg411_1, (1280, 320), (320, 1))
    assert_size_stride(arg412_1, (1280, ), (1, ))
    assert_size_stride(arg413_1, (320, 1280), (1280, 1))
    assert_size_stride(arg414_1, (320, ), (1, ))
    assert_size_stride(arg415_1, (320, ), (1, ))
    assert_size_stride(arg416_1, (320, ), (1, ))
    assert_size_stride(arg417_1, (320, 320), (320, 1))
    assert_size_stride(arg418_1, (320, ), (1, ))
    assert_size_stride(arg419_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg420_1, (320, ), (1, ))
    assert_size_stride(arg421_1, (320, ), (1, ))
    assert_size_stride(arg422_1, (320, ), (1, ))
    assert_size_stride(arg423_1, (640, 320), (320, 1))
    assert_size_stride(arg424_1, (640, ), (1, ))
    assert_size_stride(arg425_1, (320, 320), (320, 1))
    assert_size_stride(arg426_1, (320, ), (1, ))
    assert_size_stride(arg427_1, (320, ), (1, ))
    assert_size_stride(arg428_1, (320, ), (1, ))
    assert_size_stride(arg429_1, (1280, 320), (320, 1))
    assert_size_stride(arg430_1, (1280, ), (1, ))
    assert_size_stride(arg431_1, (320, 1280), (1280, 1))
    assert_size_stride(arg432_1, (320, ), (1, ))
    assert_size_stride(arg433_1, (320, ), (1, ))
    assert_size_stride(arg434_1, (320, ), (1, ))
    assert_size_stride(arg435_1, (320, 320), (320, 1))
    assert_size_stride(arg436_1, (320, ), (1, ))
    assert_size_stride(arg437_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg438_1, (320, ), (1, ))
    assert_size_stride(arg439_1, (320, ), (1, ))
    assert_size_stride(arg440_1, (320, ), (1, ))
    assert_size_stride(arg441_1, (640, 320), (320, 1))
    assert_size_stride(arg442_1, (640, ), (1, ))
    assert_size_stride(arg443_1, (320, 320), (320, 1))
    assert_size_stride(arg444_1, (320, ), (1, ))
    assert_size_stride(arg445_1, (320, ), (1, ))
    assert_size_stride(arg446_1, (320, ), (1, ))
    assert_size_stride(arg447_1, (1280, 320), (320, 1))
    assert_size_stride(arg448_1, (1280, ), (1, ))
    assert_size_stride(arg449_1, (320, 1280), (1280, 1))
    assert_size_stride(arg450_1, (320, ), (1, ))
    assert_size_stride(arg451_1, (320, ), (1, ))
    assert_size_stride(arg452_1, (320, ), (1, ))
    assert_size_stride(arg453_1, (320, 320), (320, 1))
    assert_size_stride(arg454_1, (320, ), (1, ))
    assert_size_stride(arg455_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg456_1, (320, ), (1, ))
    assert_size_stride(arg457_1, (320, ), (1, ))
    assert_size_stride(arg458_1, (320, ), (1, ))
    assert_size_stride(arg459_1, (640, 320), (320, 1))
    assert_size_stride(arg460_1, (640, ), (1, ))
    assert_size_stride(arg461_1, (320, 320), (320, 1))
    assert_size_stride(arg462_1, (320, ), (1, ))
    assert_size_stride(arg463_1, (320, ), (1, ))
    assert_size_stride(arg464_1, (320, ), (1, ))
    assert_size_stride(arg465_1, (1280, 320), (320, 1))
    assert_size_stride(arg466_1, (1280, ), (1, ))
    assert_size_stride(arg467_1, (320, 1280), (1280, 1))
    assert_size_stride(arg468_1, (320, ), (1, ))
    assert_size_stride(arg469_1, (512, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg470_1, (512, ), (1, ))
    assert_size_stride(arg471_1, (512, ), (1, ))
    assert_size_stride(arg472_1, (512, ), (1, ))
    assert_size_stride(arg473_1, (512, ), (1, ))
    assert_size_stride(arg474_1, (512, ), (1, ))
    assert_size_stride(arg475_1, (512, 512), (512, 1))
    assert_size_stride(arg476_1, (512, ), (1, ))
    assert_size_stride(arg477_1, (1024, 512), (512, 1))
    assert_size_stride(arg478_1, (1024, ), (1, ))
    assert_size_stride(arg479_1, (512, 512), (512, 1))
    assert_size_stride(arg480_1, (512, ), (1, ))
    assert_size_stride(arg481_1, (512, ), (1, ))
    assert_size_stride(arg482_1, (512, ), (1, ))
    assert_size_stride(arg483_1, (2048, 512), (512, 1))
    assert_size_stride(arg484_1, (2048, ), (1, ))
    assert_size_stride(arg485_1, (512, 2048), (2048, 1))
    assert_size_stride(arg486_1, (512, ), (1, ))
    assert_size_stride(arg487_1, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg488_1, (512, ), (1, ))
    assert_size_stride(arg489_1, (512, ), (1, ))
    assert_size_stride(arg490_1, (512, ), (1, ))
    assert_size_stride(arg491_1, (512, 512), (512, 1))
    assert_size_stride(arg492_1, (512, ), (1, ))
    assert_size_stride(arg493_1, (1024, 512), (512, 1))
    assert_size_stride(arg494_1, (1024, ), (1, ))
    assert_size_stride(arg495_1, (512, 512), (512, 1))
    assert_size_stride(arg496_1, (512, ), (1, ))
    assert_size_stride(arg497_1, (512, ), (1, ))
    assert_size_stride(arg498_1, (512, ), (1, ))
    assert_size_stride(arg499_1, (2048, 512), (512, 1))
    assert_size_stride(arg500_1, (2048, ), (1, ))
    assert_size_stride(arg501_1, (512, 2048), (2048, 1))
    assert_size_stride(arg502_1, (512, ), (1, ))
    assert_size_stride(arg503_1, (512, ), (1, ))
    assert_size_stride(arg504_1, (512, ), (1, ))
    assert_size_stride(arg505_1, (512, 512), (512, 1))
    assert_size_stride(arg506_1, (512, ), (1, ))
    assert_size_stride(arg507_1, (1024, 512), (512, 1))
    assert_size_stride(arg508_1, (1024, ), (1, ))
    assert_size_stride(arg509_1, (512, 512), (512, 1))
    assert_size_stride(arg510_1, (512, ), (1, ))
    assert_size_stride(arg511_1, (512, ), (1, ))
    assert_size_stride(arg512_1, (512, ), (1, ))
    assert_size_stride(arg513_1, (2048, 512), (512, 1))
    assert_size_stride(arg514_1, (2048, ), (1, ))
    assert_size_stride(arg515_1, (512, 2048), (2048, 1))
    assert_size_stride(arg516_1, (512, ), (1, ))
    assert_size_stride(arg517_1, (512, ), (1, ))
    assert_size_stride(arg518_1, (512, ), (1, ))
    assert_size_stride(arg519_1, (1000, 512), (512, 1))
    assert_size_stride(arg520_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg0_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((64, 3, 4, 4), (48, 1, 12, 3), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg1_1, buf1, 192, 16, grid=grid(192, 16), stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del buf0
        del buf1
        buf6 = empty_strided_cuda((8, 3136, 64), (200704, 64, 1), torch.float32)
        buf10 = empty_strided_cuda((8, 3136, 64), (200704, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_411, layer_norm_87], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2.run(buf2, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, buf6, buf10, 25088, 64, grid=grid(25088), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        buf11 = empty_strided_cuda((64, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_34], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_3.run(arg9_1, buf11, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del arg9_1
        # Topologically Sorted Source Nodes: [conv2d_34], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(reinterpret_tensor(buf10, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), buf11, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 64, 7, 7), (3136, 1, 448, 64))
        buf16 = empty_strided_cuda((8, 49, 64), (3136, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_415], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_4.run(buf12, arg10_1, arg11_1, arg12_1, buf16, 392, 64, grid=grid(392), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del buf12
        buf17 = empty_strided_cuda((392, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_142], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg14_1, reinterpret_tensor(buf16, (392, 64), (64, 1), 0), reinterpret_tensor(arg13_1, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf17)
        del arg13_1
        del arg14_1
        buf18 = reinterpret_tensor(buf2, (25088, 64), (64, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [linear_141], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf10, (25088, 64), (64, 1), 0), reinterpret_tensor(arg7_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf18)
        del arg7_1
        del arg8_1
        del buf10
        # Topologically Sorted Source Nodes: [x_416], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf18, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf17, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf17, (8, 1, 49, 64), (6272, 0, 128, 1), 64), None, False)
        buf20 = buf19[0]
        del buf19
        buf24 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (25088, 64), (64, 1), 0), reinterpret_tensor(arg15_1, (64, 64), (1, 64), 0), out=buf24)
        del arg15_1
        buf28 = reinterpret_tensor(buf20, (8, 3136, 64), (200704, 64, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_420, layer_norm_89], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf6, buf24, arg16_1, arg17_1, arg18_1, buf28, 25088, 64, grid=grid(25088), stream=stream0)
        del arg17_1
        del arg18_1
        buf29 = empty_strided_cuda((25088, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (25088, 64), (64, 1), 0), reinterpret_tensor(arg19_1, (64, 512), (1, 64), 0), out=buf29)
        del arg19_1
        buf30 = reinterpret_tensor(buf29, (8, 3136, 512), (1605632, 512, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_422], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf30, arg20_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg20_1
        buf31 = reinterpret_tensor(buf28, (25088, 64), (64, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (25088, 512), (512, 1), 0), reinterpret_tensor(arg21_1, (512, 64), (1, 512), 0), out=buf31)
        del arg21_1
        buf32 = reinterpret_tensor(buf31, (8, 3136, 64), (200704, 64, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_420, x_426], Original ATen: [aten.add]
        triton_poi_fused_add_7.run(buf32, buf6, buf24, arg16_1, arg22_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg16_1
        del arg22_1
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(reinterpret_tensor(buf32, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), arg23_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf33, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg23_1
        buf37 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_90], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_8.run(buf33, arg24_1, buf32, arg25_1, arg26_1, buf37, 25088, 64, grid=grid(25088), stream=stream0)
        del arg25_1
        del arg26_1
        buf38 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [conv2d_36], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_3.run(arg29_1, buf38, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del arg29_1
        # Topologically Sorted Source Nodes: [conv2d_36], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(reinterpret_tensor(buf37, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), buf38, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 64, 7, 7), (3136, 1, 448, 64))
        buf43 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_431], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_4.run(buf39, arg30_1, arg31_1, arg32_1, buf43, 392, 64, grid=grid(392), stream=stream0)
        del arg30_1
        del arg31_1
        del arg32_1
        del buf39
        buf44 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [linear_147], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg34_1, reinterpret_tensor(buf43, (392, 64), (64, 1), 0), reinterpret_tensor(arg33_1, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf44)
        del arg33_1
        del arg34_1
        buf45 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [linear_146], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg28_1, reinterpret_tensor(buf37, (25088, 64), (64, 1), 0), reinterpret_tensor(arg27_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf45)
        del arg27_1
        del arg28_1
        del buf37
        # Topologically Sorted Source Nodes: [x_432], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf46 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf45, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf44, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf44, (8, 1, 49, 64), (6272, 0, 128, 1), 64), None, False)
        buf47 = buf46[0]
        del buf46
        buf51 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (25088, 64), (64, 1), 0), reinterpret_tensor(arg35_1, (64, 64), (1, 64), 0), out=buf51)
        del arg35_1
        buf52 = reinterpret_tensor(buf51, (8, 3136, 64), (200704, 64, 1), 0); del buf51  # reuse
        buf56 = reinterpret_tensor(buf47, (8, 3136, 64), (200704, 64, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_436, layer_norm_92], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf52, buf33, arg24_1, buf32, arg36_1, arg37_1, arg38_1, buf56, 25088, 64, grid=grid(25088), stream=stream0)
        del arg24_1
        del arg36_1
        del arg37_1
        del arg38_1
        buf57 = reinterpret_tensor(buf30, (25088, 512), (512, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (25088, 64), (64, 1), 0), reinterpret_tensor(arg39_1, (64, 512), (1, 64), 0), out=buf57)
        del arg39_1
        buf58 = reinterpret_tensor(buf57, (8, 3136, 512), (1605632, 512, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_438], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf58, arg40_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg40_1
        buf59 = reinterpret_tensor(buf56, (25088, 64), (64, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (25088, 512), (512, 1), 0), reinterpret_tensor(arg41_1, (512, 64), (1, 512), 0), out=buf59)
        del arg41_1
        buf63 = reinterpret_tensor(buf33, (8, 3136, 64), (200704, 64, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_442, layer_norm_93], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf52, buf59, arg42_1, arg43_1, arg44_1, buf63, 25088, 64, grid=grid(25088), stream=stream0)
        del arg43_1
        del arg44_1
        buf64 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [conv2d_37], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_3.run(arg47_1, buf64, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del arg47_1
        # Topologically Sorted Source Nodes: [conv2d_37], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(reinterpret_tensor(buf63, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), buf64, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 64, 7, 7), (3136, 1, 448, 64))
        buf69 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_445], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_4.run(buf65, arg48_1, arg49_1, arg50_1, buf69, 392, 64, grid=grid(392), stream=stream0)
        del arg48_1
        del arg49_1
        del arg50_1
        del buf65
        buf70 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [linear_152], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg52_1, reinterpret_tensor(buf69, (392, 64), (64, 1), 0), reinterpret_tensor(arg51_1, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf70)
        del arg51_1
        del arg52_1
        del buf69
        buf71 = reinterpret_tensor(buf32, (25088, 64), (64, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [linear_151], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg46_1, reinterpret_tensor(buf63, (25088, 64), (64, 1), 0), reinterpret_tensor(arg45_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf71)
        del arg45_1
        del arg46_1
        del buf63
        # Topologically Sorted Source Nodes: [x_446], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf72 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf71, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf70, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf70, (8, 1, 49, 64), (6272, 0, 128, 1), 64), None, False)
        buf73 = buf72[0]
        del buf72
        buf77 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (25088, 64), (64, 1), 0), reinterpret_tensor(arg53_1, (64, 64), (1, 64), 0), out=buf77)
        del arg53_1
        buf78 = reinterpret_tensor(buf77, (8, 3136, 64), (200704, 64, 1), 0); del buf77  # reuse
        buf82 = reinterpret_tensor(buf73, (8, 3136, 64), (200704, 64, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_442, x_450, layer_norm_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf78, buf52, buf59, arg42_1, arg54_1, arg55_1, arg56_1, buf82, 25088, 64, grid=grid(25088), stream=stream0)
        del arg42_1
        del arg54_1
        del arg55_1
        del arg56_1
        del buf52
        del buf59
        buf83 = reinterpret_tensor(buf58, (25088, 512), (512, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (25088, 64), (64, 1), 0), reinterpret_tensor(arg57_1, (64, 512), (1, 64), 0), out=buf83)
        del arg57_1
        buf84 = reinterpret_tensor(buf83, (8, 3136, 512), (1605632, 512, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_452], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf84, arg58_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg58_1
        buf85 = reinterpret_tensor(buf82, (25088, 64), (64, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (25088, 512), (512, 1), 0), reinterpret_tensor(arg59_1, (512, 64), (1, 512), 0), out=buf85)
        del arg59_1
        del buf84
        buf86 = reinterpret_tensor(buf85, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_457], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf86, buf78, arg60_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg60_1
        del buf78
        buf87 = empty_strided_cuda((128, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_457, conv2d_38], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_12.run(arg61_1, buf87, 8192, 4, grid=grid(8192, 4), stream=stream0)
        del arg61_1
        # Topologically Sorted Source Nodes: [x_457, conv2d_38], Original ATen: [aten.clone, aten.convolution]
        buf88 = extern_kernels.convolution(buf86, buf87, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del buf86
        del buf87
        buf92 = empty_strided_cuda((8, 784, 128), (100352, 128, 1), torch.float32)
        buf96 = empty_strided_cuda((8, 784, 128), (100352, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_459, layer_norm_97], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_13.run(buf88, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, buf92, buf96, 6272, 128, grid=grid(6272), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        del arg66_1
        buf97 = reinterpret_tensor(buf64, (128, 128, 4, 4), (2048, 1, 512, 128), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [conv2d_39], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(arg69_1, buf97, 16384, 16, grid=grid(16384, 16), stream=stream0)
        del arg69_1
        # Topologically Sorted Source Nodes: [conv2d_39], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(reinterpret_tensor(buf96, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), buf97, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 128, 7, 7), (6272, 1, 896, 128))
        buf102 = reinterpret_tensor(buf70, (8, 49, 128), (6272, 128, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_463], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf98, arg70_1, arg71_1, arg72_1, buf102, 392, 128, grid=grid(392), stream=stream0)
        del arg70_1
        del arg71_1
        del arg72_1
        del buf98
        buf103 = empty_strided_cuda((392, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_157], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg74_1, reinterpret_tensor(buf102, (392, 128), (128, 1), 0), reinterpret_tensor(arg73_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf103)
        del arg73_1
        del arg74_1
        buf104 = reinterpret_tensor(buf88, (6272, 128), (128, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [linear_156], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg68_1, reinterpret_tensor(buf96, (6272, 128), (128, 1), 0), reinterpret_tensor(arg67_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf104)
        del arg67_1
        del arg68_1
        del buf96
        # Topologically Sorted Source Nodes: [x_464], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf105 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf104, (8, 2, 784, 64), (100352, 64, 128, 1), 0), reinterpret_tensor(buf103, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf103, (8, 2, 49, 64), (12544, 64, 256, 1), 128), None, False)
        buf106 = buf105[0]
        del buf105
        buf110 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (6272, 128), (128, 1), 0), reinterpret_tensor(arg75_1, (128, 128), (1, 128), 0), out=buf110)
        del arg75_1
        buf114 = reinterpret_tensor(buf106, (8, 784, 128), (100352, 128, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_468, layer_norm_99], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf92, buf110, arg76_1, arg77_1, arg78_1, buf114, 6272, 128, grid=grid(6272), stream=stream0)
        del arg77_1
        del arg78_1
        buf115 = empty_strided_cuda((6272, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (6272, 128), (128, 1), 0), reinterpret_tensor(arg79_1, (128, 1024), (1, 128), 0), out=buf115)
        del arg79_1
        buf116 = reinterpret_tensor(buf115, (8, 784, 1024), (802816, 1024, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_470], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf116, arg80_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg80_1
        buf117 = reinterpret_tensor(buf114, (6272, 128), (128, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg81_1, (1024, 128), (1, 1024), 0), out=buf117)
        del arg81_1
        buf118 = reinterpret_tensor(buf117, (8, 784, 128), (100352, 128, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [x_468, x_474], Original ATen: [aten.add]
        triton_poi_fused_add_18.run(buf118, buf92, buf110, arg76_1, arg82_1, 802816, grid=grid(802816), stream=stream0)
        del arg76_1
        del arg82_1
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(reinterpret_tensor(buf118, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), arg83_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf119, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del arg83_1
        buf123 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_100], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_19.run(buf119, arg84_1, buf118, arg85_1, arg86_1, buf123, 6272, 128, grid=grid(6272), stream=stream0)
        del arg85_1
        del arg86_1
        buf124 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [conv2d_41], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(arg89_1, buf124, 16384, 16, grid=grid(16384, 16), stream=stream0)
        del arg89_1
        # Topologically Sorted Source Nodes: [conv2d_41], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(reinterpret_tensor(buf123, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), buf124, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 128, 7, 7), (6272, 1, 896, 128))
        buf129 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_479], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf125, arg90_1, arg91_1, arg92_1, buf129, 392, 128, grid=grid(392), stream=stream0)
        del arg90_1
        del arg91_1
        del arg92_1
        del buf125
        buf130 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [linear_162], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg94_1, reinterpret_tensor(buf129, (392, 128), (128, 1), 0), reinterpret_tensor(arg93_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf130)
        del arg93_1
        del arg94_1
        buf131 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [linear_161], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg88_1, reinterpret_tensor(buf123, (6272, 128), (128, 1), 0), reinterpret_tensor(arg87_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf131)
        del arg87_1
        del arg88_1
        del buf123
        # Topologically Sorted Source Nodes: [x_480], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf132 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf131, (8, 2, 784, 64), (100352, 64, 128, 1), 0), reinterpret_tensor(buf130, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf130, (8, 2, 49, 64), (12544, 64, 256, 1), 128), None, False)
        buf133 = buf132[0]
        del buf132
        buf137 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf133, (6272, 128), (128, 1), 0), reinterpret_tensor(arg95_1, (128, 128), (1, 128), 0), out=buf137)
        del arg95_1
        buf138 = reinterpret_tensor(buf137, (8, 784, 128), (100352, 128, 1), 0); del buf137  # reuse
        buf142 = reinterpret_tensor(buf133, (8, 784, 128), (100352, 128, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [x_484, layer_norm_102], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf138, buf119, arg84_1, buf118, arg96_1, arg97_1, arg98_1, buf142, 6272, 128, grid=grid(6272), stream=stream0)
        del arg84_1
        del arg96_1
        del arg97_1
        del arg98_1
        buf143 = reinterpret_tensor(buf116, (6272, 1024), (1024, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (6272, 128), (128, 1), 0), reinterpret_tensor(arg99_1, (128, 1024), (1, 128), 0), out=buf143)
        del arg99_1
        buf144 = reinterpret_tensor(buf143, (8, 784, 1024), (802816, 1024, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_486], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf144, arg100_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg100_1
        buf145 = reinterpret_tensor(buf142, (6272, 128), (128, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf144, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg101_1, (1024, 128), (1, 1024), 0), out=buf145)
        del arg101_1
        buf149 = reinterpret_tensor(buf119, (8, 784, 128), (100352, 128, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [x_490, layer_norm_103], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf138, buf145, arg102_1, arg103_1, arg104_1, buf149, 6272, 128, grid=grid(6272), stream=stream0)
        del arg103_1
        del arg104_1
        buf150 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [conv2d_42], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(arg107_1, buf150, 16384, 16, grid=grid(16384, 16), stream=stream0)
        del arg107_1
        # Topologically Sorted Source Nodes: [conv2d_42], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(reinterpret_tensor(buf149, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), buf150, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 128, 7, 7), (6272, 1, 896, 128))
        buf155 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [x_493], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf151, arg108_1, arg109_1, arg110_1, buf155, 392, 128, grid=grid(392), stream=stream0)
        del arg108_1
        del arg109_1
        del arg110_1
        del buf151
        buf156 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [linear_167], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg112_1, reinterpret_tensor(buf155, (392, 128), (128, 1), 0), reinterpret_tensor(arg111_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf156)
        del arg111_1
        del arg112_1
        buf157 = reinterpret_tensor(buf118, (6272, 128), (128, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [linear_166], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg106_1, reinterpret_tensor(buf149, (6272, 128), (128, 1), 0), reinterpret_tensor(arg105_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf157)
        del arg105_1
        del arg106_1
        del buf149
        # Topologically Sorted Source Nodes: [x_494], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf158 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf157, (8, 2, 784, 64), (100352, 64, 128, 1), 0), reinterpret_tensor(buf156, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf156, (8, 2, 49, 64), (12544, 64, 256, 1), 128), None, False)
        buf159 = buf158[0]
        del buf158
        buf163 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (6272, 128), (128, 1), 0), reinterpret_tensor(arg113_1, (128, 128), (1, 128), 0), out=buf163)
        del arg113_1
        buf164 = reinterpret_tensor(buf163, (8, 784, 128), (100352, 128, 1), 0); del buf163  # reuse
        buf168 = reinterpret_tensor(buf159, (8, 784, 128), (100352, 128, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_490, x_498, layer_norm_105], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_21.run(buf164, buf138, buf145, arg102_1, arg114_1, arg115_1, arg116_1, buf168, 6272, 128, grid=grid(6272), stream=stream0)
        del arg102_1
        del arg114_1
        del arg115_1
        del arg116_1
        buf169 = reinterpret_tensor(buf144, (6272, 1024), (1024, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (6272, 128), (128, 1), 0), reinterpret_tensor(arg117_1, (128, 1024), (1, 128), 0), out=buf169)
        del arg117_1
        buf170 = reinterpret_tensor(buf169, (8, 784, 1024), (802816, 1024, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_500], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf170, arg118_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg118_1
        buf171 = reinterpret_tensor(buf168, (6272, 128), (128, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg119_1, (1024, 128), (1, 1024), 0), out=buf171)
        del arg119_1
        buf175 = reinterpret_tensor(buf145, (8, 784, 128), (100352, 128, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [x_504, layer_norm_106], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf164, buf171, arg120_1, arg121_1, arg122_1, buf175, 6272, 128, grid=grid(6272), stream=stream0)
        del arg121_1
        del arg122_1
        buf176 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [conv2d_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(arg125_1, buf176, 16384, 16, grid=grid(16384, 16), stream=stream0)
        del arg125_1
        # Topologically Sorted Source Nodes: [conv2d_43], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(reinterpret_tensor(buf175, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), buf176, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (8, 128, 7, 7), (6272, 1, 896, 128))
        del buf176
        buf181 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_507], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf177, arg126_1, arg127_1, arg128_1, buf181, 392, 128, grid=grid(392), stream=stream0)
        del arg126_1
        del arg127_1
        del arg128_1
        del buf177
        buf182 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [linear_172], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg130_1, reinterpret_tensor(buf181, (392, 128), (128, 1), 0), reinterpret_tensor(arg129_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf182)
        del arg129_1
        del arg130_1
        del buf181
        buf183 = reinterpret_tensor(buf138, (6272, 128), (128, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [linear_171], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg124_1, reinterpret_tensor(buf175, (6272, 128), (128, 1), 0), reinterpret_tensor(arg123_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf183)
        del arg123_1
        del arg124_1
        del buf175
        # Topologically Sorted Source Nodes: [x_508], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf184 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf183, (8, 2, 784, 64), (100352, 64, 128, 1), 0), reinterpret_tensor(buf182, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf182, (8, 2, 49, 64), (12544, 64, 256, 1), 128), None, False)
        del buf182
        buf185 = buf184[0]
        del buf184
        buf189 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf185, (6272, 128), (128, 1), 0), reinterpret_tensor(arg131_1, (128, 128), (1, 128), 0), out=buf189)
        del arg131_1
        buf190 = reinterpret_tensor(buf189, (8, 784, 128), (100352, 128, 1), 0); del buf189  # reuse
        buf194 = reinterpret_tensor(buf185, (8, 784, 128), (100352, 128, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [x_504, x_512, layer_norm_108], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_21.run(buf190, buf164, buf171, arg120_1, arg132_1, arg133_1, arg134_1, buf194, 6272, 128, grid=grid(6272), stream=stream0)
        del arg120_1
        del arg132_1
        del arg133_1
        del arg134_1
        del buf164
        del buf171
        buf195 = reinterpret_tensor(buf170, (6272, 1024), (1024, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (6272, 128), (128, 1), 0), reinterpret_tensor(arg135_1, (128, 1024), (1, 128), 0), out=buf195)
        del arg135_1
        buf196 = reinterpret_tensor(buf195, (8, 784, 1024), (802816, 1024, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_514], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf196, arg136_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg136_1
        buf197 = reinterpret_tensor(buf194, (6272, 128), (128, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 128), (1, 1024), 0), out=buf197)
        del arg137_1
        del buf196
        buf198 = reinterpret_tensor(buf197, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [x_519], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf198, buf190, arg138_1, 802816, grid=grid(802816), stream=stream0)
        del arg138_1
        del buf190
        buf199 = empty_strided_cuda((320, 128, 2, 2), (512, 1, 256, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_519, conv2d_44], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_23.run(arg139_1, buf199, 40960, 4, grid=grid(40960, 4), stream=stream0)
        del arg139_1
        # Topologically Sorted Source Nodes: [x_519, conv2d_44], Original ATen: [aten.clone, aten.convolution]
        buf200 = extern_kernels.convolution(buf198, buf199, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 320, 14, 14), (62720, 1, 4480, 320))
        del buf199
        buf204 = empty_strided_cuda((8, 196, 320), (62720, 320, 1), torch.float32)
        buf208 = empty_strided_cuda((8, 196, 320), (62720, 320, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_521, layer_norm_110], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_24.run(buf200, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, buf204, buf208, 1568, 320, grid=grid(1568), stream=stream0)
        del arg140_1
        del arg141_1
        del arg142_1
        del arg143_1
        del arg144_1
        buf209 = empty_strided_cuda((320, 320, 2, 2), (1280, 1, 640, 320), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_45], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg147_1, buf209, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg147_1
        # Topologically Sorted Source Nodes: [conv2d_45], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(reinterpret_tensor(buf208, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf209, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf214 = empty_strided_cuda((8, 49, 320), (15680, 320, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_525], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf210, arg148_1, arg149_1, arg150_1, buf214, 392, 320, grid=grid(392), stream=stream0)
        del arg148_1
        del arg149_1
        del arg150_1
        del buf210
        buf215 = empty_strided_cuda((392, 640), (640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_177], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg152_1, reinterpret_tensor(buf214, (392, 320), (320, 1), 0), reinterpret_tensor(arg151_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf215)
        del arg151_1
        del arg152_1
        buf216 = reinterpret_tensor(buf200, (1568, 320), (320, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [linear_176], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg146_1, reinterpret_tensor(buf208, (1568, 320), (320, 1), 0), reinterpret_tensor(arg145_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf216)
        del arg145_1
        del arg146_1
        del buf208
        # Topologically Sorted Source Nodes: [x_526], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf217 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf216, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf215, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf215, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf218 = buf217[0]
        del buf217
        buf222 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (1568, 320), (320, 1), 0), reinterpret_tensor(arg153_1, (320, 320), (1, 320), 0), out=buf222)
        del arg153_1
        buf226 = reinterpret_tensor(buf218, (8, 196, 320), (62720, 320, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [x_530, layer_norm_112], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf204, buf222, arg154_1, arg155_1, arg156_1, buf226, 1568, 320, grid=grid(1568), stream=stream0)
        del arg155_1
        del arg156_1
        buf227 = empty_strided_cuda((1568, 1280), (1280, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (1568, 320), (320, 1), 0), reinterpret_tensor(arg157_1, (320, 1280), (1, 320), 0), out=buf227)
        del arg157_1
        buf228 = reinterpret_tensor(buf227, (8, 196, 1280), (250880, 1280, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [x_532], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf228, arg158_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg158_1
        buf229 = reinterpret_tensor(buf226, (1568, 320), (320, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg159_1, (1280, 320), (1, 1280), 0), out=buf229)
        del arg159_1
        buf230 = reinterpret_tensor(buf229, (8, 196, 320), (62720, 320, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [x_530, x_536], Original ATen: [aten.add]
        triton_poi_fused_add_29.run(buf230, buf204, buf222, arg154_1, arg160_1, 501760, grid=grid(501760), stream=stream0)
        del arg154_1
        del arg160_1
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(reinterpret_tensor(buf230, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), arg161_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320, bias=None)
        assert_size_stride(buf231, (8, 320, 14, 14), (62720, 1, 4480, 320))
        del arg161_1
        buf235 = reinterpret_tensor(buf222, (8, 196, 320), (62720, 320, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_113], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_30.run(buf231, arg162_1, buf230, arg163_1, arg164_1, buf235, 1568, 320, grid=grid(1568), stream=stream0)
        del arg163_1
        del arg164_1
        buf236 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [conv2d_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg167_1, buf236, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg167_1
        # Topologically Sorted Source Nodes: [conv2d_47], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(reinterpret_tensor(buf235, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf236, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf241 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [x_541], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf237, arg168_1, arg169_1, arg170_1, buf241, 392, 320, grid=grid(392), stream=stream0)
        del arg168_1
        del arg169_1
        del arg170_1
        del buf237
        buf242 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [linear_182], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg172_1, reinterpret_tensor(buf241, (392, 320), (320, 1), 0), reinterpret_tensor(arg171_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf242)
        del arg171_1
        del arg172_1
        buf243 = reinterpret_tensor(buf204, (1568, 320), (320, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [linear_181], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg166_1, reinterpret_tensor(buf235, (1568, 320), (320, 1), 0), reinterpret_tensor(arg165_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf243)
        del arg165_1
        del arg166_1
        del buf235
        # Topologically Sorted Source Nodes: [x_542], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf244 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf243, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf242, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf242, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf245 = buf244[0]
        del buf244
        buf249 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (1568, 320), (320, 1), 0), reinterpret_tensor(arg173_1, (320, 320), (1, 320), 0), out=buf249)
        del arg173_1
        buf250 = reinterpret_tensor(buf249, (8, 196, 320), (62720, 320, 1), 0); del buf249  # reuse
        buf254 = reinterpret_tensor(buf245, (8, 196, 320), (62720, 320, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [x_546, layer_norm_115], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_31.run(buf250, buf231, arg162_1, buf230, arg174_1, arg175_1, arg176_1, buf254, 1568, 320, grid=grid(1568), stream=stream0)
        del arg162_1
        del arg174_1
        del arg175_1
        del arg176_1
        buf255 = reinterpret_tensor(buf228, (1568, 1280), (1280, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf254, (1568, 320), (320, 1), 0), reinterpret_tensor(arg177_1, (320, 1280), (1, 320), 0), out=buf255)
        del arg177_1
        buf256 = reinterpret_tensor(buf255, (8, 196, 1280), (250880, 1280, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [x_548], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf256, arg178_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg178_1
        buf257 = reinterpret_tensor(buf254, (1568, 320), (320, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg179_1, (1280, 320), (1, 1280), 0), out=buf257)
        del arg179_1
        buf261 = reinterpret_tensor(buf231, (8, 196, 320), (62720, 320, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [x_552, layer_norm_116], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf250, buf257, arg180_1, arg181_1, arg182_1, buf261, 1568, 320, grid=grid(1568), stream=stream0)
        del arg181_1
        del arg182_1
        buf262 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [conv2d_48], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg185_1, buf262, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg185_1
        # Topologically Sorted Source Nodes: [conv2d_48], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(reinterpret_tensor(buf261, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf262, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf267 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [x_555], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf263, arg186_1, arg187_1, arg188_1, buf267, 392, 320, grid=grid(392), stream=stream0)
        del arg186_1
        del arg187_1
        del arg188_1
        del buf263
        buf268 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [linear_187], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg190_1, reinterpret_tensor(buf267, (392, 320), (320, 1), 0), reinterpret_tensor(arg189_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf268)
        del arg189_1
        del arg190_1
        buf269 = reinterpret_tensor(buf230, (1568, 320), (320, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [linear_186], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg184_1, reinterpret_tensor(buf261, (1568, 320), (320, 1), 0), reinterpret_tensor(arg183_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf269)
        del arg183_1
        del arg184_1
        del buf261
        # Topologically Sorted Source Nodes: [x_556], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf270 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf269, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf268, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf268, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf271 = buf270[0]
        del buf270
        buf275 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (1568, 320), (320, 1), 0), reinterpret_tensor(arg191_1, (320, 320), (1, 320), 0), out=buf275)
        del arg191_1
        buf276 = reinterpret_tensor(buf275, (8, 196, 320), (62720, 320, 1), 0); del buf275  # reuse
        buf280 = reinterpret_tensor(buf271, (8, 196, 320), (62720, 320, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [x_552, x_560, layer_norm_118], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf276, buf250, buf257, arg180_1, arg192_1, arg193_1, arg194_1, buf280, 1568, 320, grid=grid(1568), stream=stream0)
        del arg180_1
        del arg192_1
        del arg193_1
        del arg194_1
        buf281 = reinterpret_tensor(buf256, (1568, 1280), (1280, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf280, (1568, 320), (320, 1), 0), reinterpret_tensor(arg195_1, (320, 1280), (1, 320), 0), out=buf281)
        del arg195_1
        buf282 = reinterpret_tensor(buf281, (8, 196, 1280), (250880, 1280, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [x_562], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf282, arg196_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg196_1
        buf283 = reinterpret_tensor(buf280, (1568, 320), (320, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg197_1, (1280, 320), (1, 1280), 0), out=buf283)
        del arg197_1
        buf287 = reinterpret_tensor(buf257, (8, 196, 320), (62720, 320, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [x_566, layer_norm_119], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf276, buf283, arg198_1, arg199_1, arg200_1, buf287, 1568, 320, grid=grid(1568), stream=stream0)
        del arg199_1
        del arg200_1
        buf288 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [conv2d_49], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg203_1, buf288, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg203_1
        # Topologically Sorted Source Nodes: [conv2d_49], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(reinterpret_tensor(buf287, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf288, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf293 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [x_569], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf289, arg204_1, arg205_1, arg206_1, buf293, 392, 320, grid=grid(392), stream=stream0)
        del arg204_1
        del arg205_1
        del arg206_1
        del buf289
        buf294 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [linear_192], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg208_1, reinterpret_tensor(buf293, (392, 320), (320, 1), 0), reinterpret_tensor(arg207_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf294)
        del arg207_1
        del arg208_1
        buf295 = reinterpret_tensor(buf250, (1568, 320), (320, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [linear_191], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg202_1, reinterpret_tensor(buf287, (1568, 320), (320, 1), 0), reinterpret_tensor(arg201_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf295)
        del arg201_1
        del arg202_1
        del buf287
        # Topologically Sorted Source Nodes: [x_570], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf296 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf295, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf294, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf294, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf297 = buf296[0]
        del buf296
        buf301 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf297, (1568, 320), (320, 1), 0), reinterpret_tensor(arg209_1, (320, 320), (1, 320), 0), out=buf301)
        del arg209_1
        buf302 = reinterpret_tensor(buf301, (8, 196, 320), (62720, 320, 1), 0); del buf301  # reuse
        buf306 = reinterpret_tensor(buf297, (8, 196, 320), (62720, 320, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [x_566, x_574, layer_norm_121], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf302, buf276, buf283, arg198_1, arg210_1, arg211_1, arg212_1, buf306, 1568, 320, grid=grid(1568), stream=stream0)
        del arg198_1
        del arg210_1
        del arg211_1
        del arg212_1
        buf307 = reinterpret_tensor(buf282, (1568, 1280), (1280, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf306, (1568, 320), (320, 1), 0), reinterpret_tensor(arg213_1, (320, 1280), (1, 320), 0), out=buf307)
        del arg213_1
        buf308 = reinterpret_tensor(buf307, (8, 196, 1280), (250880, 1280, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [x_576], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf308, arg214_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg214_1
        buf309 = reinterpret_tensor(buf306, (1568, 320), (320, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf308, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg215_1, (1280, 320), (1, 1280), 0), out=buf309)
        del arg215_1
        buf313 = reinterpret_tensor(buf283, (8, 196, 320), (62720, 320, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [x_580, layer_norm_122], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf302, buf309, arg216_1, arg217_1, arg218_1, buf313, 1568, 320, grid=grid(1568), stream=stream0)
        del arg217_1
        del arg218_1
        buf314 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [conv2d_50], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg221_1, buf314, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg221_1
        # Topologically Sorted Source Nodes: [conv2d_50], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(reinterpret_tensor(buf313, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf314, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf319 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [x_583], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf315, arg222_1, arg223_1, arg224_1, buf319, 392, 320, grid=grid(392), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del buf315
        buf320 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [linear_197], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg226_1, reinterpret_tensor(buf319, (392, 320), (320, 1), 0), reinterpret_tensor(arg225_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf320)
        del arg225_1
        del arg226_1
        buf321 = reinterpret_tensor(buf276, (1568, 320), (320, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [linear_196], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg220_1, reinterpret_tensor(buf313, (1568, 320), (320, 1), 0), reinterpret_tensor(arg219_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf321)
        del arg219_1
        del arg220_1
        del buf313
        # Topologically Sorted Source Nodes: [x_584], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf322 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf321, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf320, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf320, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf323 = buf322[0]
        del buf322
        buf327 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (1568, 320), (320, 1), 0), reinterpret_tensor(arg227_1, (320, 320), (1, 320), 0), out=buf327)
        del arg227_1
        buf328 = reinterpret_tensor(buf327, (8, 196, 320), (62720, 320, 1), 0); del buf327  # reuse
        buf332 = reinterpret_tensor(buf323, (8, 196, 320), (62720, 320, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [x_580, x_588, layer_norm_124], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf328, buf302, buf309, arg216_1, arg228_1, arg229_1, arg230_1, buf332, 1568, 320, grid=grid(1568), stream=stream0)
        del arg216_1
        del arg228_1
        del arg229_1
        del arg230_1
        buf333 = reinterpret_tensor(buf308, (1568, 1280), (1280, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (1568, 320), (320, 1), 0), reinterpret_tensor(arg231_1, (320, 1280), (1, 320), 0), out=buf333)
        del arg231_1
        buf334 = reinterpret_tensor(buf333, (8, 196, 1280), (250880, 1280, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [x_590], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf334, arg232_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg232_1
        buf335 = reinterpret_tensor(buf332, (1568, 320), (320, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg233_1, (1280, 320), (1, 1280), 0), out=buf335)
        del arg233_1
        buf339 = reinterpret_tensor(buf309, (8, 196, 320), (62720, 320, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [x_594, layer_norm_125], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf328, buf335, arg234_1, arg235_1, arg236_1, buf339, 1568, 320, grid=grid(1568), stream=stream0)
        del arg235_1
        del arg236_1
        buf340 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [conv2d_51], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg239_1, buf340, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg239_1
        # Topologically Sorted Source Nodes: [conv2d_51], Original ATen: [aten.convolution]
        buf341 = extern_kernels.convolution(reinterpret_tensor(buf339, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf340, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf341, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf345 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [x_597], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf341, arg240_1, arg241_1, arg242_1, buf345, 392, 320, grid=grid(392), stream=stream0)
        del arg240_1
        del arg241_1
        del arg242_1
        del buf341
        buf346 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [linear_202], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg244_1, reinterpret_tensor(buf345, (392, 320), (320, 1), 0), reinterpret_tensor(arg243_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf346)
        del arg243_1
        del arg244_1
        buf347 = reinterpret_tensor(buf302, (1568, 320), (320, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [linear_201], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg238_1, reinterpret_tensor(buf339, (1568, 320), (320, 1), 0), reinterpret_tensor(arg237_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf347)
        del arg237_1
        del arg238_1
        del buf339
        # Topologically Sorted Source Nodes: [x_598], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf348 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf347, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf346, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf346, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf349 = buf348[0]
        del buf348
        buf353 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf349, (1568, 320), (320, 1), 0), reinterpret_tensor(arg245_1, (320, 320), (1, 320), 0), out=buf353)
        del arg245_1
        buf354 = reinterpret_tensor(buf353, (8, 196, 320), (62720, 320, 1), 0); del buf353  # reuse
        buf358 = reinterpret_tensor(buf349, (8, 196, 320), (62720, 320, 1), 0); del buf349  # reuse
        # Topologically Sorted Source Nodes: [x_594, x_602, layer_norm_127], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf354, buf328, buf335, arg234_1, arg246_1, arg247_1, arg248_1, buf358, 1568, 320, grid=grid(1568), stream=stream0)
        del arg234_1
        del arg246_1
        del arg247_1
        del arg248_1
        buf359 = reinterpret_tensor(buf334, (1568, 1280), (1280, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (1568, 320), (320, 1), 0), reinterpret_tensor(arg249_1, (320, 1280), (1, 320), 0), out=buf359)
        del arg249_1
        buf360 = reinterpret_tensor(buf359, (8, 196, 1280), (250880, 1280, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [x_604], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf360, arg250_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg250_1
        buf361 = reinterpret_tensor(buf358, (1568, 320), (320, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf360, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg251_1, (1280, 320), (1, 1280), 0), out=buf361)
        del arg251_1
        buf365 = reinterpret_tensor(buf335, (8, 196, 320), (62720, 320, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [x_608, layer_norm_128], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf354, buf361, arg252_1, arg253_1, arg254_1, buf365, 1568, 320, grid=grid(1568), stream=stream0)
        del arg253_1
        del arg254_1
        buf366 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [conv2d_52], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg257_1, buf366, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg257_1
        # Topologically Sorted Source Nodes: [conv2d_52], Original ATen: [aten.convolution]
        buf367 = extern_kernels.convolution(reinterpret_tensor(buf365, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf366, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf371 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [x_611], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf367, arg258_1, arg259_1, arg260_1, buf371, 392, 320, grid=grid(392), stream=stream0)
        del arg258_1
        del arg259_1
        del arg260_1
        del buf367
        buf372 = buf346; del buf346  # reuse
        # Topologically Sorted Source Nodes: [linear_207], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg262_1, reinterpret_tensor(buf371, (392, 320), (320, 1), 0), reinterpret_tensor(arg261_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf372)
        del arg261_1
        del arg262_1
        buf373 = reinterpret_tensor(buf328, (1568, 320), (320, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [linear_206], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg256_1, reinterpret_tensor(buf365, (1568, 320), (320, 1), 0), reinterpret_tensor(arg255_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf373)
        del arg255_1
        del arg256_1
        del buf365
        # Topologically Sorted Source Nodes: [x_612], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf374 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf373, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf372, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf372, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf375 = buf374[0]
        del buf374
        buf379 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf375, (1568, 320), (320, 1), 0), reinterpret_tensor(arg263_1, (320, 320), (1, 320), 0), out=buf379)
        del arg263_1
        buf380 = reinterpret_tensor(buf379, (8, 196, 320), (62720, 320, 1), 0); del buf379  # reuse
        buf384 = reinterpret_tensor(buf375, (8, 196, 320), (62720, 320, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [x_608, x_616, layer_norm_130], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf380, buf354, buf361, arg252_1, arg264_1, arg265_1, arg266_1, buf384, 1568, 320, grid=grid(1568), stream=stream0)
        del arg252_1
        del arg264_1
        del arg265_1
        del arg266_1
        buf385 = reinterpret_tensor(buf360, (1568, 1280), (1280, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf384, (1568, 320), (320, 1), 0), reinterpret_tensor(arg267_1, (320, 1280), (1, 320), 0), out=buf385)
        del arg267_1
        buf386 = reinterpret_tensor(buf385, (8, 196, 1280), (250880, 1280, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [x_618], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf386, arg268_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg268_1
        buf387 = reinterpret_tensor(buf384, (1568, 320), (320, 1), 0); del buf384  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf386, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg269_1, (1280, 320), (1, 1280), 0), out=buf387)
        del arg269_1
        buf391 = reinterpret_tensor(buf361, (8, 196, 320), (62720, 320, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [x_622, layer_norm_131], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf380, buf387, arg270_1, arg271_1, arg272_1, buf391, 1568, 320, grid=grid(1568), stream=stream0)
        del arg271_1
        del arg272_1
        buf392 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [conv2d_53], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg275_1, buf392, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg275_1
        # Topologically Sorted Source Nodes: [conv2d_53], Original ATen: [aten.convolution]
        buf393 = extern_kernels.convolution(reinterpret_tensor(buf391, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf392, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf393, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf397 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [x_625], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf393, arg276_1, arg277_1, arg278_1, buf397, 392, 320, grid=grid(392), stream=stream0)
        del arg276_1
        del arg277_1
        del arg278_1
        del buf393
        buf398 = buf372; del buf372  # reuse
        # Topologically Sorted Source Nodes: [linear_212], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg280_1, reinterpret_tensor(buf397, (392, 320), (320, 1), 0), reinterpret_tensor(arg279_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf398)
        del arg279_1
        del arg280_1
        buf399 = reinterpret_tensor(buf354, (1568, 320), (320, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [linear_211], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg274_1, reinterpret_tensor(buf391, (1568, 320), (320, 1), 0), reinterpret_tensor(arg273_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf399)
        del arg273_1
        del arg274_1
        del buf391
        # Topologically Sorted Source Nodes: [x_626], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf400 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf399, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf398, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf398, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf401 = buf400[0]
        del buf400
        buf405 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf401, (1568, 320), (320, 1), 0), reinterpret_tensor(arg281_1, (320, 320), (1, 320), 0), out=buf405)
        del arg281_1
        buf406 = reinterpret_tensor(buf405, (8, 196, 320), (62720, 320, 1), 0); del buf405  # reuse
        buf410 = reinterpret_tensor(buf401, (8, 196, 320), (62720, 320, 1), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [x_622, x_630, layer_norm_133], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf406, buf380, buf387, arg270_1, arg282_1, arg283_1, arg284_1, buf410, 1568, 320, grid=grid(1568), stream=stream0)
        del arg270_1
        del arg282_1
        del arg283_1
        del arg284_1
        buf411 = reinterpret_tensor(buf386, (1568, 1280), (1280, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf410, (1568, 320), (320, 1), 0), reinterpret_tensor(arg285_1, (320, 1280), (1, 320), 0), out=buf411)
        del arg285_1
        buf412 = reinterpret_tensor(buf411, (8, 196, 1280), (250880, 1280, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [x_632], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf412, arg286_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg286_1
        buf413 = reinterpret_tensor(buf410, (1568, 320), (320, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg287_1, (1280, 320), (1, 1280), 0), out=buf413)
        del arg287_1
        buf417 = reinterpret_tensor(buf387, (8, 196, 320), (62720, 320, 1), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [x_636, layer_norm_134], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf406, buf413, arg288_1, arg289_1, arg290_1, buf417, 1568, 320, grid=grid(1568), stream=stream0)
        del arg289_1
        del arg290_1
        buf418 = buf392; del buf392  # reuse
        # Topologically Sorted Source Nodes: [conv2d_54], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg293_1, buf418, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg293_1
        # Topologically Sorted Source Nodes: [conv2d_54], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(reinterpret_tensor(buf417, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf418, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf423 = buf397; del buf397  # reuse
        # Topologically Sorted Source Nodes: [x_639], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf419, arg294_1, arg295_1, arg296_1, buf423, 392, 320, grid=grid(392), stream=stream0)
        del arg294_1
        del arg295_1
        del arg296_1
        del buf419
        buf424 = buf398; del buf398  # reuse
        # Topologically Sorted Source Nodes: [linear_217], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg298_1, reinterpret_tensor(buf423, (392, 320), (320, 1), 0), reinterpret_tensor(arg297_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf424)
        del arg297_1
        del arg298_1
        buf425 = reinterpret_tensor(buf380, (1568, 320), (320, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [linear_216], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg292_1, reinterpret_tensor(buf417, (1568, 320), (320, 1), 0), reinterpret_tensor(arg291_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf425)
        del arg291_1
        del arg292_1
        del buf417
        # Topologically Sorted Source Nodes: [x_640], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf426 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf425, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf424, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf424, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf427 = buf426[0]
        del buf426
        buf431 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf427, (1568, 320), (320, 1), 0), reinterpret_tensor(arg299_1, (320, 320), (1, 320), 0), out=buf431)
        del arg299_1
        buf432 = reinterpret_tensor(buf431, (8, 196, 320), (62720, 320, 1), 0); del buf431  # reuse
        buf436 = reinterpret_tensor(buf427, (8, 196, 320), (62720, 320, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [x_636, x_644, layer_norm_136], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf432, buf406, buf413, arg288_1, arg300_1, arg301_1, arg302_1, buf436, 1568, 320, grid=grid(1568), stream=stream0)
        del arg288_1
        del arg300_1
        del arg301_1
        del arg302_1
        buf437 = reinterpret_tensor(buf412, (1568, 1280), (1280, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf436, (1568, 320), (320, 1), 0), reinterpret_tensor(arg303_1, (320, 1280), (1, 320), 0), out=buf437)
        del arg303_1
        buf438 = reinterpret_tensor(buf437, (8, 196, 1280), (250880, 1280, 1), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [x_646], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf438, arg304_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg304_1
        buf439 = reinterpret_tensor(buf436, (1568, 320), (320, 1), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf438, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg305_1, (1280, 320), (1, 1280), 0), out=buf439)
        del arg305_1
        buf443 = reinterpret_tensor(buf413, (8, 196, 320), (62720, 320, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [x_650, layer_norm_137], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf432, buf439, arg306_1, arg307_1, arg308_1, buf443, 1568, 320, grid=grid(1568), stream=stream0)
        del arg307_1
        del arg308_1
        buf444 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [conv2d_55], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg311_1, buf444, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg311_1
        # Topologically Sorted Source Nodes: [conv2d_55], Original ATen: [aten.convolution]
        buf445 = extern_kernels.convolution(reinterpret_tensor(buf443, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf444, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf449 = buf423; del buf423  # reuse
        # Topologically Sorted Source Nodes: [x_653], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf445, arg312_1, arg313_1, arg314_1, buf449, 392, 320, grid=grid(392), stream=stream0)
        del arg312_1
        del arg313_1
        del arg314_1
        del buf445
        buf450 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [linear_222], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg316_1, reinterpret_tensor(buf449, (392, 320), (320, 1), 0), reinterpret_tensor(arg315_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf450)
        del arg315_1
        del arg316_1
        buf451 = reinterpret_tensor(buf406, (1568, 320), (320, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [linear_221], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg310_1, reinterpret_tensor(buf443, (1568, 320), (320, 1), 0), reinterpret_tensor(arg309_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf451)
        del arg309_1
        del arg310_1
        del buf443
        # Topologically Sorted Source Nodes: [x_654], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf452 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf451, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf450, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf450, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf453 = buf452[0]
        del buf452
        buf457 = buf451; del buf451  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf453, (1568, 320), (320, 1), 0), reinterpret_tensor(arg317_1, (320, 320), (1, 320), 0), out=buf457)
        del arg317_1
        buf458 = reinterpret_tensor(buf457, (8, 196, 320), (62720, 320, 1), 0); del buf457  # reuse
        buf462 = reinterpret_tensor(buf453, (8, 196, 320), (62720, 320, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [x_650, x_658, layer_norm_139], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf458, buf432, buf439, arg306_1, arg318_1, arg319_1, arg320_1, buf462, 1568, 320, grid=grid(1568), stream=stream0)
        del arg306_1
        del arg318_1
        del arg319_1
        del arg320_1
        buf463 = reinterpret_tensor(buf438, (1568, 1280), (1280, 1), 0); del buf438  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf462, (1568, 320), (320, 1), 0), reinterpret_tensor(arg321_1, (320, 1280), (1, 320), 0), out=buf463)
        del arg321_1
        buf464 = reinterpret_tensor(buf463, (8, 196, 1280), (250880, 1280, 1), 0); del buf463  # reuse
        # Topologically Sorted Source Nodes: [x_660], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf464, arg322_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg322_1
        buf465 = reinterpret_tensor(buf462, (1568, 320), (320, 1), 0); del buf462  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf464, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg323_1, (1280, 320), (1, 1280), 0), out=buf465)
        del arg323_1
        buf469 = reinterpret_tensor(buf439, (8, 196, 320), (62720, 320, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [x_664, layer_norm_140], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf458, buf465, arg324_1, arg325_1, arg326_1, buf469, 1568, 320, grid=grid(1568), stream=stream0)
        del arg325_1
        del arg326_1
        buf470 = buf444; del buf444  # reuse
        # Topologically Sorted Source Nodes: [conv2d_56], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg329_1, buf470, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg329_1
        # Topologically Sorted Source Nodes: [conv2d_56], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(reinterpret_tensor(buf469, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf470, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf475 = buf449; del buf449  # reuse
        # Topologically Sorted Source Nodes: [x_667], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf471, arg330_1, arg331_1, arg332_1, buf475, 392, 320, grid=grid(392), stream=stream0)
        del arg330_1
        del arg331_1
        del arg332_1
        del buf471
        buf476 = buf450; del buf450  # reuse
        # Topologically Sorted Source Nodes: [linear_227], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg334_1, reinterpret_tensor(buf475, (392, 320), (320, 1), 0), reinterpret_tensor(arg333_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf476)
        del arg333_1
        del arg334_1
        buf477 = reinterpret_tensor(buf432, (1568, 320), (320, 1), 0); del buf432  # reuse
        # Topologically Sorted Source Nodes: [linear_226], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg328_1, reinterpret_tensor(buf469, (1568, 320), (320, 1), 0), reinterpret_tensor(arg327_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf477)
        del arg327_1
        del arg328_1
        del buf469
        # Topologically Sorted Source Nodes: [x_668], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf478 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf477, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf476, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf476, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf479 = buf478[0]
        del buf478
        buf483 = buf477; del buf477  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf479, (1568, 320), (320, 1), 0), reinterpret_tensor(arg335_1, (320, 320), (1, 320), 0), out=buf483)
        del arg335_1
        buf484 = reinterpret_tensor(buf483, (8, 196, 320), (62720, 320, 1), 0); del buf483  # reuse
        buf488 = reinterpret_tensor(buf479, (8, 196, 320), (62720, 320, 1), 0); del buf479  # reuse
        # Topologically Sorted Source Nodes: [x_664, x_672, layer_norm_142], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf484, buf458, buf465, arg324_1, arg336_1, arg337_1, arg338_1, buf488, 1568, 320, grid=grid(1568), stream=stream0)
        del arg324_1
        del arg336_1
        del arg337_1
        del arg338_1
        buf489 = reinterpret_tensor(buf464, (1568, 1280), (1280, 1), 0); del buf464  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf488, (1568, 320), (320, 1), 0), reinterpret_tensor(arg339_1, (320, 1280), (1, 320), 0), out=buf489)
        del arg339_1
        buf490 = reinterpret_tensor(buf489, (8, 196, 1280), (250880, 1280, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [x_674], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf490, arg340_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg340_1
        buf491 = reinterpret_tensor(buf488, (1568, 320), (320, 1), 0); del buf488  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf490, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg341_1, (1280, 320), (1, 1280), 0), out=buf491)
        del arg341_1
        buf495 = reinterpret_tensor(buf465, (8, 196, 320), (62720, 320, 1), 0); del buf465  # reuse
        # Topologically Sorted Source Nodes: [x_678, layer_norm_143], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf484, buf491, arg342_1, arg343_1, arg344_1, buf495, 1568, 320, grid=grid(1568), stream=stream0)
        del arg343_1
        del arg344_1
        buf496 = buf470; del buf470  # reuse
        # Topologically Sorted Source Nodes: [conv2d_57], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg347_1, buf496, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg347_1
        # Topologically Sorted Source Nodes: [conv2d_57], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(reinterpret_tensor(buf495, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf496, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf501 = buf475; del buf475  # reuse
        # Topologically Sorted Source Nodes: [x_681], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf497, arg348_1, arg349_1, arg350_1, buf501, 392, 320, grid=grid(392), stream=stream0)
        del arg348_1
        del arg349_1
        del arg350_1
        del buf497
        buf502 = buf476; del buf476  # reuse
        # Topologically Sorted Source Nodes: [linear_232], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg352_1, reinterpret_tensor(buf501, (392, 320), (320, 1), 0), reinterpret_tensor(arg351_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf502)
        del arg351_1
        del arg352_1
        buf503 = reinterpret_tensor(buf458, (1568, 320), (320, 1), 0); del buf458  # reuse
        # Topologically Sorted Source Nodes: [linear_231], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg346_1, reinterpret_tensor(buf495, (1568, 320), (320, 1), 0), reinterpret_tensor(arg345_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf503)
        del arg345_1
        del arg346_1
        del buf495
        # Topologically Sorted Source Nodes: [x_682], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf504 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf503, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf502, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf502, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf505 = buf504[0]
        del buf504
        buf509 = buf503; del buf503  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf505, (1568, 320), (320, 1), 0), reinterpret_tensor(arg353_1, (320, 320), (1, 320), 0), out=buf509)
        del arg353_1
        buf510 = reinterpret_tensor(buf509, (8, 196, 320), (62720, 320, 1), 0); del buf509  # reuse
        buf514 = reinterpret_tensor(buf505, (8, 196, 320), (62720, 320, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [x_678, x_686, layer_norm_145], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf510, buf484, buf491, arg342_1, arg354_1, arg355_1, arg356_1, buf514, 1568, 320, grid=grid(1568), stream=stream0)
        del arg342_1
        del arg354_1
        del arg355_1
        del arg356_1
        buf515 = reinterpret_tensor(buf490, (1568, 1280), (1280, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf514, (1568, 320), (320, 1), 0), reinterpret_tensor(arg357_1, (320, 1280), (1, 320), 0), out=buf515)
        del arg357_1
        buf516 = reinterpret_tensor(buf515, (8, 196, 1280), (250880, 1280, 1), 0); del buf515  # reuse
        # Topologically Sorted Source Nodes: [x_688], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf516, arg358_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg358_1
        buf517 = reinterpret_tensor(buf514, (1568, 320), (320, 1), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf516, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg359_1, (1280, 320), (1, 1280), 0), out=buf517)
        del arg359_1
        buf521 = reinterpret_tensor(buf491, (8, 196, 320), (62720, 320, 1), 0); del buf491  # reuse
        # Topologically Sorted Source Nodes: [x_692, layer_norm_146], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf510, buf517, arg360_1, arg361_1, arg362_1, buf521, 1568, 320, grid=grid(1568), stream=stream0)
        del arg361_1
        del arg362_1
        buf522 = buf496; del buf496  # reuse
        # Topologically Sorted Source Nodes: [conv2d_58], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg365_1, buf522, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg365_1
        # Topologically Sorted Source Nodes: [conv2d_58], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(reinterpret_tensor(buf521, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf522, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf527 = buf501; del buf501  # reuse
        # Topologically Sorted Source Nodes: [x_695], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf523, arg366_1, arg367_1, arg368_1, buf527, 392, 320, grid=grid(392), stream=stream0)
        del arg366_1
        del arg367_1
        del arg368_1
        del buf523
        buf528 = buf502; del buf502  # reuse
        # Topologically Sorted Source Nodes: [linear_237], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg370_1, reinterpret_tensor(buf527, (392, 320), (320, 1), 0), reinterpret_tensor(arg369_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf528)
        del arg369_1
        del arg370_1
        buf529 = reinterpret_tensor(buf484, (1568, 320), (320, 1), 0); del buf484  # reuse
        # Topologically Sorted Source Nodes: [linear_236], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg364_1, reinterpret_tensor(buf521, (1568, 320), (320, 1), 0), reinterpret_tensor(arg363_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf529)
        del arg363_1
        del arg364_1
        del buf521
        # Topologically Sorted Source Nodes: [x_696], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf530 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf529, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf528, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf528, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf531 = buf530[0]
        del buf530
        buf535 = buf529; del buf529  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf531, (1568, 320), (320, 1), 0), reinterpret_tensor(arg371_1, (320, 320), (1, 320), 0), out=buf535)
        del arg371_1
        buf536 = reinterpret_tensor(buf535, (8, 196, 320), (62720, 320, 1), 0); del buf535  # reuse
        buf540 = reinterpret_tensor(buf531, (8, 196, 320), (62720, 320, 1), 0); del buf531  # reuse
        # Topologically Sorted Source Nodes: [x_692, x_700, layer_norm_148], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf536, buf510, buf517, arg360_1, arg372_1, arg373_1, arg374_1, buf540, 1568, 320, grid=grid(1568), stream=stream0)
        del arg360_1
        del arg372_1
        del arg373_1
        del arg374_1
        buf541 = reinterpret_tensor(buf516, (1568, 1280), (1280, 1), 0); del buf516  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf540, (1568, 320), (320, 1), 0), reinterpret_tensor(arg375_1, (320, 1280), (1, 320), 0), out=buf541)
        del arg375_1
        buf542 = reinterpret_tensor(buf541, (8, 196, 1280), (250880, 1280, 1), 0); del buf541  # reuse
        # Topologically Sorted Source Nodes: [x_702], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf542, arg376_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg376_1
        buf543 = reinterpret_tensor(buf540, (1568, 320), (320, 1), 0); del buf540  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf542, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg377_1, (1280, 320), (1, 1280), 0), out=buf543)
        del arg377_1
        buf547 = reinterpret_tensor(buf517, (8, 196, 320), (62720, 320, 1), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [x_706, layer_norm_149], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf536, buf543, arg378_1, arg379_1, arg380_1, buf547, 1568, 320, grid=grid(1568), stream=stream0)
        del arg379_1
        del arg380_1
        buf548 = buf522; del buf522  # reuse
        # Topologically Sorted Source Nodes: [conv2d_59], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg383_1, buf548, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg383_1
        # Topologically Sorted Source Nodes: [conv2d_59], Original ATen: [aten.convolution]
        buf549 = extern_kernels.convolution(reinterpret_tensor(buf547, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf548, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf549, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf553 = buf527; del buf527  # reuse
        # Topologically Sorted Source Nodes: [x_709], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf549, arg384_1, arg385_1, arg386_1, buf553, 392, 320, grid=grid(392), stream=stream0)
        del arg384_1
        del arg385_1
        del arg386_1
        del buf549
        buf554 = buf528; del buf528  # reuse
        # Topologically Sorted Source Nodes: [linear_242], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg388_1, reinterpret_tensor(buf553, (392, 320), (320, 1), 0), reinterpret_tensor(arg387_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf554)
        del arg387_1
        del arg388_1
        buf555 = reinterpret_tensor(buf510, (1568, 320), (320, 1), 0); del buf510  # reuse
        # Topologically Sorted Source Nodes: [linear_241], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg382_1, reinterpret_tensor(buf547, (1568, 320), (320, 1), 0), reinterpret_tensor(arg381_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf555)
        del arg381_1
        del arg382_1
        del buf547
        # Topologically Sorted Source Nodes: [x_710], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf556 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf555, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf554, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf554, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf557 = buf556[0]
        del buf556
        buf561 = buf555; del buf555  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf557, (1568, 320), (320, 1), 0), reinterpret_tensor(arg389_1, (320, 320), (1, 320), 0), out=buf561)
        del arg389_1
        buf562 = reinterpret_tensor(buf561, (8, 196, 320), (62720, 320, 1), 0); del buf561  # reuse
        buf566 = reinterpret_tensor(buf557, (8, 196, 320), (62720, 320, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [x_706, x_714, layer_norm_151], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf562, buf536, buf543, arg378_1, arg390_1, arg391_1, arg392_1, buf566, 1568, 320, grid=grid(1568), stream=stream0)
        del arg378_1
        del arg390_1
        del arg391_1
        del arg392_1
        buf567 = reinterpret_tensor(buf542, (1568, 1280), (1280, 1), 0); del buf542  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf566, (1568, 320), (320, 1), 0), reinterpret_tensor(arg393_1, (320, 1280), (1, 320), 0), out=buf567)
        del arg393_1
        buf568 = reinterpret_tensor(buf567, (8, 196, 1280), (250880, 1280, 1), 0); del buf567  # reuse
        # Topologically Sorted Source Nodes: [x_716], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf568, arg394_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg394_1
        buf569 = reinterpret_tensor(buf566, (1568, 320), (320, 1), 0); del buf566  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf568, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg395_1, (1280, 320), (1, 1280), 0), out=buf569)
        del arg395_1
        buf573 = reinterpret_tensor(buf543, (8, 196, 320), (62720, 320, 1), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [x_720, layer_norm_152], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf562, buf569, arg396_1, arg397_1, arg398_1, buf573, 1568, 320, grid=grid(1568), stream=stream0)
        del arg397_1
        del arg398_1
        buf574 = buf548; del buf548  # reuse
        # Topologically Sorted Source Nodes: [conv2d_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg401_1, buf574, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg401_1
        # Topologically Sorted Source Nodes: [conv2d_60], Original ATen: [aten.convolution]
        buf575 = extern_kernels.convolution(reinterpret_tensor(buf573, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf574, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf575, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf579 = buf553; del buf553  # reuse
        # Topologically Sorted Source Nodes: [x_723], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf575, arg402_1, arg403_1, arg404_1, buf579, 392, 320, grid=grid(392), stream=stream0)
        del arg402_1
        del arg403_1
        del arg404_1
        del buf575
        buf580 = buf554; del buf554  # reuse
        # Topologically Sorted Source Nodes: [linear_247], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg406_1, reinterpret_tensor(buf579, (392, 320), (320, 1), 0), reinterpret_tensor(arg405_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf580)
        del arg405_1
        del arg406_1
        buf581 = reinterpret_tensor(buf536, (1568, 320), (320, 1), 0); del buf536  # reuse
        # Topologically Sorted Source Nodes: [linear_246], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg400_1, reinterpret_tensor(buf573, (1568, 320), (320, 1), 0), reinterpret_tensor(arg399_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf581)
        del arg399_1
        del arg400_1
        del buf573
        # Topologically Sorted Source Nodes: [x_724], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf582 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf581, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf580, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf580, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf583 = buf582[0]
        del buf582
        buf587 = buf581; del buf581  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf583, (1568, 320), (320, 1), 0), reinterpret_tensor(arg407_1, (320, 320), (1, 320), 0), out=buf587)
        del arg407_1
        buf588 = reinterpret_tensor(buf587, (8, 196, 320), (62720, 320, 1), 0); del buf587  # reuse
        buf592 = reinterpret_tensor(buf583, (8, 196, 320), (62720, 320, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [x_720, x_728, layer_norm_154], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf588, buf562, buf569, arg396_1, arg408_1, arg409_1, arg410_1, buf592, 1568, 320, grid=grid(1568), stream=stream0)
        del arg396_1
        del arg408_1
        del arg409_1
        del arg410_1
        buf593 = reinterpret_tensor(buf568, (1568, 1280), (1280, 1), 0); del buf568  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf592, (1568, 320), (320, 1), 0), reinterpret_tensor(arg411_1, (320, 1280), (1, 320), 0), out=buf593)
        del arg411_1
        buf594 = reinterpret_tensor(buf593, (8, 196, 1280), (250880, 1280, 1), 0); del buf593  # reuse
        # Topologically Sorted Source Nodes: [x_730], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf594, arg412_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg412_1
        buf595 = reinterpret_tensor(buf592, (1568, 320), (320, 1), 0); del buf592  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf594, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg413_1, (1280, 320), (1, 1280), 0), out=buf595)
        del arg413_1
        buf599 = reinterpret_tensor(buf569, (8, 196, 320), (62720, 320, 1), 0); del buf569  # reuse
        # Topologically Sorted Source Nodes: [x_734, layer_norm_155], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf588, buf595, arg414_1, arg415_1, arg416_1, buf599, 1568, 320, grid=grid(1568), stream=stream0)
        del arg415_1
        del arg416_1
        buf600 = buf574; del buf574  # reuse
        # Topologically Sorted Source Nodes: [conv2d_61], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg419_1, buf600, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg419_1
        # Topologically Sorted Source Nodes: [conv2d_61], Original ATen: [aten.convolution]
        buf601 = extern_kernels.convolution(reinterpret_tensor(buf599, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf600, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf601, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf605 = buf579; del buf579  # reuse
        # Topologically Sorted Source Nodes: [x_737], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf601, arg420_1, arg421_1, arg422_1, buf605, 392, 320, grid=grid(392), stream=stream0)
        del arg420_1
        del arg421_1
        del arg422_1
        del buf601
        buf606 = buf580; del buf580  # reuse
        # Topologically Sorted Source Nodes: [linear_252], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg424_1, reinterpret_tensor(buf605, (392, 320), (320, 1), 0), reinterpret_tensor(arg423_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf606)
        del arg423_1
        del arg424_1
        buf607 = reinterpret_tensor(buf562, (1568, 320), (320, 1), 0); del buf562  # reuse
        # Topologically Sorted Source Nodes: [linear_251], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg418_1, reinterpret_tensor(buf599, (1568, 320), (320, 1), 0), reinterpret_tensor(arg417_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf607)
        del arg417_1
        del arg418_1
        del buf599
        # Topologically Sorted Source Nodes: [x_738], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf608 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf607, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf606, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf606, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf609 = buf608[0]
        del buf608
        buf613 = buf607; del buf607  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf609, (1568, 320), (320, 1), 0), reinterpret_tensor(arg425_1, (320, 320), (1, 320), 0), out=buf613)
        del arg425_1
        buf614 = reinterpret_tensor(buf613, (8, 196, 320), (62720, 320, 1), 0); del buf613  # reuse
        buf618 = reinterpret_tensor(buf609, (8, 196, 320), (62720, 320, 1), 0); del buf609  # reuse
        # Topologically Sorted Source Nodes: [x_734, x_742, layer_norm_157], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf614, buf588, buf595, arg414_1, arg426_1, arg427_1, arg428_1, buf618, 1568, 320, grid=grid(1568), stream=stream0)
        del arg414_1
        del arg426_1
        del arg427_1
        del arg428_1
        buf619 = reinterpret_tensor(buf594, (1568, 1280), (1280, 1), 0); del buf594  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf618, (1568, 320), (320, 1), 0), reinterpret_tensor(arg429_1, (320, 1280), (1, 320), 0), out=buf619)
        del arg429_1
        buf620 = reinterpret_tensor(buf619, (8, 196, 1280), (250880, 1280, 1), 0); del buf619  # reuse
        # Topologically Sorted Source Nodes: [x_744], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf620, arg430_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg430_1
        buf621 = reinterpret_tensor(buf618, (1568, 320), (320, 1), 0); del buf618  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf620, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg431_1, (1280, 320), (1, 1280), 0), out=buf621)
        del arg431_1
        buf625 = reinterpret_tensor(buf595, (8, 196, 320), (62720, 320, 1), 0); del buf595  # reuse
        # Topologically Sorted Source Nodes: [x_748, layer_norm_158], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf614, buf621, arg432_1, arg433_1, arg434_1, buf625, 1568, 320, grid=grid(1568), stream=stream0)
        del arg433_1
        del arg434_1
        buf626 = buf600; del buf600  # reuse
        # Topologically Sorted Source Nodes: [conv2d_62], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg437_1, buf626, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg437_1
        # Topologically Sorted Source Nodes: [conv2d_62], Original ATen: [aten.convolution]
        buf627 = extern_kernels.convolution(reinterpret_tensor(buf625, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf626, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf627, (8, 320, 7, 7), (15680, 1, 2240, 320))
        buf631 = buf605; del buf605  # reuse
        # Topologically Sorted Source Nodes: [x_751], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf627, arg438_1, arg439_1, arg440_1, buf631, 392, 320, grid=grid(392), stream=stream0)
        del arg438_1
        del arg439_1
        del arg440_1
        del buf627
        buf632 = buf606; del buf606  # reuse
        # Topologically Sorted Source Nodes: [linear_257], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg442_1, reinterpret_tensor(buf631, (392, 320), (320, 1), 0), reinterpret_tensor(arg441_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf632)
        del arg441_1
        del arg442_1
        buf633 = reinterpret_tensor(buf588, (1568, 320), (320, 1), 0); del buf588  # reuse
        # Topologically Sorted Source Nodes: [linear_256], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg436_1, reinterpret_tensor(buf625, (1568, 320), (320, 1), 0), reinterpret_tensor(arg435_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf633)
        del arg435_1
        del arg436_1
        del buf625
        # Topologically Sorted Source Nodes: [x_752], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf634 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf633, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf632, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf632, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        buf635 = buf634[0]
        del buf634
        buf639 = buf633; del buf633  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf635, (1568, 320), (320, 1), 0), reinterpret_tensor(arg443_1, (320, 320), (1, 320), 0), out=buf639)
        del arg443_1
        buf640 = reinterpret_tensor(buf639, (8, 196, 320), (62720, 320, 1), 0); del buf639  # reuse
        buf644 = reinterpret_tensor(buf635, (8, 196, 320), (62720, 320, 1), 0); del buf635  # reuse
        # Topologically Sorted Source Nodes: [x_748, x_756, layer_norm_160], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf640, buf614, buf621, arg432_1, arg444_1, arg445_1, arg446_1, buf644, 1568, 320, grid=grid(1568), stream=stream0)
        del arg432_1
        del arg444_1
        del arg445_1
        del arg446_1
        buf645 = reinterpret_tensor(buf620, (1568, 1280), (1280, 1), 0); del buf620  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf644, (1568, 320), (320, 1), 0), reinterpret_tensor(arg447_1, (320, 1280), (1, 320), 0), out=buf645)
        del arg447_1
        buf646 = reinterpret_tensor(buf645, (8, 196, 1280), (250880, 1280, 1), 0); del buf645  # reuse
        # Topologically Sorted Source Nodes: [x_758], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf646, arg448_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg448_1
        buf647 = reinterpret_tensor(buf644, (1568, 320), (320, 1), 0); del buf644  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf646, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg449_1, (1280, 320), (1, 1280), 0), out=buf647)
        del arg449_1
        buf651 = reinterpret_tensor(buf621, (8, 196, 320), (62720, 320, 1), 0); del buf621  # reuse
        # Topologically Sorted Source Nodes: [x_762, layer_norm_161], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf640, buf647, arg450_1, arg451_1, arg452_1, buf651, 1568, 320, grid=grid(1568), stream=stream0)
        del arg451_1
        del arg452_1
        buf652 = buf626; del buf626  # reuse
        # Topologically Sorted Source Nodes: [conv2d_63], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(arg455_1, buf652, 102400, 4, grid=grid(102400, 4), stream=stream0)
        del arg455_1
        # Topologically Sorted Source Nodes: [conv2d_63], Original ATen: [aten.convolution]
        buf653 = extern_kernels.convolution(reinterpret_tensor(buf651, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf652, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf653, (8, 320, 7, 7), (15680, 1, 2240, 320))
        del buf652
        buf657 = buf631; del buf631  # reuse
        # Topologically Sorted Source Nodes: [x_765], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf653, arg456_1, arg457_1, arg458_1, buf657, 392, 320, grid=grid(392), stream=stream0)
        del arg456_1
        del arg457_1
        del arg458_1
        del buf653
        buf658 = buf632; del buf632  # reuse
        # Topologically Sorted Source Nodes: [linear_262], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg460_1, reinterpret_tensor(buf657, (392, 320), (320, 1), 0), reinterpret_tensor(arg459_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf658)
        del arg459_1
        del arg460_1
        del buf657
        buf659 = reinterpret_tensor(buf614, (1568, 320), (320, 1), 0); del buf614  # reuse
        # Topologically Sorted Source Nodes: [linear_261], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg454_1, reinterpret_tensor(buf651, (1568, 320), (320, 1), 0), reinterpret_tensor(arg453_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf659)
        del arg453_1
        del arg454_1
        del buf651
        # Topologically Sorted Source Nodes: [x_766], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf660 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf659, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf658, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf658, (8, 5, 49, 64), (31360, 64, 640, 1), 320), None, False)
        del buf658
        buf661 = buf660[0]
        del buf660
        buf665 = buf659; del buf659  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf661, (1568, 320), (320, 1), 0), reinterpret_tensor(arg461_1, (320, 320), (1, 320), 0), out=buf665)
        del arg461_1
        buf666 = reinterpret_tensor(buf665, (8, 196, 320), (62720, 320, 1), 0); del buf665  # reuse
        buf670 = reinterpret_tensor(buf661, (8, 196, 320), (62720, 320, 1), 0); del buf661  # reuse
        # Topologically Sorted Source Nodes: [x_762, x_770, layer_norm_163], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf666, buf640, buf647, arg450_1, arg462_1, arg463_1, arg464_1, buf670, 1568, 320, grid=grid(1568), stream=stream0)
        del arg450_1
        del arg462_1
        del arg463_1
        del arg464_1
        del buf640
        del buf647
        buf671 = reinterpret_tensor(buf646, (1568, 1280), (1280, 1), 0); del buf646  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf670, (1568, 320), (320, 1), 0), reinterpret_tensor(arg465_1, (320, 1280), (1, 320), 0), out=buf671)
        del arg465_1
        buf672 = reinterpret_tensor(buf671, (8, 196, 1280), (250880, 1280, 1), 0); del buf671  # reuse
        # Topologically Sorted Source Nodes: [x_772], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf672, arg466_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg466_1
        buf673 = reinterpret_tensor(buf670, (1568, 320), (320, 1), 0); del buf670  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf672, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg467_1, (1280, 320), (1, 1280), 0), out=buf673)
        del arg467_1
        del buf672
        buf674 = reinterpret_tensor(buf673, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf673  # reuse
        # Topologically Sorted Source Nodes: [x_777], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf674, buf666, arg468_1, 501760, grid=grid(501760), stream=stream0)
        del arg468_1
        del buf666
        buf675 = empty_strided_cuda((512, 320, 2, 2), (1280, 1, 640, 320), torch.float32)
        # Topologically Sorted Source Nodes: [x_777, conv2d_64], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_34.run(arg469_1, buf675, 163840, 4, grid=grid(163840, 4), stream=stream0)
        del arg469_1
        # Topologically Sorted Source Nodes: [x_777, conv2d_64], Original ATen: [aten.clone, aten.convolution]
        buf676 = extern_kernels.convolution(buf674, buf675, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf676, (8, 512, 7, 7), (25088, 1, 3584, 512))
        del buf674
        del buf675
        buf680 = empty_strided_cuda((8, 49, 512), (25088, 512, 1), torch.float32)
        buf684 = empty_strided_cuda((8, 49, 512), (25088, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_779, layer_norm_165], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_35.run(buf676, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, buf680, buf684, 392, 512, grid=grid(392), stream=stream0)
        del arg470_1
        del arg471_1
        del arg472_1
        del arg473_1
        del arg474_1
        buf685 = empty_strided_cuda((392, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_267], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg478_1, reinterpret_tensor(buf684, (392, 512), (512, 1), 0), reinterpret_tensor(arg477_1, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf685)
        del arg477_1
        del arg478_1
        buf686 = reinterpret_tensor(buf676, (392, 512), (512, 1), 0); del buf676  # reuse
        # Topologically Sorted Source Nodes: [linear_266], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg476_1, reinterpret_tensor(buf684, (392, 512), (512, 1), 0), reinterpret_tensor(arg475_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf686)
        del arg475_1
        del arg476_1
        del buf684
        # Topologically Sorted Source Nodes: [x_781], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf687 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf686, (8, 8, 49, 64), (25088, 64, 512, 1), 0), reinterpret_tensor(buf685, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf685, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), None, False)
        buf688 = buf687[0]
        del buf687
        buf692 = buf686; del buf686  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf688, (392, 512), (512, 1), 0), reinterpret_tensor(arg479_1, (512, 512), (1, 512), 0), out=buf692)
        del arg479_1
        buf696 = reinterpret_tensor(buf688, (8, 49, 512), (25088, 512, 1), 0); del buf688  # reuse
        # Topologically Sorted Source Nodes: [x_785, layer_norm_166], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_36.run(buf680, buf692, arg480_1, arg481_1, arg482_1, buf696, 392, 512, grid=grid(392), stream=stream0)
        del arg481_1
        del arg482_1
        buf697 = reinterpret_tensor(buf198, (392, 2048), (2048, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf696, (392, 512), (512, 1), 0), reinterpret_tensor(arg483_1, (512, 2048), (1, 512), 0), out=buf697)
        del arg483_1
        buf698 = reinterpret_tensor(buf697, (8, 49, 2048), (100352, 2048, 1), 0); del buf697  # reuse
        # Topologically Sorted Source Nodes: [x_787], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf698, arg484_1, 802816, grid=grid(802816), stream=stream0)
        del arg484_1
        buf699 = reinterpret_tensor(buf696, (392, 512), (512, 1), 0); del buf696  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf698, (392, 2048), (2048, 1), 0), reinterpret_tensor(arg485_1, (2048, 512), (1, 2048), 0), out=buf699)
        del arg485_1
        buf700 = reinterpret_tensor(buf699, (8, 49, 512), (25088, 512, 1), 0); del buf699  # reuse
        # Topologically Sorted Source Nodes: [x_785, x_791], Original ATen: [aten.add]
        triton_poi_fused_add_38.run(buf700, buf680, buf692, arg480_1, arg486_1, 200704, grid=grid(200704), stream=stream0)
        del arg480_1
        del arg486_1
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf701 = extern_kernels.convolution(reinterpret_tensor(buf700, (8, 512, 7, 7), (25088, 1, 3584, 512), 0), arg487_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf701, (8, 512, 7, 7), (25088, 1, 3584, 512))
        del arg487_1
        buf705 = reinterpret_tensor(buf692, (8, 49, 512), (25088, 512, 1), 0); del buf692  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_167], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_39.run(buf701, arg488_1, buf700, arg489_1, arg490_1, buf705, 392, 512, grid=grid(392), stream=stream0)
        del arg489_1
        del arg490_1
        buf706 = buf685; del buf685  # reuse
        # Topologically Sorted Source Nodes: [linear_272], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg494_1, reinterpret_tensor(buf705, (392, 512), (512, 1), 0), reinterpret_tensor(arg493_1, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf706)
        del arg493_1
        del arg494_1
        buf707 = reinterpret_tensor(buf680, (392, 512), (512, 1), 0); del buf680  # reuse
        # Topologically Sorted Source Nodes: [linear_271], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg492_1, reinterpret_tensor(buf705, (392, 512), (512, 1), 0), reinterpret_tensor(arg491_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf707)
        del arg491_1
        del arg492_1
        del buf705
        # Topologically Sorted Source Nodes: [x_794], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf708 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf707, (8, 8, 49, 64), (25088, 64, 512, 1), 0), reinterpret_tensor(buf706, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf706, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), None, False)
        buf709 = buf708[0]
        del buf708
        buf713 = buf707; del buf707  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf709, (392, 512), (512, 1), 0), reinterpret_tensor(arg495_1, (512, 512), (1, 512), 0), out=buf713)
        del arg495_1
        buf714 = reinterpret_tensor(buf713, (8, 49, 512), (25088, 512, 1), 0); del buf713  # reuse
        buf718 = reinterpret_tensor(buf709, (8, 49, 512), (25088, 512, 1), 0); del buf709  # reuse
        # Topologically Sorted Source Nodes: [x_798, layer_norm_168], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_40.run(buf714, buf701, arg488_1, buf700, arg496_1, arg497_1, arg498_1, buf718, 392, 512, grid=grid(392), stream=stream0)
        del arg488_1
        del arg496_1
        del arg497_1
        del arg498_1
        buf719 = reinterpret_tensor(buf698, (392, 2048), (2048, 1), 0); del buf698  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf718, (392, 512), (512, 1), 0), reinterpret_tensor(arg499_1, (512, 2048), (1, 512), 0), out=buf719)
        del arg499_1
        buf720 = reinterpret_tensor(buf719, (8, 49, 2048), (100352, 2048, 1), 0); del buf719  # reuse
        # Topologically Sorted Source Nodes: [x_800], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf720, arg500_1, 802816, grid=grid(802816), stream=stream0)
        del arg500_1
        buf721 = reinterpret_tensor(buf718, (392, 512), (512, 1), 0); del buf718  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf720, (392, 2048), (2048, 1), 0), reinterpret_tensor(arg501_1, (2048, 512), (1, 2048), 0), out=buf721)
        del arg501_1
        buf725 = reinterpret_tensor(buf701, (8, 49, 512), (25088, 512, 1), 0); del buf701  # reuse
        # Topologically Sorted Source Nodes: [x_804, layer_norm_169], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_36.run(buf714, buf721, arg502_1, arg503_1, arg504_1, buf725, 392, 512, grid=grid(392), stream=stream0)
        del arg503_1
        del arg504_1
        buf726 = buf706; del buf706  # reuse
        # Topologically Sorted Source Nodes: [linear_277], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg508_1, reinterpret_tensor(buf725, (392, 512), (512, 1), 0), reinterpret_tensor(arg507_1, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf726)
        del arg507_1
        del arg508_1
        buf727 = reinterpret_tensor(buf700, (392, 512), (512, 1), 0); del buf700  # reuse
        # Topologically Sorted Source Nodes: [linear_276], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg506_1, reinterpret_tensor(buf725, (392, 512), (512, 1), 0), reinterpret_tensor(arg505_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf727)
        del arg505_1
        del arg506_1
        del buf725
        # Topologically Sorted Source Nodes: [x_805], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf728 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf727, (8, 8, 49, 64), (25088, 64, 512, 1), 0), reinterpret_tensor(buf726, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf726, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), None, False)
        del buf726
        buf729 = buf728[0]
        del buf728
        buf733 = buf727; del buf727  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf729, (392, 512), (512, 1), 0), reinterpret_tensor(arg509_1, (512, 512), (1, 512), 0), out=buf733)
        del arg509_1
        buf734 = reinterpret_tensor(buf733, (8, 49, 512), (25088, 512, 1), 0); del buf733  # reuse
        buf738 = reinterpret_tensor(buf729, (8, 49, 512), (25088, 512, 1), 0); del buf729  # reuse
        # Topologically Sorted Source Nodes: [x_804, x_809, layer_norm_170], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_41.run(buf734, buf714, buf721, arg502_1, arg510_1, arg511_1, arg512_1, buf738, 392, 512, grid=grid(392), stream=stream0)
        del arg502_1
        del arg510_1
        del arg511_1
        del arg512_1
        del buf714
        del buf721
        buf739 = reinterpret_tensor(buf720, (392, 2048), (2048, 1), 0); del buf720  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf738, (392, 512), (512, 1), 0), reinterpret_tensor(arg513_1, (512, 2048), (1, 512), 0), out=buf739)
        del arg513_1
        buf740 = reinterpret_tensor(buf739, (8, 49, 2048), (100352, 2048, 1), 0); del buf739  # reuse
        # Topologically Sorted Source Nodes: [x_811], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf740, arg514_1, 802816, grid=grid(802816), stream=stream0)
        del arg514_1
        buf741 = reinterpret_tensor(buf738, (392, 512), (512, 1), 0); del buf738  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf740, (392, 2048), (2048, 1), 0), reinterpret_tensor(arg515_1, (2048, 512), (1, 2048), 0), out=buf741)
        del arg515_1
        del buf740
        buf742 = empty_strided_cuda((8, 49, 1), (49, 1, 392), torch.float32)
        buf743 = empty_strided_cuda((8, 49, 1), (49, 1, 392), torch.float32)
        # Topologically Sorted Source Nodes: [x_815, x_816], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_42.run(buf734, buf741, arg516_1, buf742, buf743, 392, 512, grid=grid(392), stream=stream0)
        buf746 = empty_strided_cuda((8, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_815, x_816, x_817], Original ATen: [aten.add, aten.native_layer_norm, aten.mean]
        triton_per_fused_add_mean_native_layer_norm_43.run(buf734, buf741, arg516_1, buf742, buf743, arg517_1, arg518_1, buf746, 4096, 49, grid=grid(4096), stream=stream0)
        del arg516_1
        del arg517_1
        del arg518_1
        del buf734
        del buf741
        del buf742
        del buf743
        buf747 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_815, x_816, x_817, x_819], Original ATen: [aten.add, aten.native_layer_norm, aten.mean, aten.addmm]
        extern_kernels.addmm(arg520_1, buf746, reinterpret_tensor(arg519_1, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf747)
        del arg519_1
        del arg520_1
        del buf746
    return (buf747, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((320, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((512, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('twins_pcpvt_base', benchmark_compiled_module)
