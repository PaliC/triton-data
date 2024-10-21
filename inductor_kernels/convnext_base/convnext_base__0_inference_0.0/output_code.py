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


# kernel path: /tmp/torchinductor_sahanp/3f/c3fncxgde2thpvzyp2u4w2psqtj2hypvbt33orumjzg67j6yuyoc.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_5 => convolution_40
# Graph fragment:
#   %convolution_40 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32, 131072], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 82944
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
    tmp0 = tl.load(in_ptr0 + (x2 + (82944*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (248832*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cp/ccpwiv4jvbvrotwcrcklbxgckypzkqvt6p5m5ptvxpjexxn4aol7.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_5 => convolution_40
# Graph fragment:
#   %convolution_40 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
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


# kernel path: /tmp/torchinductor_sahanp/ul/culmjphnkvfq5ijn67uiktsyb2c2knu45qrpybjuiojhcgs2anxi.py
# Topologically Sorted Source Nodes: [x_416], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_416 => add_154, add_155, clone_74, mul_226, mul_227, rsqrt_41, sub_41, var_mean_41
# Graph fragment:
#   %clone_74 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_155,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_41 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_74, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_74, %getitem_83), kwargs = {})
#   %add_154 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_82, 1e-06), kwargs = {})
#   %rsqrt_41 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_154,), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %rsqrt_41), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_226, %arg3_1), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_227, %arg4_1), kwargs = {})
triton_per_fused_native_layer_norm_2 = async_compile.triton('triton_per_fused_native_layer_norm_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 41472
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
    tmp22 = 1e-06
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ew/cewyp7a6dhef6imwg6t2pinhddw3n3jzxy2sszvc74tbtznqt6mc.py
# Topologically Sorted Source Nodes: [x_422], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_422 => add_158, erf_36, mul_230, mul_231, mul_232
# Graph fragment:
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_182, 0.5), kwargs = {})
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_182, 0.7071067811865476), kwargs = {})
#   %erf_36 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_231,), kwargs = {})
#   %add_158 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_36, 1), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_230, %add_158), kwargs = {})
triton_poi_fused_gelu_3 = async_compile.triton('triton_poi_fused_gelu_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21233664
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


# kernel path: /tmp/torchinductor_sahanp/pn/cpntcckvloqr5436s53atf43f3uvf36z7ddpwmomvboouoavet2k.py
# Topologically Sorted Source Nodes: [x_427, x_428], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_427 => mul_233
#   x_428 => add_159
# Graph fragment:
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_160, %view_185), kwargs = {})
#   %add_159 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_233, %permute_156), kwargs = {})
triton_poi_fused_add_mul_4 = async_compile.triton('triton_poi_fused_add_mul_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4h/c4hzpmuxruywvsvundtzvrvcvwgoaq3v4v26okaupybuja5nnfss.py
# Topologically Sorted Source Nodes: [x_438, x_439], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_438 => mul_239
#   x_439 => add_163
# Graph fragment:
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_164, %view_190), kwargs = {})
#   %add_163 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_239, %add_159), kwargs = {})
triton_poi_fused_add_mul_5 = async_compile.triton('triton_poi_fused_add_mul_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qz/cqzk2djko7deka4k4g257r3sgscrqphkqxq4ph6jvmxo753jaqwj.py
# Topologically Sorted Source Nodes: [x_452], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_452 => add_168, add_169, mul_246, mul_247, rsqrt_45, sub_45, var_mean_45
# Graph fragment:
#   %var_mean_45 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_169, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_169, %getitem_91), kwargs = {})
#   %add_168 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_90, 1e-06), kwargs = {})
#   %rsqrt_45 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_168,), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %rsqrt_45), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_246, %arg32_1), kwargs = {})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_247, %arg33_1), kwargs = {})
triton_per_fused_native_layer_norm_6 = async_compile.triton('triton_per_fused_native_layer_norm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 41472
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
    tmp3 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (r1 + (128*x0)), xmask, other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tmp6 - tmp16
    tmp24 = 128.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp33, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3q/c3qhcoehsbiyeo5droqj7264cvcc2qe2gpchia3v3qgknjk3mylw.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_6 => convolution_44
# Graph fragment:
#   %convolution_44 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_170, %arg34_1, %arg35_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_7 = async_compile.triton('triton_poi_fused_convolution_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
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


# kernel path: /tmp/torchinductor_sahanp/2x/c2x3azxwbk6k6sd4lqvqhljma3abhxatkhoxx57qpdqlludd6avj.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_6 => convolution_44
# Graph fragment:
#   %convolution_44 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_170, %arg34_1, %arg35_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_8 = async_compile.triton('triton_poi_fused_convolution_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4t/c4tftypvafaz4razvquoaka4lb52py7so3ptz57xucxtkw7hifqa.py
# Topologically Sorted Source Nodes: [x_456], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_456 => add_170, add_171, mul_248, mul_249, rsqrt_46, sub_46, var_mean_46
# Graph fragment:
#   %var_mean_46 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_171, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_171, %getitem_93), kwargs = {})
#   %add_170 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_92, 1e-06), kwargs = {})
#   %rsqrt_46 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_170,), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %rsqrt_46), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_248, %arg38_1), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_249, %arg39_1), kwargs = {})
triton_red_fused_native_layer_norm_9 = async_compile.triton('triton_red_fused_native_layer_norm_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10368
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x2 = xindex % 1296
    x3 = (xindex // 1296)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 256.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-06
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tl.store(out_ptr2 + (r1 + (256*x0)), tmp20, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dv/cdv4afu4s2v5phrbadfywwudl2jiueery7gsalqbkdgraiczvdmg.py
# Topologically Sorted Source Nodes: [x_458], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_458 => add_172, erf_39, mul_250, mul_251, mul_252
# Graph fragment:
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_197, 0.5), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_197, 0.7071067811865476), kwargs = {})
#   %erf_39 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_251,), kwargs = {})
#   %add_172 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_39, 1), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_250, %add_172), kwargs = {})
triton_poi_fused_gelu_10 = async_compile.triton('triton_poi_fused_gelu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10616832
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


# kernel path: /tmp/torchinductor_sahanp/y2/cy24he3jq7p24lorafdvenjexgmsm4ld43ix4bzqaxk3umhfmzk5.py
# Topologically Sorted Source Nodes: [x_463, x_464], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_463 => mul_253
#   x_464 => add_173
# Graph fragment:
#   %mul_253 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_174, %view_200), kwargs = {})
#   %add_173 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_253, %convolution_44), kwargs = {})
triton_poi_fused_add_mul_11 = async_compile.triton('triton_poi_fused_add_mul_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sh/cshvqqze67thtr77bfwkoi4kiuyzhepytylpwy5s24uon3jtlujr.py
# Topologically Sorted Source Nodes: [x_488], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_488 => add_182, add_183, mul_266, mul_267, rsqrt_49, sub_49, var_mean_49
# Graph fragment:
#   %var_mean_49 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_183, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_183, %getitem_99), kwargs = {})
#   %add_182 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_98, 1e-06), kwargs = {})
#   %rsqrt_49 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_182,), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %rsqrt_49), kwargs = {})
#   %mul_267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_266, %arg63_1), kwargs = {})
#   %add_183 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_267, %arg64_1), kwargs = {})
triton_red_fused_native_layer_norm_12 = async_compile.triton('triton_red_fused_native_layer_norm_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10368
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x2 = xindex % 1296
    x3 = (xindex // 1296)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (256*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp6 = tmp4 + tmp5
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr3 + (r1 + (256*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17 - tmp8
        tmp19 = 256.0
        tmp20 = tmp9 / tmp19
        tmp21 = 1e-06
        tmp22 = tmp20 + tmp21
        tmp23 = libdevice.rsqrt(tmp22)
        tmp24 = tmp18 * tmp23
        tmp26 = tmp24 * tmp25
        tmp28 = tmp26 + tmp27
        tl.store(out_ptr2 + (r1 + (256*x0)), tmp28, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uv/cuv4yss53zn66x4iyepkhzcjurylts7zzhjhkulxekqbtn2wqw5t.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_7 => convolution_48
# Graph fragment:
#   %convolution_48 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_184, %arg65_1, %arg66_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_13 = async_compile.triton('triton_poi_fused_convolution_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (1024*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/me/cmeks5rhewydpxepa5ka5u5akpjvg6s76xaukqmiwkxkkz5k56ni.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_7 => convolution_48
# Graph fragment:
#   %convolution_48 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_184, %arg65_1, %arg66_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_14 = async_compile.triton('triton_poi_fused_convolution_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1327104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pp/cppo57kwzdyl54qbw4i5gmhnyqwefdivfxffdg3brykvckbinuz6.py
# Topologically Sorted Source Nodes: [x_492], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_492 => add_184, add_185, mul_268, mul_269, rsqrt_50, sub_50, var_mean_50
# Graph fragment:
#   %var_mean_50 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_185, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_185, %getitem_101), kwargs = {})
#   %add_184 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_100, 1e-06), kwargs = {})
#   %rsqrt_50 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_184,), kwargs = {})
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %rsqrt_50), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_268, %arg69_1), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_269, %arg70_1), kwargs = {})
triton_per_fused_native_layer_norm_15 = async_compile.triton('triton_per_fused_native_layer_norm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 2592
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
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ik/cikd4tjog4m37w6a66vrl2ioa32wpeyta2rh27wp23g4gwkuddwr.py
# Topologically Sorted Source Nodes: [x_494], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_494 => add_186, erf_42, mul_270, mul_271, mul_272
# Graph fragment:
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_212, 0.5), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_212, 0.7071067811865476), kwargs = {})
#   %erf_42 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_271,), kwargs = {})
#   %add_186 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_42, 1), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_270, %add_186), kwargs = {})
triton_poi_fused_gelu_16 = async_compile.triton('triton_poi_fused_gelu_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_16(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
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


# kernel path: /tmp/torchinductor_sahanp/7r/c7rane3bwjxk664coia7llniqj3uhfaqkzxtwh2hq3a3tujh7bcl.py
# Topologically Sorted Source Nodes: [x_499, x_500], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_499 => mul_273
#   x_500 => add_187
# Graph fragment:
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_188, %view_215), kwargs = {})
#   %add_187 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_273, %convolution_48), kwargs = {})
triton_poi_fused_add_mul_17 = async_compile.triton('triton_poi_fused_add_mul_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1327104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bw/cbwhwbui5yu2ew4qonjxplf45fnxgmvnedbas7hqktkpnjepzfhw.py
# Topologically Sorted Source Nodes: [x_532, x_533], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_532 => mul_291
#   x_533 => add_199
# Graph fragment:
#   %mul_291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_200, %view_230), kwargs = {})
#   %add_199 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_291, %add_195), kwargs = {})
triton_poi_fused_add_mul_18 = async_compile.triton('triton_poi_fused_add_mul_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1327104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oc/coc4dklz6tvbd2xhbzmllt7dzdfqc4kjjbyjj3tjhfgnbgxxqmoh.py
# Topologically Sorted Source Nodes: [x_788], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_788 => add_292, add_293, mul_430, mul_431, rsqrt_77, sub_77, var_mean_77
# Graph fragment:
#   %var_mean_77 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_293, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_293, %getitem_155), kwargs = {})
#   %add_292 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_154, 1e-06), kwargs = {})
#   %rsqrt_77 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_292,), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %rsqrt_77), kwargs = {})
#   %mul_431 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_430, %arg310_1), kwargs = {})
#   %add_293 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_431, %arg311_1), kwargs = {})
triton_per_fused_native_layer_norm_19 = async_compile.triton('triton_per_fused_native_layer_norm_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 2592
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
    tmp3 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), None)
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp7 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = tmp6 - tmp14
    tmp21 = 512.0
    tmp22 = tmp19 / tmp21
    tmp23 = 1e-06
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp30, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3a/c3av4uevjbdomdzc7fpxngciblypd6etx7k5g5bdp3jondabxesf.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_8 => convolution_76
# Graph fragment:
#   %convolution_76 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_294, %arg312_1, %arg313_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_20 = async_compile.triton('triton_poi_fused_convolution_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[524288, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 4
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (2048*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/m6/cm6k44tn6ufqoutkv2zwjekajckusm73hmvycbj3pjuk3jsfx7bw.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_8 => convolution_76
# Graph fragment:
#   %convolution_76 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_294, %arg312_1, %arg313_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_21 = async_compile.triton('triton_poi_fused_convolution_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 663552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3i/c3ic6mksntnn46msqbshsg75j7gcy2fo4w6nmpriuv4ab4ddl5ie.py
# Topologically Sorted Source Nodes: [x_792], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_792 => add_294, add_295, mul_432, mul_433, rsqrt_78, sub_78, var_mean_78
# Graph fragment:
#   %var_mean_78 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_295, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_295, %getitem_157), kwargs = {})
#   %add_294 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_156, 1e-06), kwargs = {})
#   %rsqrt_78 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_294,), kwargs = {})
#   %mul_432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %rsqrt_78), kwargs = {})
#   %mul_433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_432, %arg316_1), kwargs = {})
#   %add_295 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_433, %arg317_1), kwargs = {})
triton_per_fused_native_layer_norm_22 = async_compile.triton('triton_per_fused_native_layer_norm_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 648
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 1024, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp2 - tmp10
    tmp17 = 1024.0
    tmp18 = tmp15 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tn/ctnoc6nfs4ru2n6o7ttrjs6qka6wp6aggpdzgjjkwy6kihxppk5q.py
# Topologically Sorted Source Nodes: [x_794], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_794 => add_296, erf_69, mul_434, mul_435, mul_436
# Graph fragment:
#   %mul_434 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_347, 0.5), kwargs = {})
#   %mul_435 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_347, 0.7071067811865476), kwargs = {})
#   %erf_69 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_435,), kwargs = {})
#   %add_296 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_69, 1), kwargs = {})
#   %mul_436 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_434, %add_296), kwargs = {})
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
    xnumel = 2654208
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


# kernel path: /tmp/torchinductor_sahanp/cy/ccyiwfeswaycnfnwgjh4kkv2vkokgkihzpbheanwfzjgyhlcfx5l.py
# Topologically Sorted Source Nodes: [x_799, x_800], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_799 => mul_437
#   x_800 => add_297
# Graph fragment:
#   %mul_437 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_298, %view_350), kwargs = {})
#   %add_297 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_437, %convolution_76), kwargs = {})
triton_poi_fused_add_mul_24 = async_compile.triton('triton_poi_fused_add_mul_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 663552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lk/clkbvrck5tgv6wziljg5abwygsr6xgq3rp4bzicu5itigk3jv5fl.py
# Topologically Sorted Source Nodes: [x_821, x_822, x_823], Original ATen: [aten.mul, aten.add, aten.mean]
# Source node to ATen node mapping:
#   x_821 => mul_449
#   x_822 => add_305
#   x_823 => mean_1
# Graph fragment:
#   %mul_449 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_306, %view_360), kwargs = {})
#   %add_305 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_449, %add_301), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_305, [-1, -2], True), kwargs = {})
triton_red_fused_add_mean_mul_25 = async_compile.triton('triton_red_fused_add_mean_mul_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mean_mul_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mean_mul_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 81
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (82944*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x0 + (1024*r2) + (82944*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp10 = 81.0
    tmp11 = tmp8 / tmp10
    tl.store(out_ptr1 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nh/cnhlkduwh7b5igbvqquv6wcetdwbirztzfylivd7qo7yta4vvwps.py
# Topologically Sorted Source Nodes: [x_825], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_825 => add_306, add_307, mul_450, mul_451, rsqrt_81, sub_81, var_mean_81
# Graph fragment:
#   %var_mean_81 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_307, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_307, %getitem_163), kwargs = {})
#   %add_306 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_162, 1e-06), kwargs = {})
#   %rsqrt_81 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_306,), kwargs = {})
#   %mul_450 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %rsqrt_81), kwargs = {})
#   %mul_451 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_450, %arg341_1), kwargs = {})
#   %add_307 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_451, %arg342_1), kwargs = {})
triton_per_fused_native_layer_norm_26 = async_compile.triton('triton_per_fused_native_layer_norm_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_26(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 8
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
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp24, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg1_1, (128, ), (1, ))
    assert_size_stride(arg2_1, (8, 3, 288, 288), (248832, 82944, 288, 1))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (512, 128), (128, 1))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (128, 512), (512, 1))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, ), (1, ))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (512, 128), (128, 1))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (128, 512), (512, 1))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (512, 128), (128, 1))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (128, 512), (512, 1))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (256, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (1024, 256), (256, 1))
    assert_size_stride(arg41_1, (1024, ), (1, ))
    assert_size_stride(arg42_1, (256, 1024), (1024, 1))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (1024, 256), (256, 1))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (256, 1024), (1024, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (1024, 256), (256, 1))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (256, 1024), (1024, 1))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (256, ), (1, ))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (512, 256, 2, 2), (1024, 4, 2, 1))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (2048, 512), (512, 1))
    assert_size_stride(arg72_1, (2048, ), (1, ))
    assert_size_stride(arg73_1, (512, 2048), (2048, 1))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (2048, 512), (512, 1))
    assert_size_stride(arg81_1, (2048, ), (1, ))
    assert_size_stride(arg82_1, (512, 2048), (2048, 1))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg86_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (2048, 512), (512, 1))
    assert_size_stride(arg90_1, (2048, ), (1, ))
    assert_size_stride(arg91_1, (512, 2048), (2048, 1))
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg95_1, (512, ), (1, ))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (2048, 512), (512, 1))
    assert_size_stride(arg99_1, (2048, ), (1, ))
    assert_size_stride(arg100_1, (512, 2048), (2048, 1))
    assert_size_stride(arg101_1, (512, ), (1, ))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (512, ), (1, ))
    assert_size_stride(arg107_1, (2048, 512), (512, 1))
    assert_size_stride(arg108_1, (2048, ), (1, ))
    assert_size_stride(arg109_1, (512, 2048), (2048, 1))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (512, ), (1, ))
    assert_size_stride(arg112_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (2048, 512), (512, 1))
    assert_size_stride(arg117_1, (2048, ), (1, ))
    assert_size_stride(arg118_1, (512, 2048), (2048, 1))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg122_1, (512, ), (1, ))
    assert_size_stride(arg123_1, (512, ), (1, ))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (2048, 512), (512, 1))
    assert_size_stride(arg126_1, (2048, ), (1, ))
    assert_size_stride(arg127_1, (512, 2048), (2048, 1))
    assert_size_stride(arg128_1, (512, ), (1, ))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (512, ), (1, ))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (2048, 512), (512, 1))
    assert_size_stride(arg135_1, (2048, ), (1, ))
    assert_size_stride(arg136_1, (512, 2048), (2048, 1))
    assert_size_stride(arg137_1, (512, ), (1, ))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg140_1, (512, ), (1, ))
    assert_size_stride(arg141_1, (512, ), (1, ))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (2048, 512), (512, 1))
    assert_size_stride(arg144_1, (2048, ), (1, ))
    assert_size_stride(arg145_1, (512, 2048), (2048, 1))
    assert_size_stride(arg146_1, (512, ), (1, ))
    assert_size_stride(arg147_1, (512, ), (1, ))
    assert_size_stride(arg148_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (512, ), (1, ))
    assert_size_stride(arg151_1, (512, ), (1, ))
    assert_size_stride(arg152_1, (2048, 512), (512, 1))
    assert_size_stride(arg153_1, (2048, ), (1, ))
    assert_size_stride(arg154_1, (512, 2048), (2048, 1))
    assert_size_stride(arg155_1, (512, ), (1, ))
    assert_size_stride(arg156_1, (512, ), (1, ))
    assert_size_stride(arg157_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg158_1, (512, ), (1, ))
    assert_size_stride(arg159_1, (512, ), (1, ))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (2048, 512), (512, 1))
    assert_size_stride(arg162_1, (2048, ), (1, ))
    assert_size_stride(arg163_1, (512, 2048), (2048, 1))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg167_1, (512, ), (1, ))
    assert_size_stride(arg168_1, (512, ), (1, ))
    assert_size_stride(arg169_1, (512, ), (1, ))
    assert_size_stride(arg170_1, (2048, 512), (512, 1))
    assert_size_stride(arg171_1, (2048, ), (1, ))
    assert_size_stride(arg172_1, (512, 2048), (2048, 1))
    assert_size_stride(arg173_1, (512, ), (1, ))
    assert_size_stride(arg174_1, (512, ), (1, ))
    assert_size_stride(arg175_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg176_1, (512, ), (1, ))
    assert_size_stride(arg177_1, (512, ), (1, ))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (2048, 512), (512, 1))
    assert_size_stride(arg180_1, (2048, ), (1, ))
    assert_size_stride(arg181_1, (512, 2048), (2048, 1))
    assert_size_stride(arg182_1, (512, ), (1, ))
    assert_size_stride(arg183_1, (512, ), (1, ))
    assert_size_stride(arg184_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg185_1, (512, ), (1, ))
    assert_size_stride(arg186_1, (512, ), (1, ))
    assert_size_stride(arg187_1, (512, ), (1, ))
    assert_size_stride(arg188_1, (2048, 512), (512, 1))
    assert_size_stride(arg189_1, (2048, ), (1, ))
    assert_size_stride(arg190_1, (512, 2048), (2048, 1))
    assert_size_stride(arg191_1, (512, ), (1, ))
    assert_size_stride(arg192_1, (512, ), (1, ))
    assert_size_stride(arg193_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (512, ), (1, ))
    assert_size_stride(arg196_1, (512, ), (1, ))
    assert_size_stride(arg197_1, (2048, 512), (512, 1))
    assert_size_stride(arg198_1, (2048, ), (1, ))
    assert_size_stride(arg199_1, (512, 2048), (2048, 1))
    assert_size_stride(arg200_1, (512, ), (1, ))
    assert_size_stride(arg201_1, (512, ), (1, ))
    assert_size_stride(arg202_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg203_1, (512, ), (1, ))
    assert_size_stride(arg204_1, (512, ), (1, ))
    assert_size_stride(arg205_1, (512, ), (1, ))
    assert_size_stride(arg206_1, (2048, 512), (512, 1))
    assert_size_stride(arg207_1, (2048, ), (1, ))
    assert_size_stride(arg208_1, (512, 2048), (2048, 1))
    assert_size_stride(arg209_1, (512, ), (1, ))
    assert_size_stride(arg210_1, (512, ), (1, ))
    assert_size_stride(arg211_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg212_1, (512, ), (1, ))
    assert_size_stride(arg213_1, (512, ), (1, ))
    assert_size_stride(arg214_1, (512, ), (1, ))
    assert_size_stride(arg215_1, (2048, 512), (512, 1))
    assert_size_stride(arg216_1, (2048, ), (1, ))
    assert_size_stride(arg217_1, (512, 2048), (2048, 1))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (512, ), (1, ))
    assert_size_stride(arg220_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg221_1, (512, ), (1, ))
    assert_size_stride(arg222_1, (512, ), (1, ))
    assert_size_stride(arg223_1, (512, ), (1, ))
    assert_size_stride(arg224_1, (2048, 512), (512, 1))
    assert_size_stride(arg225_1, (2048, ), (1, ))
    assert_size_stride(arg226_1, (512, 2048), (2048, 1))
    assert_size_stride(arg227_1, (512, ), (1, ))
    assert_size_stride(arg228_1, (512, ), (1, ))
    assert_size_stride(arg229_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg230_1, (512, ), (1, ))
    assert_size_stride(arg231_1, (512, ), (1, ))
    assert_size_stride(arg232_1, (512, ), (1, ))
    assert_size_stride(arg233_1, (2048, 512), (512, 1))
    assert_size_stride(arg234_1, (2048, ), (1, ))
    assert_size_stride(arg235_1, (512, 2048), (2048, 1))
    assert_size_stride(arg236_1, (512, ), (1, ))
    assert_size_stride(arg237_1, (512, ), (1, ))
    assert_size_stride(arg238_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg239_1, (512, ), (1, ))
    assert_size_stride(arg240_1, (512, ), (1, ))
    assert_size_stride(arg241_1, (512, ), (1, ))
    assert_size_stride(arg242_1, (2048, 512), (512, 1))
    assert_size_stride(arg243_1, (2048, ), (1, ))
    assert_size_stride(arg244_1, (512, 2048), (2048, 1))
    assert_size_stride(arg245_1, (512, ), (1, ))
    assert_size_stride(arg246_1, (512, ), (1, ))
    assert_size_stride(arg247_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg248_1, (512, ), (1, ))
    assert_size_stride(arg249_1, (512, ), (1, ))
    assert_size_stride(arg250_1, (512, ), (1, ))
    assert_size_stride(arg251_1, (2048, 512), (512, 1))
    assert_size_stride(arg252_1, (2048, ), (1, ))
    assert_size_stride(arg253_1, (512, 2048), (2048, 1))
    assert_size_stride(arg254_1, (512, ), (1, ))
    assert_size_stride(arg255_1, (512, ), (1, ))
    assert_size_stride(arg256_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg257_1, (512, ), (1, ))
    assert_size_stride(arg258_1, (512, ), (1, ))
    assert_size_stride(arg259_1, (512, ), (1, ))
    assert_size_stride(arg260_1, (2048, 512), (512, 1))
    assert_size_stride(arg261_1, (2048, ), (1, ))
    assert_size_stride(arg262_1, (512, 2048), (2048, 1))
    assert_size_stride(arg263_1, (512, ), (1, ))
    assert_size_stride(arg264_1, (512, ), (1, ))
    assert_size_stride(arg265_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg266_1, (512, ), (1, ))
    assert_size_stride(arg267_1, (512, ), (1, ))
    assert_size_stride(arg268_1, (512, ), (1, ))
    assert_size_stride(arg269_1, (2048, 512), (512, 1))
    assert_size_stride(arg270_1, (2048, ), (1, ))
    assert_size_stride(arg271_1, (512, 2048), (2048, 1))
    assert_size_stride(arg272_1, (512, ), (1, ))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg275_1, (512, ), (1, ))
    assert_size_stride(arg276_1, (512, ), (1, ))
    assert_size_stride(arg277_1, (512, ), (1, ))
    assert_size_stride(arg278_1, (2048, 512), (512, 1))
    assert_size_stride(arg279_1, (2048, ), (1, ))
    assert_size_stride(arg280_1, (512, 2048), (2048, 1))
    assert_size_stride(arg281_1, (512, ), (1, ))
    assert_size_stride(arg282_1, (512, ), (1, ))
    assert_size_stride(arg283_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg284_1, (512, ), (1, ))
    assert_size_stride(arg285_1, (512, ), (1, ))
    assert_size_stride(arg286_1, (512, ), (1, ))
    assert_size_stride(arg287_1, (2048, 512), (512, 1))
    assert_size_stride(arg288_1, (2048, ), (1, ))
    assert_size_stride(arg289_1, (512, 2048), (2048, 1))
    assert_size_stride(arg290_1, (512, ), (1, ))
    assert_size_stride(arg291_1, (512, ), (1, ))
    assert_size_stride(arg292_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg293_1, (512, ), (1, ))
    assert_size_stride(arg294_1, (512, ), (1, ))
    assert_size_stride(arg295_1, (512, ), (1, ))
    assert_size_stride(arg296_1, (2048, 512), (512, 1))
    assert_size_stride(arg297_1, (2048, ), (1, ))
    assert_size_stride(arg298_1, (512, 2048), (2048, 1))
    assert_size_stride(arg299_1, (512, ), (1, ))
    assert_size_stride(arg300_1, (512, ), (1, ))
    assert_size_stride(arg301_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg302_1, (512, ), (1, ))
    assert_size_stride(arg303_1, (512, ), (1, ))
    assert_size_stride(arg304_1, (512, ), (1, ))
    assert_size_stride(arg305_1, (2048, 512), (512, 1))
    assert_size_stride(arg306_1, (2048, ), (1, ))
    assert_size_stride(arg307_1, (512, 2048), (2048, 1))
    assert_size_stride(arg308_1, (512, ), (1, ))
    assert_size_stride(arg309_1, (512, ), (1, ))
    assert_size_stride(arg310_1, (512, ), (1, ))
    assert_size_stride(arg311_1, (512, ), (1, ))
    assert_size_stride(arg312_1, (1024, 512, 2, 2), (2048, 4, 2, 1))
    assert_size_stride(arg313_1, (1024, ), (1, ))
    assert_size_stride(arg314_1, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg315_1, (1024, ), (1, ))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg319_1, (4096, ), (1, ))
    assert_size_stride(arg320_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg321_1, (1024, ), (1, ))
    assert_size_stride(arg322_1, (1024, ), (1, ))
    assert_size_stride(arg323_1, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg324_1, (1024, ), (1, ))
    assert_size_stride(arg325_1, (1024, ), (1, ))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg328_1, (4096, ), (1, ))
    assert_size_stride(arg329_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg330_1, (1024, ), (1, ))
    assert_size_stride(arg331_1, (1024, ), (1, ))
    assert_size_stride(arg332_1, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg333_1, (1024, ), (1, ))
    assert_size_stride(arg334_1, (1024, ), (1, ))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg337_1, (4096, ), (1, ))
    assert_size_stride(arg338_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg339_1, (1024, ), (1, ))
    assert_size_stride(arg340_1, (1024, ), (1, ))
    assert_size_stride(arg341_1, (1024, ), (1, ))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg344_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 288, 288), (248832, 1, 864, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg2_1, buf0, 24, 82944, grid=grid(24, 82944), stream=stream0)
        del arg2_1
        buf1 = empty_strided_cuda((128, 3, 4, 4), (48, 1, 12, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 384, 16, grid=grid(384, 16), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 128, 72, 72), (663552, 1, 9216, 128))
        del buf0
        del buf1
        buf6 = empty_strided_cuda((8, 72, 72, 128), (663552, 9216, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_416], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2.run(buf2, arg1_1, arg3_1, arg4_1, buf6, 41472, 128, grid=grid(41472), stream=stream0)
        del arg1_1
        del arg3_1
        del arg4_1
        # Topologically Sorted Source Nodes: [x_418], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 128, 72, 72), (663552, 1, 9216, 128), 0), arg5_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf7, (8, 128, 72, 72), (663552, 1, 9216, 128))
        del arg5_1
        buf11 = reinterpret_tensor(buf2, (8, 72, 72, 128), (663552, 9216, 128, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_420], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2.run(buf7, arg6_1, arg7_1, arg8_1, buf11, 41472, 128, grid=grid(41472), stream=stream0)
        del arg6_1
        del arg7_1
        del arg8_1
        del buf7
        buf12 = empty_strided_cuda((41472, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (41472, 128), (128, 1), 0), reinterpret_tensor(arg9_1, (128, 512), (1, 128), 0), out=buf12)
        del arg9_1
        buf13 = reinterpret_tensor(buf12, (8, 72, 72, 512), (2654208, 36864, 512, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_422], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf13, arg10_1, 21233664, grid=grid(21233664), stream=stream0)
        del arg10_1
        buf14 = reinterpret_tensor(buf11, (41472, 128), (128, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf13, (41472, 512), (512, 1), 0), reinterpret_tensor(arg11_1, (512, 128), (1, 512), 0), out=buf14)
        del arg11_1
        buf15 = reinterpret_tensor(buf14, (8, 128, 72, 72), (663552, 1, 9216, 128), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_427, x_428], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_4.run(buf15, arg12_1, arg13_1, buf6, 5308416, grid=grid(5308416), stream=stream0)
        del arg12_1
        del arg13_1
        # Topologically Sorted Source Nodes: [x_429], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg14_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf16, (8, 128, 72, 72), (663552, 1, 9216, 128))
        del arg14_1
        buf20 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_431], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2.run(buf16, arg15_1, arg16_1, arg17_1, buf20, 41472, 128, grid=grid(41472), stream=stream0)
        del arg15_1
        del arg16_1
        del arg17_1
        del buf16
        buf21 = reinterpret_tensor(buf13, (41472, 512), (512, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (41472, 128), (128, 1), 0), reinterpret_tensor(arg18_1, (128, 512), (1, 128), 0), out=buf21)
        del arg18_1
        buf22 = reinterpret_tensor(buf21, (8, 72, 72, 512), (2654208, 36864, 512, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_433], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf22, arg19_1, 21233664, grid=grid(21233664), stream=stream0)
        del arg19_1
        buf23 = reinterpret_tensor(buf20, (41472, 128), (128, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (41472, 512), (512, 1), 0), reinterpret_tensor(arg20_1, (512, 128), (1, 512), 0), out=buf23)
        del arg20_1
        buf24 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_438, x_439], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_5.run(buf24, buf23, arg21_1, arg22_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg21_1
        del arg22_1
        # Topologically Sorted Source Nodes: [x_440], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, arg23_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf25, (8, 128, 72, 72), (663552, 1, 9216, 128))
        del arg23_1
        buf29 = reinterpret_tensor(buf23, (8, 72, 72, 128), (663552, 9216, 128, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_442], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2.run(buf25, arg24_1, arg25_1, arg26_1, buf29, 41472, 128, grid=grid(41472), stream=stream0)
        del arg24_1
        del arg25_1
        del arg26_1
        buf30 = reinterpret_tensor(buf22, (41472, 512), (512, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (41472, 128), (128, 1), 0), reinterpret_tensor(arg27_1, (128, 512), (1, 128), 0), out=buf30)
        del arg27_1
        buf31 = reinterpret_tensor(buf30, (8, 72, 72, 512), (2654208, 36864, 512, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_444], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf31, arg28_1, 21233664, grid=grid(21233664), stream=stream0)
        del arg28_1
        buf32 = reinterpret_tensor(buf29, (41472, 128), (128, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (41472, 512), (512, 1), 0), reinterpret_tensor(arg29_1, (512, 128), (1, 512), 0), out=buf32)
        del arg29_1
        del buf31
        buf36 = reinterpret_tensor(buf25, (8, 72, 72, 128), (663552, 9216, 128, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_452], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf32, arg30_1, arg31_1, buf24, arg32_1, arg33_1, buf36, 41472, 128, grid=grid(41472), stream=stream0)
        del arg30_1
        del arg31_1
        del arg32_1
        del arg33_1
        del buf24
        del buf32
        buf37 = empty_strided_cuda((256, 128, 2, 2), (512, 1, 256, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(arg34_1, buf37, 32768, 4, grid=grid(32768, 4), stream=stream0)
        del arg34_1
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(reinterpret_tensor(buf36, (8, 128, 72, 72), (663552, 1, 9216, 128), 0), buf37, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 256, 36, 36), (331776, 1, 9216, 256))
        del buf37
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf39, arg35_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg35_1
        # Topologically Sorted Source Nodes: [x_454], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg36_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf40, (8, 256, 36, 36), (331776, 1, 9216, 256))
        del arg36_1
        buf44 = empty_strided_cuda((8, 36, 36, 256), (331776, 9216, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_456], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_9.run(buf40, arg37_1, arg38_1, arg39_1, buf44, 10368, 256, grid=grid(10368), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del buf40
        buf45 = empty_strided_cuda((10368, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (10368, 256), (256, 1), 0), reinterpret_tensor(arg40_1, (256, 1024), (1, 256), 0), out=buf45)
        del arg40_1
        buf46 = reinterpret_tensor(buf45, (8, 36, 36, 1024), (1327104, 36864, 1024, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [x_458], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf46, arg41_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg41_1
        buf47 = reinterpret_tensor(buf44, (10368, 256), (256, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (10368, 1024), (1024, 1), 0), reinterpret_tensor(arg42_1, (1024, 256), (1, 1024), 0), out=buf47)
        del arg42_1
        buf48 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_463, x_464], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_11.run(buf48, buf47, arg43_1, arg44_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg43_1
        del arg44_1
        # Topologically Sorted Source Nodes: [x_465], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg45_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf49, (8, 256, 36, 36), (331776, 1, 9216, 256))
        del arg45_1
        buf53 = reinterpret_tensor(buf47, (8, 36, 36, 256), (331776, 9216, 256, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_467], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_9.run(buf49, arg46_1, arg47_1, arg48_1, buf53, 10368, 256, grid=grid(10368), stream=stream0)
        del arg46_1
        del arg47_1
        del arg48_1
        del buf49
        buf54 = reinterpret_tensor(buf46, (10368, 1024), (1024, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (10368, 256), (256, 1), 0), reinterpret_tensor(arg49_1, (256, 1024), (1, 256), 0), out=buf54)
        del arg49_1
        buf55 = reinterpret_tensor(buf54, (8, 36, 36, 1024), (1327104, 36864, 1024, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_469], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf55, arg50_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg50_1
        buf56 = reinterpret_tensor(buf53, (10368, 256), (256, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf55, (10368, 1024), (1024, 1), 0), reinterpret_tensor(arg51_1, (1024, 256), (1, 1024), 0), out=buf56)
        del arg51_1
        buf57 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_474, x_475], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_11.run(buf57, buf56, arg52_1, arg53_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg52_1
        del arg53_1
        # Topologically Sorted Source Nodes: [x_476], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg54_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf58, (8, 256, 36, 36), (331776, 1, 9216, 256))
        del arg54_1
        buf62 = reinterpret_tensor(buf56, (8, 36, 36, 256), (331776, 9216, 256, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_478], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_9.run(buf58, arg55_1, arg56_1, arg57_1, buf62, 10368, 256, grid=grid(10368), stream=stream0)
        del arg55_1
        del arg56_1
        del arg57_1
        buf63 = reinterpret_tensor(buf55, (10368, 1024), (1024, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf62, (10368, 256), (256, 1), 0), reinterpret_tensor(arg58_1, (256, 1024), (1, 256), 0), out=buf63)
        del arg58_1
        buf64 = reinterpret_tensor(buf63, (8, 36, 36, 1024), (1327104, 36864, 1024, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [x_480], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf64, arg59_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg59_1
        buf65 = reinterpret_tensor(buf62, (10368, 256), (256, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (10368, 1024), (1024, 1), 0), reinterpret_tensor(arg60_1, (1024, 256), (1, 1024), 0), out=buf65)
        del arg60_1
        del buf64
        buf69 = reinterpret_tensor(buf58, (8, 36, 36, 256), (331776, 9216, 256, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_488], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_12.run(buf65, arg61_1, arg62_1, buf57, arg63_1, arg64_1, buf69, 10368, 256, grid=grid(10368), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        del arg64_1
        del buf57
        del buf65
        buf70 = empty_strided_cuda((512, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg65_1, buf70, 131072, 4, grid=grid(131072, 4), stream=stream0)
        del arg65_1
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(reinterpret_tensor(buf69, (8, 256, 36, 36), (331776, 1, 9216, 256), 0), buf70, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del buf70
        buf72 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf72, arg66_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg66_1
        # Topologically Sorted Source Nodes: [x_490], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg67_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf73, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg67_1
        buf77 = empty_strided_cuda((8, 18, 18, 512), (165888, 9216, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_492], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf73, arg68_1, arg69_1, arg70_1, buf77, 2592, 512, grid=grid(2592), stream=stream0)
        del arg68_1
        del arg69_1
        del arg70_1
        del buf73
        buf78 = reinterpret_tensor(buf36, (2592, 2048), (2048, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (2592, 512), (512, 1), 0), reinterpret_tensor(arg71_1, (512, 2048), (1, 512), 0), out=buf78)
        del arg71_1
        buf79 = reinterpret_tensor(buf78, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_494], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf79, arg72_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg72_1
        buf80 = reinterpret_tensor(buf77, (2592, 512), (512, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg73_1, (2048, 512), (1, 2048), 0), out=buf80)
        del arg73_1
        buf81 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_499, x_500], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf81, buf80, arg74_1, arg75_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg74_1
        del arg75_1
        # Topologically Sorted Source Nodes: [x_501], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg76_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf82, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg76_1
        buf86 = reinterpret_tensor(buf80, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_503], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf82, arg77_1, arg78_1, arg79_1, buf86, 2592, 512, grid=grid(2592), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del buf82
        buf87 = reinterpret_tensor(buf79, (2592, 2048), (2048, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf86, (2592, 512), (512, 1), 0), reinterpret_tensor(arg80_1, (512, 2048), (1, 512), 0), out=buf87)
        del arg80_1
        buf88 = reinterpret_tensor(buf87, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_505], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf88, arg81_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg81_1
        buf89 = reinterpret_tensor(buf86, (2592, 512), (512, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf88, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg82_1, (2048, 512), (1, 2048), 0), out=buf89)
        del arg82_1
        buf90 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_510, x_511], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf90, buf89, arg83_1, arg84_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg83_1
        del arg84_1
        # Topologically Sorted Source Nodes: [x_512], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, arg85_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf91, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg85_1
        buf95 = reinterpret_tensor(buf89, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [x_514], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf91, arg86_1, arg87_1, arg88_1, buf95, 2592, 512, grid=grid(2592), stream=stream0)
        del arg86_1
        del arg87_1
        del arg88_1
        del buf91
        buf96 = reinterpret_tensor(buf88, (2592, 2048), (2048, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (2592, 512), (512, 1), 0), reinterpret_tensor(arg89_1, (512, 2048), (1, 512), 0), out=buf96)
        del arg89_1
        buf97 = reinterpret_tensor(buf96, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_516], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf97, arg90_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg90_1
        buf98 = reinterpret_tensor(buf95, (2592, 512), (512, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg91_1, (2048, 512), (1, 2048), 0), out=buf98)
        del arg91_1
        buf99 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_521, x_522], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf99, buf98, arg92_1, arg93_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg92_1
        del arg93_1
        # Topologically Sorted Source Nodes: [x_523], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg94_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf100, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg94_1
        buf104 = reinterpret_tensor(buf98, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [x_525], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf100, arg95_1, arg96_1, arg97_1, buf104, 2592, 512, grid=grid(2592), stream=stream0)
        del arg95_1
        del arg96_1
        del arg97_1
        del buf100
        buf105 = reinterpret_tensor(buf97, (2592, 2048), (2048, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf104, (2592, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 2048), (1, 512), 0), out=buf105)
        del arg98_1
        buf106 = reinterpret_tensor(buf105, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_527], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf106, arg99_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg99_1
        buf107 = reinterpret_tensor(buf104, (2592, 512), (512, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg100_1, (2048, 512), (1, 2048), 0), out=buf107)
        del arg100_1
        buf108 = reinterpret_tensor(buf107, (8, 512, 18, 18), (165888, 1, 9216, 512), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [x_532, x_533], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_18.run(buf108, arg101_1, arg102_1, buf99, 1327104, grid=grid(1327104), stream=stream0)
        del arg101_1
        del arg102_1
        # Topologically Sorted Source Nodes: [x_534], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, arg103_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf109, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg103_1
        buf113 = reinterpret_tensor(buf99, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [x_536], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf109, arg104_1, arg105_1, arg106_1, buf113, 2592, 512, grid=grid(2592), stream=stream0)
        del arg104_1
        del arg105_1
        del arg106_1
        del buf109
        buf114 = reinterpret_tensor(buf106, (2592, 2048), (2048, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf113, (2592, 512), (512, 1), 0), reinterpret_tensor(arg107_1, (512, 2048), (1, 512), 0), out=buf114)
        del arg107_1
        buf115 = reinterpret_tensor(buf114, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [x_538], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf115, arg108_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg108_1
        buf116 = reinterpret_tensor(buf113, (2592, 512), (512, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf115, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg109_1, (2048, 512), (1, 2048), 0), out=buf116)
        del arg109_1
        buf117 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_543, x_544], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf117, buf116, arg110_1, arg111_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg110_1
        del arg111_1
        # Topologically Sorted Source Nodes: [x_545], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg112_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf118, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg112_1
        buf122 = reinterpret_tensor(buf116, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_547], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf118, arg113_1, arg114_1, arg115_1, buf122, 2592, 512, grid=grid(2592), stream=stream0)
        del arg113_1
        del arg114_1
        del arg115_1
        del buf118
        buf123 = reinterpret_tensor(buf115, (2592, 2048), (2048, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (2592, 512), (512, 1), 0), reinterpret_tensor(arg116_1, (512, 2048), (1, 512), 0), out=buf123)
        del arg116_1
        buf124 = reinterpret_tensor(buf123, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_549], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf124, arg117_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg117_1
        buf125 = reinterpret_tensor(buf122, (2592, 512), (512, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg118_1, (2048, 512), (1, 2048), 0), out=buf125)
        del arg118_1
        buf126 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [x_554, x_555], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf126, buf125, arg119_1, arg120_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg119_1
        del arg120_1
        # Topologically Sorted Source Nodes: [x_556], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, arg121_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf127, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg121_1
        buf131 = reinterpret_tensor(buf125, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [x_558], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf127, arg122_1, arg123_1, arg124_1, buf131, 2592, 512, grid=grid(2592), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del buf127
        buf132 = reinterpret_tensor(buf124, (2592, 2048), (2048, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf131, (2592, 512), (512, 1), 0), reinterpret_tensor(arg125_1, (512, 2048), (1, 512), 0), out=buf132)
        del arg125_1
        buf133 = reinterpret_tensor(buf132, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [x_560], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf133, arg126_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg126_1
        buf134 = reinterpret_tensor(buf131, (2592, 512), (512, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf133, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg127_1, (2048, 512), (1, 2048), 0), out=buf134)
        del arg127_1
        buf135 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_565, x_566], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf135, buf134, arg128_1, arg129_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg128_1
        del arg129_1
        # Topologically Sorted Source Nodes: [x_567], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg130_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf136, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg130_1
        buf140 = reinterpret_tensor(buf134, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [x_569], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf136, arg131_1, arg132_1, arg133_1, buf140, 2592, 512, grid=grid(2592), stream=stream0)
        del arg131_1
        del arg132_1
        del arg133_1
        del buf136
        buf141 = reinterpret_tensor(buf133, (2592, 2048), (2048, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (2592, 512), (512, 1), 0), reinterpret_tensor(arg134_1, (512, 2048), (1, 512), 0), out=buf141)
        del arg134_1
        buf142 = reinterpret_tensor(buf141, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [x_571], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf142, arg135_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg135_1
        buf143 = reinterpret_tensor(buf140, (2592, 512), (512, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg136_1, (2048, 512), (1, 2048), 0), out=buf143)
        del arg136_1
        buf144 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [x_576, x_577], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf144, buf143, arg137_1, arg138_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg137_1
        del arg138_1
        # Topologically Sorted Source Nodes: [x_578], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg139_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf145, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg139_1
        buf149 = reinterpret_tensor(buf143, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_580], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf145, arg140_1, arg141_1, arg142_1, buf149, 2592, 512, grid=grid(2592), stream=stream0)
        del arg140_1
        del arg141_1
        del arg142_1
        del buf145
        buf150 = reinterpret_tensor(buf142, (2592, 2048), (2048, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (2592, 512), (512, 1), 0), reinterpret_tensor(arg143_1, (512, 2048), (1, 512), 0), out=buf150)
        del arg143_1
        buf151 = reinterpret_tensor(buf150, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [x_582], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf151, arg144_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg144_1
        buf152 = reinterpret_tensor(buf149, (2592, 512), (512, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg145_1, (2048, 512), (1, 2048), 0), out=buf152)
        del arg145_1
        buf153 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_587, x_588], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf153, buf152, arg146_1, arg147_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg146_1
        del arg147_1
        # Topologically Sorted Source Nodes: [x_589], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, arg148_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf154, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg148_1
        buf158 = reinterpret_tensor(buf152, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [x_591], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf154, arg149_1, arg150_1, arg151_1, buf158, 2592, 512, grid=grid(2592), stream=stream0)
        del arg149_1
        del arg150_1
        del arg151_1
        del buf154
        buf159 = reinterpret_tensor(buf151, (2592, 2048), (2048, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (2592, 512), (512, 1), 0), reinterpret_tensor(arg152_1, (512, 2048), (1, 512), 0), out=buf159)
        del arg152_1
        buf160 = reinterpret_tensor(buf159, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_593], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf160, arg153_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg153_1
        buf161 = reinterpret_tensor(buf158, (2592, 512), (512, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg154_1, (2048, 512), (1, 2048), 0), out=buf161)
        del arg154_1
        buf162 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [x_598, x_599], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf162, buf161, arg155_1, arg156_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg155_1
        del arg156_1
        # Topologically Sorted Source Nodes: [x_600], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, arg157_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf163, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg157_1
        buf167 = reinterpret_tensor(buf161, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_602], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf163, arg158_1, arg159_1, arg160_1, buf167, 2592, 512, grid=grid(2592), stream=stream0)
        del arg158_1
        del arg159_1
        del arg160_1
        del buf163
        buf168 = reinterpret_tensor(buf160, (2592, 2048), (2048, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (2592, 512), (512, 1), 0), reinterpret_tensor(arg161_1, (512, 2048), (1, 512), 0), out=buf168)
        del arg161_1
        buf169 = reinterpret_tensor(buf168, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_604], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf169, arg162_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg162_1
        buf170 = reinterpret_tensor(buf167, (2592, 512), (512, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg163_1, (2048, 512), (1, 2048), 0), out=buf170)
        del arg163_1
        buf171 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_609, x_610], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf171, buf170, arg164_1, arg165_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg164_1
        del arg165_1
        # Topologically Sorted Source Nodes: [x_611], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, arg166_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf172, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg166_1
        buf176 = reinterpret_tensor(buf170, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [x_613], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf172, arg167_1, arg168_1, arg169_1, buf176, 2592, 512, grid=grid(2592), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del buf172
        buf177 = reinterpret_tensor(buf169, (2592, 2048), (2048, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (2592, 512), (512, 1), 0), reinterpret_tensor(arg170_1, (512, 2048), (1, 512), 0), out=buf177)
        del arg170_1
        buf178 = reinterpret_tensor(buf177, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_615], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf178, arg171_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg171_1
        buf179 = reinterpret_tensor(buf176, (2592, 512), (512, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg172_1, (2048, 512), (1, 2048), 0), out=buf179)
        del arg172_1
        buf180 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_620, x_621], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf180, buf179, arg173_1, arg174_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg173_1
        del arg174_1
        # Topologically Sorted Source Nodes: [x_622], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, arg175_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf181, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg175_1
        buf185 = reinterpret_tensor(buf179, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_624], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf181, arg176_1, arg177_1, arg178_1, buf185, 2592, 512, grid=grid(2592), stream=stream0)
        del arg176_1
        del arg177_1
        del arg178_1
        del buf181
        buf186 = reinterpret_tensor(buf178, (2592, 2048), (2048, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf185, (2592, 512), (512, 1), 0), reinterpret_tensor(arg179_1, (512, 2048), (1, 512), 0), out=buf186)
        del arg179_1
        buf187 = reinterpret_tensor(buf186, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_626], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf187, arg180_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg180_1
        buf188 = reinterpret_tensor(buf185, (2592, 512), (512, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg181_1, (2048, 512), (1, 2048), 0), out=buf188)
        del arg181_1
        buf189 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [x_631, x_632], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf189, buf188, arg182_1, arg183_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg182_1
        del arg183_1
        # Topologically Sorted Source Nodes: [x_633], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, arg184_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf190, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg184_1
        buf194 = reinterpret_tensor(buf188, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_635], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf190, arg185_1, arg186_1, arg187_1, buf194, 2592, 512, grid=grid(2592), stream=stream0)
        del arg185_1
        del arg186_1
        del arg187_1
        del buf190
        buf195 = reinterpret_tensor(buf187, (2592, 2048), (2048, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (2592, 512), (512, 1), 0), reinterpret_tensor(arg188_1, (512, 2048), (1, 512), 0), out=buf195)
        del arg188_1
        buf196 = reinterpret_tensor(buf195, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_637], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf196, arg189_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg189_1
        buf197 = reinterpret_tensor(buf194, (2592, 512), (512, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg190_1, (2048, 512), (1, 2048), 0), out=buf197)
        del arg190_1
        buf198 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_642, x_643], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf198, buf197, arg191_1, arg192_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg191_1
        del arg192_1
        # Topologically Sorted Source Nodes: [x_644], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, arg193_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf199, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg193_1
        buf203 = reinterpret_tensor(buf197, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [x_646], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf199, arg194_1, arg195_1, arg196_1, buf203, 2592, 512, grid=grid(2592), stream=stream0)
        del arg194_1
        del arg195_1
        del arg196_1
        del buf199
        buf204 = reinterpret_tensor(buf196, (2592, 2048), (2048, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (2592, 512), (512, 1), 0), reinterpret_tensor(arg197_1, (512, 2048), (1, 512), 0), out=buf204)
        del arg197_1
        buf205 = reinterpret_tensor(buf204, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_648], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf205, arg198_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg198_1
        buf206 = reinterpret_tensor(buf203, (2592, 512), (512, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf205, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg199_1, (2048, 512), (1, 2048), 0), out=buf206)
        del arg199_1
        buf207 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [x_653, x_654], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf207, buf206, arg200_1, arg201_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg200_1
        del arg201_1
        # Topologically Sorted Source Nodes: [x_655], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, arg202_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf208, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg202_1
        buf212 = reinterpret_tensor(buf206, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [x_657], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf208, arg203_1, arg204_1, arg205_1, buf212, 2592, 512, grid=grid(2592), stream=stream0)
        del arg203_1
        del arg204_1
        del arg205_1
        del buf208
        buf213 = reinterpret_tensor(buf205, (2592, 2048), (2048, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (2592, 512), (512, 1), 0), reinterpret_tensor(arg206_1, (512, 2048), (1, 512), 0), out=buf213)
        del arg206_1
        buf214 = reinterpret_tensor(buf213, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [x_659], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf214, arg207_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg207_1
        buf215 = reinterpret_tensor(buf212, (2592, 512), (512, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg208_1, (2048, 512), (1, 2048), 0), out=buf215)
        del arg208_1
        buf216 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_664, x_665], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf216, buf215, arg209_1, arg210_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg209_1
        del arg210_1
        # Topologically Sorted Source Nodes: [x_666], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, arg211_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf217, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg211_1
        buf221 = reinterpret_tensor(buf215, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [x_668], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf217, arg212_1, arg213_1, arg214_1, buf221, 2592, 512, grid=grid(2592), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del buf217
        buf222 = reinterpret_tensor(buf214, (2592, 2048), (2048, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (2592, 512), (512, 1), 0), reinterpret_tensor(arg215_1, (512, 2048), (1, 512), 0), out=buf222)
        del arg215_1
        buf223 = reinterpret_tensor(buf222, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [x_670], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf223, arg216_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg216_1
        buf224 = reinterpret_tensor(buf221, (2592, 512), (512, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg217_1, (2048, 512), (1, 2048), 0), out=buf224)
        del arg217_1
        buf225 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [x_675, x_676], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf225, buf224, arg218_1, arg219_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg218_1
        del arg219_1
        # Topologically Sorted Source Nodes: [x_677], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, arg220_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf226, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg220_1
        buf230 = reinterpret_tensor(buf224, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [x_679], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf226, arg221_1, arg222_1, arg223_1, buf230, 2592, 512, grid=grid(2592), stream=stream0)
        del arg221_1
        del arg222_1
        del arg223_1
        del buf226
        buf231 = reinterpret_tensor(buf223, (2592, 2048), (2048, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (2592, 512), (512, 1), 0), reinterpret_tensor(arg224_1, (512, 2048), (1, 512), 0), out=buf231)
        del arg224_1
        buf232 = reinterpret_tensor(buf231, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [x_681], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf232, arg225_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg225_1
        buf233 = reinterpret_tensor(buf230, (2592, 512), (512, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg226_1, (2048, 512), (1, 2048), 0), out=buf233)
        del arg226_1
        buf234 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [x_686, x_687], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf234, buf233, arg227_1, arg228_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg227_1
        del arg228_1
        # Topologically Sorted Source Nodes: [x_688], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, arg229_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf235, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg229_1
        buf239 = reinterpret_tensor(buf233, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_690], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf235, arg230_1, arg231_1, arg232_1, buf239, 2592, 512, grid=grid(2592), stream=stream0)
        del arg230_1
        del arg231_1
        del arg232_1
        del buf235
        buf240 = reinterpret_tensor(buf232, (2592, 2048), (2048, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (2592, 512), (512, 1), 0), reinterpret_tensor(arg233_1, (512, 2048), (1, 512), 0), out=buf240)
        del arg233_1
        buf241 = reinterpret_tensor(buf240, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [x_692], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf241, arg234_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg234_1
        buf242 = reinterpret_tensor(buf239, (2592, 512), (512, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf241, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg235_1, (2048, 512), (1, 2048), 0), out=buf242)
        del arg235_1
        buf243 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [x_697, x_698], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf243, buf242, arg236_1, arg237_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg236_1
        del arg237_1
        # Topologically Sorted Source Nodes: [x_699], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, arg238_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf244, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg238_1
        buf248 = reinterpret_tensor(buf242, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [x_701], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf244, arg239_1, arg240_1, arg241_1, buf248, 2592, 512, grid=grid(2592), stream=stream0)
        del arg239_1
        del arg240_1
        del arg241_1
        del buf244
        buf249 = reinterpret_tensor(buf241, (2592, 2048), (2048, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (2592, 512), (512, 1), 0), reinterpret_tensor(arg242_1, (512, 2048), (1, 512), 0), out=buf249)
        del arg242_1
        buf250 = reinterpret_tensor(buf249, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [x_703], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf250, arg243_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg243_1
        buf251 = reinterpret_tensor(buf248, (2592, 512), (512, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg244_1, (2048, 512), (1, 2048), 0), out=buf251)
        del arg244_1
        buf252 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [x_708, x_709], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf252, buf251, arg245_1, arg246_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg245_1
        del arg246_1
        # Topologically Sorted Source Nodes: [x_710], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, arg247_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf253, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg247_1
        buf257 = reinterpret_tensor(buf251, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [x_712], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf253, arg248_1, arg249_1, arg250_1, buf257, 2592, 512, grid=grid(2592), stream=stream0)
        del arg248_1
        del arg249_1
        del arg250_1
        del buf253
        buf258 = reinterpret_tensor(buf250, (2592, 2048), (2048, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (2592, 512), (512, 1), 0), reinterpret_tensor(arg251_1, (512, 2048), (1, 512), 0), out=buf258)
        del arg251_1
        buf259 = reinterpret_tensor(buf258, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [x_714], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf259, arg252_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg252_1
        buf260 = reinterpret_tensor(buf257, (2592, 512), (512, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf259, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg253_1, (2048, 512), (1, 2048), 0), out=buf260)
        del arg253_1
        buf261 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [x_719, x_720], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf261, buf260, arg254_1, arg255_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg254_1
        del arg255_1
        # Topologically Sorted Source Nodes: [x_721], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, arg256_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf262, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg256_1
        buf266 = reinterpret_tensor(buf260, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [x_723], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf262, arg257_1, arg258_1, arg259_1, buf266, 2592, 512, grid=grid(2592), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del buf262
        buf267 = reinterpret_tensor(buf259, (2592, 2048), (2048, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf266, (2592, 512), (512, 1), 0), reinterpret_tensor(arg260_1, (512, 2048), (1, 512), 0), out=buf267)
        del arg260_1
        buf268 = reinterpret_tensor(buf267, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [x_725], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf268, arg261_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg261_1
        buf269 = reinterpret_tensor(buf266, (2592, 512), (512, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf268, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg262_1, (2048, 512), (1, 2048), 0), out=buf269)
        del arg262_1
        buf270 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [x_730, x_731], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf270, buf269, arg263_1, arg264_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg263_1
        del arg264_1
        # Topologically Sorted Source Nodes: [x_732], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, arg265_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf271, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg265_1
        buf275 = reinterpret_tensor(buf269, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [x_734], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf271, arg266_1, arg267_1, arg268_1, buf275, 2592, 512, grid=grid(2592), stream=stream0)
        del arg266_1
        del arg267_1
        del arg268_1
        del buf271
        buf276 = reinterpret_tensor(buf268, (2592, 2048), (2048, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf275, (2592, 512), (512, 1), 0), reinterpret_tensor(arg269_1, (512, 2048), (1, 512), 0), out=buf276)
        del arg269_1
        buf277 = reinterpret_tensor(buf276, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [x_736], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf277, arg270_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg270_1
        buf278 = reinterpret_tensor(buf275, (2592, 512), (512, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf277, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg271_1, (2048, 512), (1, 2048), 0), out=buf278)
        del arg271_1
        buf279 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [x_741, x_742], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf279, buf278, arg272_1, arg273_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg272_1
        del arg273_1
        # Topologically Sorted Source Nodes: [x_743], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, arg274_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf280, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg274_1
        buf284 = reinterpret_tensor(buf278, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [x_745], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf280, arg275_1, arg276_1, arg277_1, buf284, 2592, 512, grid=grid(2592), stream=stream0)
        del arg275_1
        del arg276_1
        del arg277_1
        del buf280
        buf285 = reinterpret_tensor(buf277, (2592, 2048), (2048, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (2592, 512), (512, 1), 0), reinterpret_tensor(arg278_1, (512, 2048), (1, 512), 0), out=buf285)
        del arg278_1
        buf286 = reinterpret_tensor(buf285, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [x_747], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf286, arg279_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg279_1
        buf287 = reinterpret_tensor(buf284, (2592, 512), (512, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf286, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg280_1, (2048, 512), (1, 2048), 0), out=buf287)
        del arg280_1
        buf288 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [x_752, x_753], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf288, buf287, arg281_1, arg282_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg281_1
        del arg282_1
        # Topologically Sorted Source Nodes: [x_754], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, arg283_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf289, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg283_1
        buf293 = reinterpret_tensor(buf287, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [x_756], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf289, arg284_1, arg285_1, arg286_1, buf293, 2592, 512, grid=grid(2592), stream=stream0)
        del arg284_1
        del arg285_1
        del arg286_1
        del buf289
        buf294 = reinterpret_tensor(buf286, (2592, 2048), (2048, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf293, (2592, 512), (512, 1), 0), reinterpret_tensor(arg287_1, (512, 2048), (1, 512), 0), out=buf294)
        del arg287_1
        buf295 = reinterpret_tensor(buf294, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [x_758], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf295, arg288_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg288_1
        buf296 = reinterpret_tensor(buf293, (2592, 512), (512, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf295, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg289_1, (2048, 512), (1, 2048), 0), out=buf296)
        del arg289_1
        buf297 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [x_763, x_764], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf297, buf296, arg290_1, arg291_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg290_1
        del arg291_1
        # Topologically Sorted Source Nodes: [x_765], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, arg292_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf298, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg292_1
        buf302 = reinterpret_tensor(buf296, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [x_767], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf298, arg293_1, arg294_1, arg295_1, buf302, 2592, 512, grid=grid(2592), stream=stream0)
        del arg293_1
        del arg294_1
        del arg295_1
        del buf298
        buf303 = reinterpret_tensor(buf295, (2592, 2048), (2048, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2592, 512), (512, 1), 0), reinterpret_tensor(arg296_1, (512, 2048), (1, 512), 0), out=buf303)
        del arg296_1
        buf304 = reinterpret_tensor(buf303, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [x_769], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf304, arg297_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg297_1
        buf305 = reinterpret_tensor(buf302, (2592, 512), (512, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg298_1, (2048, 512), (1, 2048), 0), out=buf305)
        del arg298_1
        buf306 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [x_774, x_775], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_17.run(buf306, buf305, arg299_1, arg300_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg299_1
        del arg300_1
        # Topologically Sorted Source Nodes: [x_776], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf306, arg301_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf307, (8, 512, 18, 18), (165888, 1, 9216, 512))
        del arg301_1
        buf311 = reinterpret_tensor(buf305, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [x_778], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf307, arg302_1, arg303_1, arg304_1, buf311, 2592, 512, grid=grid(2592), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        buf312 = reinterpret_tensor(buf304, (2592, 2048), (2048, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf311, (2592, 512), (512, 1), 0), reinterpret_tensor(arg305_1, (512, 2048), (1, 512), 0), out=buf312)
        del arg305_1
        buf313 = reinterpret_tensor(buf312, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [x_780], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf313, arg306_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg306_1
        buf314 = reinterpret_tensor(buf311, (2592, 512), (512, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg307_1, (2048, 512), (1, 2048), 0), out=buf314)
        del arg307_1
        del buf313
        buf318 = reinterpret_tensor(buf307, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [x_788], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_19.run(buf314, arg308_1, arg309_1, buf306, arg310_1, arg311_1, buf318, 2592, 512, grid=grid(2592), stream=stream0)
        del arg308_1
        del arg309_1
        del arg310_1
        del arg311_1
        del buf306
        del buf314
        buf319 = empty_strided_cuda((1024, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg312_1, buf319, 524288, 4, grid=grid(524288, 4), stream=stream0)
        del arg312_1
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(reinterpret_tensor(buf318, (8, 512, 18, 18), (165888, 1, 9216, 512), 0), buf319, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (8, 1024, 9, 9), (82944, 1, 9216, 1024))
        del buf318
        del buf319
        buf321 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_21.run(buf321, arg313_1, 663552, grid=grid(663552), stream=stream0)
        del arg313_1
        # Topologically Sorted Source Nodes: [x_790], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, arg314_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024, bias=None)
        assert_size_stride(buf322, (8, 1024, 9, 9), (82944, 1, 9216, 1024))
        del arg314_1
        buf326 = empty_strided_cuda((8, 9, 9, 1024), (82944, 9216, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_792], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_22.run(buf322, arg315_1, arg316_1, arg317_1, buf326, 648, 1024, grid=grid(648), stream=stream0)
        del arg315_1
        del arg316_1
        del arg317_1
        del buf322
        buf327 = reinterpret_tensor(buf69, (648, 4096), (4096, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (648, 1024), (1024, 1), 0), reinterpret_tensor(arg318_1, (1024, 4096), (1, 1024), 0), out=buf327)
        del arg318_1
        buf328 = reinterpret_tensor(buf327, (8, 9, 9, 4096), (331776, 36864, 4096, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [x_794], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf328, arg319_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg319_1
        buf329 = reinterpret_tensor(buf326, (648, 1024), (1024, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf328, (648, 4096), (4096, 1), 0), reinterpret_tensor(arg320_1, (4096, 1024), (1, 4096), 0), out=buf329)
        del arg320_1
        buf330 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [x_799, x_800], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_24.run(buf330, buf329, arg321_1, arg322_1, 663552, grid=grid(663552), stream=stream0)
        del arg321_1
        del arg322_1
        # Topologically Sorted Source Nodes: [x_801], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, arg323_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024, bias=None)
        assert_size_stride(buf331, (8, 1024, 9, 9), (82944, 1, 9216, 1024))
        del arg323_1
        buf335 = reinterpret_tensor(buf329, (8, 9, 9, 1024), (82944, 9216, 1024, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [x_803], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_22.run(buf331, arg324_1, arg325_1, arg326_1, buf335, 648, 1024, grid=grid(648), stream=stream0)
        del arg324_1
        del arg325_1
        del arg326_1
        del buf331
        buf336 = reinterpret_tensor(buf328, (648, 4096), (4096, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (648, 1024), (1024, 1), 0), reinterpret_tensor(arg327_1, (1024, 4096), (1, 1024), 0), out=buf336)
        del arg327_1
        buf337 = reinterpret_tensor(buf336, (8, 9, 9, 4096), (331776, 36864, 4096, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [x_805], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf337, arg328_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg328_1
        buf338 = reinterpret_tensor(buf335, (648, 1024), (1024, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf337, (648, 4096), (4096, 1), 0), reinterpret_tensor(arg329_1, (4096, 1024), (1, 4096), 0), out=buf338)
        del arg329_1
        buf339 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [x_810, x_811], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_24.run(buf339, buf338, arg330_1, arg331_1, 663552, grid=grid(663552), stream=stream0)
        del arg330_1
        del arg331_1
        # Topologically Sorted Source Nodes: [x_812], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, arg332_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024, bias=None)
        assert_size_stride(buf340, (8, 1024, 9, 9), (82944, 1, 9216, 1024))
        del arg332_1
        buf344 = reinterpret_tensor(buf338, (8, 9, 9, 1024), (82944, 9216, 1024, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [x_814], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_22.run(buf340, arg333_1, arg334_1, arg335_1, buf344, 648, 1024, grid=grid(648), stream=stream0)
        del arg333_1
        del arg334_1
        del arg335_1
        del buf340
        buf345 = reinterpret_tensor(buf337, (648, 4096), (4096, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf344, (648, 1024), (1024, 1), 0), reinterpret_tensor(arg336_1, (1024, 4096), (1, 1024), 0), out=buf345)
        del arg336_1
        buf346 = reinterpret_tensor(buf345, (8, 9, 9, 4096), (331776, 36864, 4096, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [x_816], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf346, arg337_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg337_1
        buf347 = reinterpret_tensor(buf344, (648, 1024), (1024, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (648, 4096), (4096, 1), 0), reinterpret_tensor(arg338_1, (4096, 1024), (1, 4096), 0), out=buf347)
        del arg338_1
        del buf346
        buf349 = empty_strided_cuda((8, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_821, x_822, x_823], Original ATen: [aten.mul, aten.add, aten.mean]
        triton_red_fused_add_mean_mul_25.run(buf347, arg339_1, arg340_1, buf339, buf349, 8192, 81, grid=grid(8192), stream=stream0)
        del arg339_1
        del arg340_1
        del buf339
        del buf347
        buf353 = empty_strided_cuda((8, 1, 1, 1024), (1024, 1, 8192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_825], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf349, arg341_1, arg342_1, buf353, 8, 1024, grid=grid(8), stream=stream0)
        del arg341_1
        del arg342_1
        del buf349
        buf354 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_829], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg344_1, reinterpret_tensor(buf353, (8, 1024), (1024, 1), 0), reinterpret_tensor(arg343_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf354)
        del arg343_1
        del arg344_1
        del buf353
    return (buf354, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((8, 3, 288, 288), (248832, 82944, 288, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, 256, 2, 2), (1024, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, 512, 2, 2), (2048, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convnext_base', benchmark_compiled_module)
