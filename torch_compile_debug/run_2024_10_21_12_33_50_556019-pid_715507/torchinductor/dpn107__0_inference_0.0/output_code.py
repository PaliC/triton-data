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
# Topologically Sorted Source Nodes: [x_227], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_227 => convolution_111
# Graph fragment:
#   %convolution_111 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/6w/c6wq3upchze2pmlksr3iofdnyslqkkhurovwyqukrvey7tukuoj2.py
# Topologically Sorted Source Nodes: [x_227], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_227 => convolution_111
# Graph fragment:
#   %convolution_111 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
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
    ynumel = 384
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


# kernel path: /tmp/torchinductor_sahanp/cf/ccfpmvbrr3jr6di5h57mxjs6lfvqhbrplij3pza6mx2z36jf5isg.py
# Topologically Sorted Source Nodes: [x_228, x_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_228 => add_258, mul_334, mul_335, sub_111
#   x_229 => relu_111
# Graph fragment:
#   %sub_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_111, %unsqueeze_889), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_111, %unsqueeze_891), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_334, %unsqueeze_893), kwargs = {})
#   %add_258 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_335, %unsqueeze_895), kwargs = {})
#   %relu_111 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_258,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dn/cdninzmwehxzf6gw6qndtu27kxcxnbldzsbyioenjkc5k3uhvz5n.py
# Topologically Sorted Source Nodes: [x_228, x_229, input_2, x_230, x_231, x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_2 => _low_memory_max_pool2d_with_offsets_1
#   x_228 => add_258, mul_334, mul_335, sub_111
#   x_229 => relu_111
#   x_230 => add_260, mul_337, mul_338, sub_112
#   x_231 => relu_112
#   x_232 => add_262, mul_340, mul_341, sub_113
#   x_233 => relu_113
# Graph fragment:
#   %sub_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_111, %unsqueeze_889), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_111, %unsqueeze_891), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_334, %unsqueeze_893), kwargs = {})
#   %add_258 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_335, %unsqueeze_895), kwargs = {})
#   %relu_111 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_258,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_111, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %sub_112 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_2, %unsqueeze_897), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_112, %unsqueeze_899), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_337, %unsqueeze_901), kwargs = {})
#   %add_260 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_338, %unsqueeze_903), kwargs = {})
#   %relu_112 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_260,), kwargs = {})
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_2, %unsqueeze_905), kwargs = {})
#   %mul_340 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %unsqueeze_907), kwargs = {})
#   %mul_341 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_340, %unsqueeze_909), kwargs = {})
#   %add_262 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_341, %unsqueeze_911), kwargs = {})
#   %relu_113 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_262,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 7168) % 56
    x1 = (xindex // 128) % 56
    x0 = xindex % 128
    x6 = (xindex // 7168)
    x7 = xindex
    tmp52 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-14464) + x0 + (256*x1) + (28672*x6)), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-14336) + x0 + (256*x1) + (28672*x6)), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-14208) + x0 + (256*x1) + (28672*x6)), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-128) + x0 + (256*x1) + (28672*x6)), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + (256*x1) + (28672*x6)), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (128 + x0 + (256*x1) + (28672*x6)), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + (2*x2)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (14208 + x0 + (256*x1) + (28672*x6)), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (14336 + x0 + (256*x1) + (28672*x6)), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (14464 + x0 + (256*x1) + (28672*x6)), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp53 = tmp51 - tmp52
    tmp55 = 0.001
    tmp56 = tmp54 + tmp55
    tmp57 = libdevice.sqrt(tmp56)
    tmp58 = tl.full([1], 1, tl.int32)
    tmp59 = tmp58 / tmp57
    tmp60 = 1.0
    tmp61 = tmp59 * tmp60
    tmp62 = tmp53 * tmp61
    tmp64 = tmp62 * tmp63
    tmp66 = tmp64 + tmp65
    tmp67 = tl.full([1], 0, tl.int32)
    tmp68 = triton_helpers.maximum(tmp67, tmp66)
    tmp70 = tmp51 - tmp69
    tmp72 = tmp71 + tmp55
    tmp73 = libdevice.sqrt(tmp72)
    tmp74 = tmp58 / tmp73
    tmp75 = tmp74 * tmp60
    tmp76 = tmp70 * tmp75
    tmp78 = tmp76 * tmp77
    tmp80 = tmp78 + tmp79
    tmp81 = triton_helpers.maximum(tmp67, tmp80)
    tl.store(out_ptr1 + (x7), tmp68, None)
    tl.store(out_ptr2 + (x7), tmp81, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6c/c6ccimrnosro5gkj462lfwv7x2rh4u4rbqftgfooijddxvgcefk2.py
# Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_234 => add_264, mul_343, mul_344, sub_114
#   x_235 => relu_114
# Graph fragment:
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_113, %unsqueeze_913), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_915), kwargs = {})
#   %mul_344 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_343, %unsqueeze_917), kwargs = {})
#   %add_264 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_344, %unsqueeze_919), kwargs = {})
#   %relu_114 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_264,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5017600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 200
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/j3/cj33cyyf265un627yar7yjtl53qkz5su4qjf6gidvijo47xfgoqz.py
# Topologically Sorted Source Nodes: [x_234, x_235, x_in_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_234 => add_264, mul_343, mul_344, sub_114
#   x_235 => relu_114
#   x_in_140 => convolution_114
# Graph fragment:
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_113, %unsqueeze_913), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_915), kwargs = {})
#   %mul_344 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_343, %unsqueeze_917), kwargs = {})
#   %add_264 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_344, %unsqueeze_919), kwargs = {})
#   %relu_114 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_264,), kwargs = {})
#   %convolution_114 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_114, %arg20_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 800
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 4
    y1 = (yindex // 4)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (4*x2) + (36*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/z5/cz5ucpn6tiaffyr5s6js3levl3ojhznpfd5w477fs77dmnmacste.py
# Topologically Sorted Source Nodes: [x_in_142, x_238, x_239], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_238 => add_269, mul_349, mul_350, sub_116
#   x_239 => relu_116
#   x_in_142 => cat_71
# Graph fragment:
#   %cat_71 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_267, %cat_70], 1), kwargs = {})
#   %sub_116 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_71, %unsqueeze_929), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_116, %unsqueeze_931), kwargs = {})
#   %mul_350 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_349, %unsqueeze_933), kwargs = {})
#   %add_269 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_350, %unsqueeze_935), kwargs = {})
#   %relu_116 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_269,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2528
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 316
    x2 = xindex
    y1 = (yindex // 316)
    y3 = yindex
    tmp28 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((296*x2) + (928256*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((276*x2) + (865536*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 316, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.broadcast_to((-256) + y0, [XBLOCK, YBLOCK])
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1, 1], 40, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr0 + (256 + (296*x2) + (928256*y1) + ((-256) + y0)), tmp17 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp13 >= tmp15
    tmp20 = tl.full([1, 1], 60, tl.int64)
    tmp21 = tmp13 < tmp20
    tmp22 = tmp19 & tmp10
    tmp23 = tl.load(in_ptr1 + (256 + (276*x2) + (865536*y1) + ((-40) + ((-256) + y0))), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp16, tmp18, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp9, tmp26)
    tmp29 = tmp27 - tmp28
    tmp31 = 0.001
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tl.full([1, 1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full([1, 1], 0, tl.int32)
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tl.store(out_ptr1 + (y0 + (316*x2) + (990976*y1)), tmp44, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yf/cyfjhpg3wdn5xxdat4zwwvoapqrefuzlr65gyowcvuumhptt6bog.py
# Topologically Sorted Source Nodes: [x_in_146, x_244, x_245], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_244 => add_276, mul_358, mul_359, sub_119
#   x_245 => relu_119
#   x_in_146 => cat_73
# Graph fragment:
#   %cat_73 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_274, %cat_72], 1), kwargs = {})
#   %sub_119 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_73, %unsqueeze_953), kwargs = {})
#   %mul_358 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_119, %unsqueeze_955), kwargs = {})
#   %mul_359 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_358, %unsqueeze_957), kwargs = {})
#   %add_276 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_359, %unsqueeze_959), kwargs = {})
#   %relu_119 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_276,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 336
    x2 = xindex
    y1 = (yindex // 336)
    y3 = yindex
    tmp39 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((296*x2) + (928256*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((276*x2) + (865536*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((276*x2) + (865536*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 336, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.broadcast_to((-256) + y0, [XBLOCK, YBLOCK])
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1, 1], 60, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1, 1], 40, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + (256 + (296*x2) + (928256*y1) + ((-256) + y0)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp15 >= tmp20
    tmp25 = tmp24 & tmp19
    tmp26 = tl.load(in_ptr1 + (256 + (276*x2) + (865536*y1) + ((-40) + ((-256) + y0))), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp21, tmp23, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp19, tmp27, tmp28)
    tmp30 = tmp15 >= tmp17
    tmp31 = tl.full([1, 1], 80, tl.int64)
    tmp32 = tmp15 < tmp31
    tmp33 = tmp30 & tmp12
    tmp34 = tl.load(in_ptr2 + (256 + (276*x2) + (865536*y1) + ((-60) + ((-256) + y0))), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp18, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp12, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp11, tmp37)
    tmp40 = tmp38 - tmp39
    tmp42 = 0.001
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.sqrt(tmp43)
    tmp45 = tl.full([1, 1], 1, tl.int32)
    tmp46 = tmp45 / tmp44
    tmp47 = 1.0
    tmp48 = tmp46 * tmp47
    tmp49 = tmp40 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tmp54 = tl.full([1, 1], 0, tl.int32)
    tmp55 = triton_helpers.maximum(tmp54, tmp53)
    tl.store(out_ptr1 + (y0 + (336*x2) + (1053696*y1)), tmp55, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yw/cywt72pzj5vasccog3gygxbdr5bvigi2ubog6ia6lqmoapqsbxbi.py
# Topologically Sorted Source Nodes: [dense_37], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_37 => cat_74
# Graph fragment:
#   %cat_74 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_72, %slice_342], 1), kwargs = {})
triton_poi_fused_cat_8 = async_compile.triton('triton_poi_fused_cat_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 100
    x0 = xindex % 3136
    x2 = (xindex // 313600)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 60, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 40, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (256 + (296*x0) + (928256*x2) + x1), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp0 >= tmp8
    tmp13 = tmp12 & tmp7
    tmp14 = tl.load(in_ptr1 + (256 + (276*x0) + (865536*x2) + ((-40) + x1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp11, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp7, tmp15, tmp16)
    tmp18 = tmp0 >= tmp5
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + (256 + (276*x0) + (865536*x2) + ((-60) + x1)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp6, tmp17, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 100, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (256 + (276*x0) + (865536*x2) + ((-80) + x1)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2p/c2pb5q7alm3hekbpzn23webqzwfbbov62j3vrkicxqeefknidur2.py
# Topologically Sorted Source Nodes: [x_in_150, x_250, x_251], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_250 => add_283, mul_367, mul_368, sub_122
#   x_251 => relu_122
#   x_in_150 => cat_75
# Graph fragment:
#   %cat_75 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_281, %cat_74], 1), kwargs = {})
#   %sub_122 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_75, %unsqueeze_977), kwargs = {})
#   %mul_367 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_122, %unsqueeze_979), kwargs = {})
#   %mul_368 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_367, %unsqueeze_981), kwargs = {})
#   %add_283 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_368, %unsqueeze_983), kwargs = {})
#   %relu_122 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_283,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2848
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 356
    x2 = xindex
    y1 = (yindex // 356)
    y3 = yindex
    tmp19 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((296*x2) + (928256*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((276*x2) + (865536*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((276*x2) + (865536*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((276*x2) + (865536*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 356, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr4 + (x2 + (3136*((-256) + y0)) + (313600*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.where(tmp4, tmp13, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 0.001
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tl.full([1, 1], 1, tl.int32)
    tmp26 = tmp25 / tmp24
    tmp27 = 1.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp20 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1, 1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(out_ptr1 + (y0 + (356*x2) + (1116416*y1)), tmp35, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fi/cfir42dkmgm7msaezlmfxlx6wktq72vdcjzqhgyql4otlzuv7cnu.py
# Topologically Sorted Source Nodes: [x_in_154, x_256, x_257, x_258, x_259], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_256 => add_290, mul_376, mul_377, sub_125
#   x_257 => relu_125
#   x_258 => add_292, mul_379, mul_380, sub_126
#   x_259 => relu_126
#   x_in_154 => cat_77
# Graph fragment:
#   %cat_77 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_288, %cat_76], 1), kwargs = {})
#   %sub_125 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_77, %unsqueeze_1001), kwargs = {})
#   %mul_376 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_125, %unsqueeze_1003), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_376, %unsqueeze_1005), kwargs = {})
#   %add_290 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_377, %unsqueeze_1007), kwargs = {})
#   %relu_125 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_290,), kwargs = {})
#   %sub_126 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_77, %unsqueeze_1009), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_126, %unsqueeze_1011), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_379, %unsqueeze_1013), kwargs = {})
#   %add_292 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_380, %unsqueeze_1015), kwargs = {})
#   %relu_126 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_292,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3008
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 376
    x2 = xindex
    y1 = (yindex // 376)
    y3 = yindex
    tmp34 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr10 + (y0), ymask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr11 + (y0), ymask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr12 + (y0), ymask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr13 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((296*x2) + (928256*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((276*x2) + (865536*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((276*x2) + (865536*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((276*x2) + (865536*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr4 + ((276*x2) + (865536*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1, 1], 376, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.broadcast_to((-256) + y0, [XBLOCK, YBLOCK])
    tmp20 = tmp19 >= tmp1
    tmp21 = tl.full([1, 1], 100, tl.int64)
    tmp22 = tmp19 < tmp21
    tmp23 = tmp22 & tmp16
    tmp24 = tl.load(in_ptr5 + (x2 + (3136*((-256) + y0)) + (313600*y1)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp19 >= tmp21
    tmp26 = tl.full([1, 1], 120, tl.int64)
    tmp27 = tmp19 < tmp26
    tmp28 = tmp25 & tmp16
    tmp29 = tl.load(in_ptr4 + (256 + (276*x2) + (865536*y1) + ((-100) + ((-256) + y0))), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.where(tmp22, tmp24, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp16, tmp30, tmp31)
    tmp33 = tl.where(tmp4, tmp15, tmp32)
    tmp35 = tmp33 - tmp34
    tmp37 = 0.001
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.sqrt(tmp38)
    tmp40 = tl.full([1, 1], 1, tl.int32)
    tmp41 = tmp40 / tmp39
    tmp42 = 1.0
    tmp43 = tmp41 * tmp42
    tmp44 = tmp35 * tmp43
    tmp46 = tmp44 * tmp45
    tmp48 = tmp46 + tmp47
    tmp49 = tl.full([1, 1], 0, tl.int32)
    tmp50 = triton_helpers.maximum(tmp49, tmp48)
    tmp52 = tmp33 - tmp51
    tmp54 = tmp53 + tmp37
    tmp55 = libdevice.sqrt(tmp54)
    tmp56 = tmp40 / tmp55
    tmp57 = tmp56 * tmp42
    tmp58 = tmp52 * tmp57
    tmp60 = tmp58 * tmp59
    tmp62 = tmp60 + tmp61
    tmp63 = triton_helpers.maximum(tmp49, tmp62)
    tl.store(out_ptr1 + (y0 + (376*x2) + (1179136*y1)), tmp50, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (376*x2) + (1179136*y1)), tmp63, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fa/cfa4ljmoxstmm7snpwp6wcin44qonfd7ucybqf4ryx6ilvuiuyin.py
# Topologically Sorted Source Nodes: [x_260, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_260 => add_294, mul_382, mul_383, sub_127
#   x_261 => relu_127
# Graph fragment:
#   %sub_127 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_126, %unsqueeze_1017), kwargs = {})
#   %mul_382 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_127, %unsqueeze_1019), kwargs = {})
#   %mul_383 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_382, %unsqueeze_1021), kwargs = {})
#   %add_294 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_383, %unsqueeze_1023), kwargs = {})
#   %relu_127 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_294,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10035200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 400
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n7/cn7exoiqsabaznhrbxj2xi6ydnvgc4t56euyzbgp6d5ahx46lvww.py
# Topologically Sorted Source Nodes: [x_260, x_261, x_in_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_260 => add_294, mul_382, mul_383, sub_127
#   x_261 => relu_127
#   x_in_156 => convolution_127
# Graph fragment:
#   %sub_127 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_126, %unsqueeze_1017), kwargs = {})
#   %mul_382 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_127, %unsqueeze_1019), kwargs = {})
#   %mul_383 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_382, %unsqueeze_1021), kwargs = {})
#   %add_294 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_383, %unsqueeze_1023), kwargs = {})
#   %relu_127 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_294,), kwargs = {})
#   %convolution_127 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_127, %arg85_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 50), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3200
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 8
    y1 = (yindex // 8)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (8*x2) + (72*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oh/cohmvk7k65ahwbe6b6me3gtw3gxmguk4ad4ptvbglmxpz27a5jeu.py
# Topologically Sorted Source Nodes: [x_262, x_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_262 => add_296, mul_385, mul_386, sub_128
#   x_263 => relu_128
# Graph fragment:
#   %sub_128 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_127, %unsqueeze_1025), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_128, %unsqueeze_1027), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_385, %unsqueeze_1029), kwargs = {})
#   %add_296 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_386, %unsqueeze_1031), kwargs = {})
#   %relu_128 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_296,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 400
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xv/cxv2qog7kgzbt5ke53ts36dyuqod5bxfjrtveh3auy272bbmjqd3.py
# Topologically Sorted Source Nodes: [x_in_158, x_264, x_265], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_264 => add_299, mul_388, mul_389, sub_129
#   x_265 => relu_129
#   x_in_158 => cat_79
# Graph fragment:
#   %cat_79 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_297, %cat_78], 1), kwargs = {})
#   %sub_129 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_79, %unsqueeze_1033), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %unsqueeze_1035), kwargs = {})
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_388, %unsqueeze_1037), kwargs = {})
#   %add_299 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_389, %unsqueeze_1039), kwargs = {})
#   %relu_129 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_299,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5632
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 704
    x2 = xindex
    y1 = (yindex // 704)
    y3 = yindex
    tmp28 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((640*x2) + (501760*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 704, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.broadcast_to((-512) + y0, [XBLOCK, YBLOCK])
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1, 1], 128, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr0 + (512 + (640*x2) + (501760*y1) + ((-512) + y0)), tmp17 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp13 >= tmp15
    tmp20 = tl.full([1, 1], 192, tl.int64)
    tmp21 = tmp13 < tmp20
    tmp22 = tmp19 & tmp10
    tmp23 = tl.load(in_ptr1 + (512 + (576*x2) + (451584*y1) + ((-128) + ((-512) + y0))), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp16, tmp18, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp9, tmp26)
    tmp29 = tmp27 - tmp28
    tmp31 = 0.001
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tl.full([1, 1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full([1, 1], 0, tl.int32)
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tl.store(out_ptr1 + (y0 + (704*x2) + (551936*y1)), tmp44, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xg/cxg73wf2vrsdw2yljs4aymlgoe3okkgj3sxxoljjp33cxgl66gri.py
# Topologically Sorted Source Nodes: [x_in_162, x_270, x_271], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_270 => add_306, mul_397, mul_398, sub_132
#   x_271 => relu_132
#   x_in_162 => cat_81
# Graph fragment:
#   %cat_81 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_304, %cat_80], 1), kwargs = {})
#   %sub_132 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_81, %unsqueeze_1057), kwargs = {})
#   %mul_397 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_132, %unsqueeze_1059), kwargs = {})
#   %mul_398 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_397, %unsqueeze_1061), kwargs = {})
#   %add_306 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_398, %unsqueeze_1063), kwargs = {})
#   %relu_132 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_306,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 768
    x2 = xindex
    y1 = (yindex // 768)
    y3 = yindex
    tmp39 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((640*x2) + (501760*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 768, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.broadcast_to((-512) + y0, [XBLOCK, YBLOCK])
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1, 1], 192, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1, 1], 128, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + (512 + (640*x2) + (501760*y1) + ((-512) + y0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp15 >= tmp20
    tmp25 = tmp24 & tmp19
    tmp26 = tl.load(in_ptr1 + (512 + (576*x2) + (451584*y1) + ((-128) + ((-512) + y0))), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp21, tmp23, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp19, tmp27, tmp28)
    tmp30 = tmp15 >= tmp17
    tmp31 = tl.full([1, 1], 256, tl.int64)
    tmp32 = tmp15 < tmp31
    tmp33 = tmp30 & tmp12
    tmp34 = tl.load(in_ptr2 + (512 + (576*x2) + (451584*y1) + ((-192) + ((-512) + y0))), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp18, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp12, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp11, tmp37)
    tmp40 = tmp38 - tmp39
    tmp42 = 0.001
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.sqrt(tmp43)
    tmp45 = tl.full([1, 1], 1, tl.int32)
    tmp46 = tmp45 / tmp44
    tmp47 = 1.0
    tmp48 = tmp46 * tmp47
    tmp49 = tmp40 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tmp54 = tl.full([1, 1], 0, tl.int32)
    tmp55 = triton_helpers.maximum(tmp54, tmp53)
    tl.store(out_ptr1 + (y0 + (768*x2) + (602112*y1)), tmp55, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/26/c26o4i6iw5l2z27wwxoinxo53ohebn5hg2pt6bkfamyp4hweqkrq.py
# Topologically Sorted Source Nodes: [dense_41], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_41 => cat_82
# Graph fragment:
#   %cat_82 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_80, %slice_382], 1), kwargs = {})
triton_poi_fused_cat_16 = async_compile.triton('triton_poi_fused_cat_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 784) % 320
    x0 = xindex % 784
    x2 = (xindex // 250880)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 192, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 128, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (512 + (640*x0) + (501760*x2) + x1), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp0 >= tmp8
    tmp13 = tmp12 & tmp7
    tmp14 = tl.load(in_ptr1 + (512 + (576*x0) + (451584*x2) + ((-128) + x1)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp11, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp7, tmp15, tmp16)
    tmp18 = tmp0 >= tmp5
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + (512 + (576*x0) + (451584*x2) + ((-192) + x1)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp6, tmp17, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 320, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (512 + (576*x0) + (451584*x2) + ((-256) + x1)), tmp24, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5m/c5mrxa3zinvhovsppjjf5tgxvgbtchgmfqw4e56mxqjhttoy6ckm.py
# Topologically Sorted Source Nodes: [x_in_166, x_276, x_277], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_276 => add_313, mul_406, mul_407, sub_135
#   x_277 => relu_135
#   x_in_166 => cat_83
# Graph fragment:
#   %cat_83 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_311, %cat_82], 1), kwargs = {})
#   %sub_135 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_83, %unsqueeze_1081), kwargs = {})
#   %mul_406 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_135, %unsqueeze_1083), kwargs = {})
#   %mul_407 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_406, %unsqueeze_1085), kwargs = {})
#   %add_313 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_407, %unsqueeze_1087), kwargs = {})
#   %relu_135 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_313,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6656
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 832
    x2 = xindex
    y1 = (yindex // 832)
    y3 = yindex
    tmp19 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((640*x2) + (501760*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 832, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr4 + (x2 + (784*((-512) + y0)) + (250880*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.where(tmp4, tmp13, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 0.001
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tl.full([1, 1], 1, tl.int32)
    tmp26 = tmp25 / tmp24
    tmp27 = 1.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp20 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1, 1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(out_ptr1 + (y0 + (832*x2) + (652288*y1)), tmp35, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/of/cofmpbq22uefg6nfpwqbinq3ouynqerowiohln7y7v7ba5u3ptx6.py
# Topologically Sorted Source Nodes: [resid_39, resid_40, resid_41, resid_42], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_39 => add_297
#   resid_40 => add_304
#   resid_41 => add_311
#   resid_42 => add_318
# Graph fragment:
#   %add_297 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_354, %slice_362), kwargs = {})
#   %add_304 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_297, %slice_370), kwargs = {})
#   %add_311 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_304, %slice_378), kwargs = {})
#   %add_318 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_311, %slice_386), kwargs = {})
triton_poi_fused_add_18 = async_compile.triton('triton_poi_fused_add_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (640*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (576*x1)), None)
    tmp3 = tl.load(in_ptr2 + (x0 + (576*x1)), None)
    tmp5 = tl.load(in_ptr3 + (x0 + (576*x1)), None)
    tmp7 = tl.load(in_ptr4 + (x0 + (576*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3x/c3x4ikyng2askdicjipu6pxbqn7bd4ongho4ot4jwslez6rtwkkt.py
# Topologically Sorted Source Nodes: [x_in_170, x_282, x_283], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_282 => add_320, mul_415, mul_416, sub_138
#   x_283 => relu_138
#   x_in_170 => cat_85
# Graph fragment:
#   %cat_85 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_318, %cat_84], 1), kwargs = {})
#   %sub_138 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_85, %unsqueeze_1105), kwargs = {})
#   %mul_415 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_138, %unsqueeze_1107), kwargs = {})
#   %mul_416 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_415, %unsqueeze_1109), kwargs = {})
#   %add_320 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_416, %unsqueeze_1111), kwargs = {})
#   %relu_138 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_320,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7168
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 896
    x2 = xindex
    y1 = (yindex // 896)
    y3 = yindex
    tmp24 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((512*x2) + (401408*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 896, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.broadcast_to((-512) + y0, [XBLOCK, YBLOCK])
    tmp10 = tmp9 >= tmp1
    tmp11 = tl.full([1, 1], 320, tl.int64)
    tmp12 = tmp9 < tmp11
    tmp13 = tmp12 & tmp6
    tmp14 = tl.load(in_ptr1 + (x2 + (784*((-512) + y0)) + (250880*y1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp9 >= tmp11
    tmp16 = tl.full([1, 1], 384, tl.int64)
    tmp17 = tmp9 < tmp16
    tmp18 = tmp15 & tmp6
    tmp19 = tl.load(in_ptr2 + (512 + (576*x2) + (451584*y1) + ((-320) + ((-512) + y0))), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp12, tmp14, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp6, tmp20, tmp21)
    tmp23 = tl.where(tmp4, tmp5, tmp22)
    tmp25 = tmp23 - tmp24
    tmp27 = 0.001
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tl.full([1, 1], 1, tl.int32)
    tmp31 = tmp30 / tmp29
    tmp32 = 1.0
    tmp33 = tmp31 * tmp32
    tmp34 = tmp25 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tl.full([1, 1], 0, tl.int32)
    tmp40 = triton_helpers.maximum(tmp39, tmp38)
    tl.store(out_ptr1 + (y0 + (896*x2) + (702464*y1)), tmp40, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/a2/ca2tq3a4abje4dugg4fjwzuf33ynss32vi5pv6p4ie6mg5cjpdtc.py
# Topologically Sorted Source Nodes: [x_in_174, x_288, x_289], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_288 => add_327, mul_424, mul_425, sub_141
#   x_289 => relu_141
#   x_in_174 => cat_87
# Graph fragment:
#   %cat_87 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_325, %cat_86], 1), kwargs = {})
#   %sub_141 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_87, %unsqueeze_1129), kwargs = {})
#   %mul_424 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_141, %unsqueeze_1131), kwargs = {})
#   %mul_425 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_424, %unsqueeze_1133), kwargs = {})
#   %add_327 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_425, %unsqueeze_1135), kwargs = {})
#   %relu_141 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_327,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 960
    x2 = xindex
    y1 = (yindex // 960)
    y3 = yindex
    tmp37 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((512*x2) + (401408*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 960, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.broadcast_to((-512) + y0, [XBLOCK, YBLOCK])
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1, 1], 384, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.full([1, 1], 320, tl.int64)
    tmp19 = tmp13 < tmp18
    tmp20 = tmp19 & tmp17
    tmp21 = tl.load(in_ptr2 + (x2 + (784*((-512) + y0)) + (250880*y1)), tmp20 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp13 >= tmp18
    tmp23 = tmp22 & tmp17
    tmp24 = tl.load(in_ptr3 + (512 + (576*x2) + (451584*y1) + ((-320) + ((-512) + y0))), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.where(tmp19, tmp21, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp17, tmp25, tmp26)
    tmp28 = tmp13 >= tmp15
    tmp29 = tl.full([1, 1], 448, tl.int64)
    tmp30 = tmp13 < tmp29
    tmp31 = tmp28 & tmp10
    tmp32 = tl.load(in_ptr1 + (512 + (576*x2) + (451584*y1) + ((-384) + ((-512) + y0))), tmp31 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp16, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp10, tmp33, tmp34)
    tmp36 = tl.where(tmp4, tmp9, tmp35)
    tmp38 = tmp36 - tmp37
    tmp40 = 0.001
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.sqrt(tmp41)
    tmp43 = tl.full([1, 1], 1, tl.int32)
    tmp44 = tmp43 / tmp42
    tmp45 = 1.0
    tmp46 = tmp44 * tmp45
    tmp47 = tmp38 * tmp46
    tmp49 = tmp47 * tmp48
    tmp51 = tmp49 + tmp50
    tmp52 = tl.full([1, 1], 0, tl.int32)
    tmp53 = triton_helpers.maximum(tmp52, tmp51)
    tl.store(out_ptr1 + (y0 + (960*x2) + (752640*y1)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3u/c3u3ilm7ms6s62whifs5fqdhpfww3y6tqybydnqzkkqnugaz7pkd.py
# Topologically Sorted Source Nodes: [dense_44], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_44 => cat_88
# Graph fragment:
#   %cat_88 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_86, %slice_406], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 784) % 512
    x0 = xindex % 784
    x2 = (xindex // 401408)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 448, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 384, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 320, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (x0 + (784*x1) + (250880*x2)), tmp10, other=0.0)
    tmp12 = tmp0 >= tmp8
    tmp13 = tmp12 & tmp7
    tmp14 = tl.load(in_ptr1 + (512 + (576*x0) + (451584*x2) + ((-320) + x1)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp11, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp7, tmp15, tmp16)
    tmp18 = tmp0 >= tmp5
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + (512 + (576*x0) + (451584*x2) + ((-384) + x1)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp6, tmp17, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 512, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (512 + (576*x0) + (451584*x2) + ((-448) + x1)), tmp24, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lc/clcayds5zg4zk5qu6puuu2ofjgoxnt4uc5t6lgdtlfqb5lqf4cic.py
# Topologically Sorted Source Nodes: [x_in_178, x_294, x_295], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_294 => add_334, mul_433, mul_434, sub_144
#   x_295 => relu_144
#   x_in_178 => cat_89
# Graph fragment:
#   %cat_89 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_332, %cat_88], 1), kwargs = {})
#   %sub_144 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_89, %unsqueeze_1153), kwargs = {})
#   %mul_433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_144, %unsqueeze_1155), kwargs = {})
#   %mul_434 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_433, %unsqueeze_1157), kwargs = {})
#   %add_334 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_434, %unsqueeze_1159), kwargs = {})
#   %relu_144 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_334,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1024
    x2 = xindex
    y1 = (yindex // 1024)
    y3 = yindex
    tmp17 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((512*x2) + (401408*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 1024, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr3 + (x2 + (784*((-512) + y0)) + (401408*y1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp4, tmp11, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 0.001
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1, 1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1, 1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr1 + (y0 + (1024*x2) + (802816*y1)), tmp33, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xf/cxf234hxescg5oun3i7ondztv3u3kaluiet7lqimjkyzyfghzszu.py
# Topologically Sorted Source Nodes: [x_in_182, x_300, x_301], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_300 => add_341, mul_442, mul_443, sub_147
#   x_301 => relu_147
#   x_in_182 => cat_91
# Graph fragment:
#   %cat_91 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_339, %cat_90], 1), kwargs = {})
#   %sub_147 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_91, %unsqueeze_1177), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_147, %unsqueeze_1179), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_442, %unsqueeze_1181), kwargs = {})
#   %add_341 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_443, %unsqueeze_1183), kwargs = {})
#   %relu_147 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_341,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8704
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1088
    x2 = xindex
    y1 = (yindex // 1088)
    y3 = yindex
    tmp31 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((512*x2) + (401408*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 1088, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.broadcast_to((-512) + y0, [XBLOCK, YBLOCK])
    tmp18 = tmp17 >= tmp1
    tmp19 = tmp17 < tmp3
    tmp20 = tmp19 & tmp14
    tmp21 = tl.load(in_ptr4 + (x2 + (784*((-512) + y0)) + (401408*y1)), tmp20 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp17 >= tmp3
    tmp23 = tl.full([1, 1], 576, tl.int64)
    tmp24 = tmp17 < tmp23
    tmp25 = tmp22 & tmp14
    tmp26 = tl.load(in_ptr3 + (512 + (576*x2) + (451584*y1) + ((-512) + ((-512) + y0))), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp19, tmp21, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp14, tmp27, tmp28)
    tmp30 = tl.where(tmp4, tmp13, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 0.001
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.sqrt(tmp35)
    tmp37 = tl.full([1, 1], 1, tl.int32)
    tmp38 = tmp37 / tmp36
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp32 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tl.full([1, 1], 0, tl.int32)
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tl.store(out_ptr1 + (y0 + (1088*x2) + (852992*y1)), tmp47, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bo/cbon36p5fmifzojbvoew6gprrdjefbj32kvgsfkqa5eyouio7a3u.py
# Topologically Sorted Source Nodes: [x_in_186, x_306, x_307, x_308, x_309], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_306 => add_348, mul_451, mul_452, sub_150
#   x_307 => relu_150
#   x_308 => add_350, mul_454, mul_455, sub_151
#   x_309 => relu_151
#   x_in_186 => cat_93
# Graph fragment:
#   %cat_93 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_346, %cat_92], 1), kwargs = {})
#   %sub_150 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_93, %unsqueeze_1201), kwargs = {})
#   %mul_451 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_150, %unsqueeze_1203), kwargs = {})
#   %mul_452 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_451, %unsqueeze_1205), kwargs = {})
#   %add_348 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_452, %unsqueeze_1207), kwargs = {})
#   %relu_150 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_348,), kwargs = {})
#   %sub_151 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_93, %unsqueeze_1209), kwargs = {})
#   %mul_454 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_151, %unsqueeze_1211), kwargs = {})
#   %mul_455 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_454, %unsqueeze_1213), kwargs = {})
#   %add_350 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_455, %unsqueeze_1215), kwargs = {})
#   %relu_151 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_350,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1152
    x2 = xindex
    y1 = (yindex // 1152)
    y3 = yindex
    tmp42 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr10 + (y0), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr11 + (y0), None, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr12 + (y0), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr13 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((512*x2) + (401408*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr4 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1, 1], 1152, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.broadcast_to((-512) + y0, [XBLOCK, YBLOCK])
    tmp20 = tmp19 >= tmp1
    tmp21 = tl.full([1, 1], 576, tl.int64)
    tmp22 = tmp19 < tmp21
    tmp23 = tmp22 & tmp16
    tmp24 = tmp19 < tmp3
    tmp25 = tmp24 & tmp23
    tmp26 = tl.load(in_ptr5 + (x2 + (784*((-512) + y0)) + (401408*y1)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp19 >= tmp3
    tmp28 = tmp27 & tmp23
    tmp29 = tl.load(in_ptr3 + (512 + (576*x2) + (451584*y1) + ((-512) + ((-512) + y0))), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.where(tmp24, tmp26, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp23, tmp30, tmp31)
    tmp33 = tmp19 >= tmp21
    tmp34 = tl.full([1, 1], 640, tl.int64)
    tmp35 = tmp19 < tmp34
    tmp36 = tmp33 & tmp16
    tmp37 = tl.load(in_ptr4 + (512 + (576*x2) + (451584*y1) + ((-576) + ((-512) + y0))), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.where(tmp22, tmp32, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp16, tmp38, tmp39)
    tmp41 = tl.where(tmp4, tmp15, tmp40)
    tmp43 = tmp41 - tmp42
    tmp45 = 0.001
    tmp46 = tmp44 + tmp45
    tmp47 = libdevice.sqrt(tmp46)
    tmp48 = tl.full([1, 1], 1, tl.int32)
    tmp49 = tmp48 / tmp47
    tmp50 = 1.0
    tmp51 = tmp49 * tmp50
    tmp52 = tmp43 * tmp51
    tmp54 = tmp52 * tmp53
    tmp56 = tmp54 + tmp55
    tmp57 = tl.full([1, 1], 0, tl.int32)
    tmp58 = triton_helpers.maximum(tmp57, tmp56)
    tmp60 = tmp41 - tmp59
    tmp62 = tmp61 + tmp45
    tmp63 = libdevice.sqrt(tmp62)
    tmp64 = tmp48 / tmp63
    tmp65 = tmp64 * tmp50
    tmp66 = tmp60 * tmp65
    tmp68 = tmp66 * tmp67
    tmp70 = tmp68 + tmp69
    tmp71 = triton_helpers.maximum(tmp57, tmp70)
    tl.store(out_ptr1 + (y0 + (1152*x2) + (903168*y1)), tmp58, xmask)
    tl.store(out_ptr2 + (y0 + (1152*x2) + (903168*y1)), tmp71, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h3/ch3agkxfbbbz4seg5vomhbwuq2waumlzf5rrcybpfccimginfixy.py
# Topologically Sorted Source Nodes: [x_310, x_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_310 => add_352, mul_457, mul_458, sub_152
#   x_311 => relu_152
# Graph fragment:
#   %sub_152 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_151, %unsqueeze_1217), kwargs = {})
#   %mul_457 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_152, %unsqueeze_1219), kwargs = {})
#   %mul_458 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_457, %unsqueeze_1221), kwargs = {})
#   %add_352 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_458, %unsqueeze_1223), kwargs = {})
#   %relu_152 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_352,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5017600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 800
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/he/cheovmn64mvu4kc5u6mrg4thu6pkuzs3il3q7af6uy4ky656wkpm.py
# Topologically Sorted Source Nodes: [x_310, x_311, x_in_188], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_310 => add_352, mul_457, mul_458, sub_152
#   x_311 => relu_152
#   x_in_188 => convolution_152
# Graph fragment:
#   %sub_152 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_151, %unsqueeze_1217), kwargs = {})
#   %mul_457 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_152, %unsqueeze_1219), kwargs = {})
#   %mul_458 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_457, %unsqueeze_1221), kwargs = {})
#   %add_352 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_458, %unsqueeze_1223), kwargs = {})
#   %relu_152 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_352,), kwargs = {})
#   %convolution_152 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_152, %arg210_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 50), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12800
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (144*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kf/ckfiyysobo74quhtodu2rhlai3lt3fktb6iv4ysv2wd5tw2ymsky.py
# Topologically Sorted Source Nodes: [x_312, x_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_312 => add_354, mul_460, mul_461, sub_153
#   x_313 => relu_153
# Graph fragment:
#   %sub_153 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_152, %unsqueeze_1225), kwargs = {})
#   %mul_460 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_153, %unsqueeze_1227), kwargs = {})
#   %mul_461 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_460, %unsqueeze_1229), kwargs = {})
#   %add_354 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_461, %unsqueeze_1231), kwargs = {})
#   %relu_153 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_354,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1254400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 800
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hh/chh6pnah6zocuwr6gareoam7xw3whe2ond2ql2ltd7crhcrcjout.py
# Topologically Sorted Source Nodes: [x_in_190, x_314, x_315], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_314 => add_357, mul_463, mul_464, sub_154
#   x_315 => relu_154
#   x_in_190 => cat_95
# Graph fragment:
#   %cat_95 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_355, %cat_94], 1), kwargs = {})
#   %sub_154 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_95, %unsqueeze_1233), kwargs = {})
#   %mul_463 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_154, %unsqueeze_1235), kwargs = {})
#   %mul_464 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_463, %unsqueeze_1237), kwargs = {})
#   %add_357 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_464, %unsqueeze_1239), kwargs = {})
#   %relu_154 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_357,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9728
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1216
    x2 = xindex
    y1 = (yindex // 1216)
    y3 = yindex
    tmp28 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1152*x2) + (225792*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 1216, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1, 1], 128, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr0 + (1024 + (1152*x2) + (225792*y1) + ((-1024) + y0)), tmp17 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp13 >= tmp15
    tmp20 = tl.full([1, 1], 192, tl.int64)
    tmp21 = tmp13 < tmp20
    tmp22 = tmp19 & tmp10
    tmp23 = tl.load(in_ptr1 + (1024 + (1088*x2) + (213248*y1) + ((-128) + ((-1024) + y0))), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp16, tmp18, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp9, tmp26)
    tmp29 = tmp27 - tmp28
    tmp31 = 0.001
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tl.full([1, 1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full([1, 1], 0, tl.int32)
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tl.store(out_ptr1 + (y0 + (1216*x2) + (238336*y1)), tmp44, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eb/cebacbfm66b23gzwyj3xn7z6p7ippipzy3k5yreepi3tk6kleobw.py
# Topologically Sorted Source Nodes: [x_in_194, x_320, x_321], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_320 => add_364, mul_472, mul_473, sub_157
#   x_321 => relu_157
#   x_in_194 => cat_97
# Graph fragment:
#   %cat_97 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_362, %cat_96], 1), kwargs = {})
#   %sub_157 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_97, %unsqueeze_1257), kwargs = {})
#   %mul_472 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_157, %unsqueeze_1259), kwargs = {})
#   %mul_473 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_472, %unsqueeze_1261), kwargs = {})
#   %add_364 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_473, %unsqueeze_1263), kwargs = {})
#   %relu_157 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_364,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1280
    x2 = xindex
    y1 = (yindex // 1280)
    y3 = yindex
    tmp39 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1152*x2) + (225792*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 1280, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1, 1], 192, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1, 1], 128, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + (1024 + (1152*x2) + (225792*y1) + ((-1024) + y0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp15 >= tmp20
    tmp25 = tmp24 & tmp19
    tmp26 = tl.load(in_ptr1 + (1024 + (1088*x2) + (213248*y1) + ((-128) + ((-1024) + y0))), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp21, tmp23, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp19, tmp27, tmp28)
    tmp30 = tmp15 >= tmp17
    tmp31 = tl.full([1, 1], 256, tl.int64)
    tmp32 = tmp15 < tmp31
    tmp33 = tmp30 & tmp12
    tmp34 = tl.load(in_ptr2 + (1024 + (1088*x2) + (213248*y1) + ((-192) + ((-1024) + y0))), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp18, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp12, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp11, tmp37)
    tmp40 = tmp38 - tmp39
    tmp42 = 0.001
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.sqrt(tmp43)
    tmp45 = tl.full([1, 1], 1, tl.int32)
    tmp46 = tmp45 / tmp44
    tmp47 = 1.0
    tmp48 = tmp46 * tmp47
    tmp49 = tmp40 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tmp54 = tl.full([1, 1], 0, tl.int32)
    tmp55 = triton_helpers.maximum(tmp54, tmp53)
    tl.store(out_ptr1 + (y0 + (1280*x2) + (250880*y1)), tmp55, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kx/ckxdjqz342huieqdlb7yebsoguievpzwsrfngftgirnmxbldxydm.py
# Topologically Sorted Source Nodes: [dense_49], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_49 => cat_98
# Graph fragment:
#   %cat_98 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_96, %slice_454], 1), kwargs = {})
triton_poi_fused_cat_30 = async_compile.triton('triton_poi_fused_cat_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 320
    x0 = xindex % 196
    x2 = (xindex // 62720)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 192, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 128, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (1024 + (1152*x0) + (225792*x2) + x1), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp0 >= tmp8
    tmp13 = tmp12 & tmp7
    tmp14 = tl.load(in_ptr1 + (1024 + (1088*x0) + (213248*x2) + ((-128) + x1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp11, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp7, tmp15, tmp16)
    tmp18 = tmp0 >= tmp5
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + (1024 + (1088*x0) + (213248*x2) + ((-192) + x1)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp6, tmp17, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 320, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (1024 + (1088*x0) + (213248*x2) + ((-256) + x1)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bd/cbdaze2uynrh27x2sfewqgtu3yyuxqwr5ujszve2bq673mzpa7sy.py
# Topologically Sorted Source Nodes: [x_in_198, x_326, x_327], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_326 => add_371, mul_481, mul_482, sub_160
#   x_327 => relu_160
#   x_in_198 => cat_99
# Graph fragment:
#   %cat_99 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_369, %cat_98], 1), kwargs = {})
#   %sub_160 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_99, %unsqueeze_1281), kwargs = {})
#   %mul_481 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_160, %unsqueeze_1283), kwargs = {})
#   %mul_482 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_481, %unsqueeze_1285), kwargs = {})
#   %add_371 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_482, %unsqueeze_1287), kwargs = {})
#   %relu_160 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_371,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10752
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1344
    x2 = xindex
    y1 = (yindex // 1344)
    y3 = yindex
    tmp19 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1152*x2) + (225792*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 1344, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr4 + (x2 + (196*((-1024) + y0)) + (62720*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.where(tmp4, tmp13, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 0.001
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tl.full([1, 1], 1, tl.int32)
    tmp26 = tmp25 / tmp24
    tmp27 = 1.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp20 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1, 1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(out_ptr1 + (y0 + (1344*x2) + (263424*y1)), tmp35, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/or/corkshowpxfykkjbvqp6gtne3tscqonqzuf2eagaqtyqv2iayfta.py
# Topologically Sorted Source Nodes: [resid_47, resid_48, resid_49, resid_50], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_47 => add_355
#   resid_48 => add_362
#   resid_49 => add_369
#   resid_50 => add_376
# Graph fragment:
#   %add_355 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_426, %slice_434), kwargs = {})
#   %add_362 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_355, %slice_442), kwargs = {})
#   %add_369 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_362, %slice_450), kwargs = {})
#   %add_376 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_369, %slice_458), kwargs = {})
triton_poi_fused_add_32 = async_compile.triton('triton_poi_fused_add_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (1088*x1)), None)
    tmp3 = tl.load(in_ptr2 + (x0 + (1088*x1)), None)
    tmp5 = tl.load(in_ptr3 + (x0 + (1088*x1)), None)
    tmp7 = tl.load(in_ptr4 + (x0 + (1088*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4p/c4p4l5zdlyactsbxvnjegfk5p7eljdv6r6hnv6twt4bsijosxh3r.py
# Topologically Sorted Source Nodes: [x_in_202, x_332, x_333], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_332 => add_378, mul_490, mul_491, sub_163
#   x_333 => relu_163
#   x_in_202 => cat_101
# Graph fragment:
#   %cat_101 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_376, %cat_100], 1), kwargs = {})
#   %sub_163 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_101, %unsqueeze_1305), kwargs = {})
#   %mul_490 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_163, %unsqueeze_1307), kwargs = {})
#   %mul_491 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_490, %unsqueeze_1309), kwargs = {})
#   %add_378 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_491, %unsqueeze_1311), kwargs = {})
#   %relu_163 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_378,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 11264
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1408
    x2 = xindex
    y1 = (yindex // 1408)
    y3 = yindex
    tmp24 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 1408, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp10 = tmp9 >= tmp1
    tmp11 = tl.full([1, 1], 320, tl.int64)
    tmp12 = tmp9 < tmp11
    tmp13 = tmp12 & tmp6
    tmp14 = tl.load(in_ptr1 + (x2 + (196*((-1024) + y0)) + (62720*y1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp9 >= tmp11
    tmp16 = tl.full([1, 1], 384, tl.int64)
    tmp17 = tmp9 < tmp16
    tmp18 = tmp15 & tmp6
    tmp19 = tl.load(in_ptr2 + (1024 + (1088*x2) + (213248*y1) + ((-320) + ((-1024) + y0))), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp12, tmp14, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp6, tmp20, tmp21)
    tmp23 = tl.where(tmp4, tmp5, tmp22)
    tmp25 = tmp23 - tmp24
    tmp27 = 0.001
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tl.full([1, 1], 1, tl.int32)
    tmp31 = tmp30 / tmp29
    tmp32 = 1.0
    tmp33 = tmp31 * tmp32
    tmp34 = tmp25 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tl.full([1, 1], 0, tl.int32)
    tmp40 = triton_helpers.maximum(tmp39, tmp38)
    tl.store(out_ptr1 + (y0 + (1408*x2) + (275968*y1)), tmp40, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3g/c3g2kzin4t6f2iy3cyaw5wysf62rgdv5qzm4ntuoulrjwgoq6dll.py
# Topologically Sorted Source Nodes: [x_in_206, x_338, x_339], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_338 => add_385, mul_499, mul_500, sub_166
#   x_339 => relu_166
#   x_in_206 => cat_103
# Graph fragment:
#   %cat_103 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_383, %cat_102], 1), kwargs = {})
#   %sub_166 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_103, %unsqueeze_1329), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_166, %unsqueeze_1331), kwargs = {})
#   %mul_500 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_499, %unsqueeze_1333), kwargs = {})
#   %add_385 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_500, %unsqueeze_1335), kwargs = {})
#   %relu_166 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_385,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 11776
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1472
    x2 = xindex
    y1 = (yindex // 1472)
    y3 = yindex
    tmp37 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 1472, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1, 1], 384, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.full([1, 1], 320, tl.int64)
    tmp19 = tmp13 < tmp18
    tmp20 = tmp19 & tmp17
    tmp21 = tl.load(in_ptr2 + (x2 + (196*((-1024) + y0)) + (62720*y1)), tmp20 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp13 >= tmp18
    tmp23 = tmp22 & tmp17
    tmp24 = tl.load(in_ptr3 + (1024 + (1088*x2) + (213248*y1) + ((-320) + ((-1024) + y0))), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.where(tmp19, tmp21, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp17, tmp25, tmp26)
    tmp28 = tmp13 >= tmp15
    tmp29 = tl.full([1, 1], 448, tl.int64)
    tmp30 = tmp13 < tmp29
    tmp31 = tmp28 & tmp10
    tmp32 = tl.load(in_ptr1 + (1024 + (1088*x2) + (213248*y1) + ((-384) + ((-1024) + y0))), tmp31 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp16, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp10, tmp33, tmp34)
    tmp36 = tl.where(tmp4, tmp9, tmp35)
    tmp38 = tmp36 - tmp37
    tmp40 = 0.001
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.sqrt(tmp41)
    tmp43 = tl.full([1, 1], 1, tl.int32)
    tmp44 = tmp43 / tmp42
    tmp45 = 1.0
    tmp46 = tmp44 * tmp45
    tmp47 = tmp38 * tmp46
    tmp49 = tmp47 * tmp48
    tmp51 = tmp49 + tmp50
    tmp52 = tl.full([1, 1], 0, tl.int32)
    tmp53 = triton_helpers.maximum(tmp52, tmp51)
    tl.store(out_ptr1 + (y0 + (1472*x2) + (288512*y1)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3q/c3qtdvnyt5cvznkbg26yp6xckda67v7h2qlxmwmsknoe4cp6zh7z.py
# Topologically Sorted Source Nodes: [dense_52], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_52 => cat_104
# Graph fragment:
#   %cat_104 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_102, %slice_478], 1), kwargs = {})
triton_poi_fused_cat_35 = async_compile.triton('triton_poi_fused_cat_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 196) % 512
    x0 = xindex % 196
    x2 = (xindex // 100352)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 448, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 384, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 320, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (x0 + (196*x1) + (62720*x2)), tmp10, other=0.0)
    tmp12 = tmp0 >= tmp8
    tmp13 = tmp12 & tmp7
    tmp14 = tl.load(in_ptr1 + (1024 + (1088*x0) + (213248*x2) + ((-320) + x1)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp11, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp7, tmp15, tmp16)
    tmp18 = tmp0 >= tmp5
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + (1024 + (1088*x0) + (213248*x2) + ((-384) + x1)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp6, tmp17, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 512, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (1024 + (1088*x0) + (213248*x2) + ((-448) + x1)), tmp24, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6j/c6jmcvpwxtf2zudox4bkb6wynsyindlyncwupb5l2vl6g23z4dyw.py
# Topologically Sorted Source Nodes: [x_in_210, x_344, x_345], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_344 => add_392, mul_508, mul_509, sub_169
#   x_345 => relu_169
#   x_in_210 => cat_105
# Graph fragment:
#   %cat_105 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_390, %cat_104], 1), kwargs = {})
#   %sub_169 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_105, %unsqueeze_1353), kwargs = {})
#   %mul_508 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_169, %unsqueeze_1355), kwargs = {})
#   %mul_509 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_508, %unsqueeze_1357), kwargs = {})
#   %add_392 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_509, %unsqueeze_1359), kwargs = {})
#   %relu_169 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_392,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1536
    x2 = xindex
    y1 = (yindex // 1536)
    y3 = yindex
    tmp17 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 1536, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr3 + (x2 + (196*((-1024) + y0)) + (100352*y1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp4, tmp11, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 0.001
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1, 1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1, 1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr1 + (y0 + (1536*x2) + (301056*y1)), tmp33, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ku/ckuts6v66aayw343sv3o3zzmapip4z6ollmitf3osqqwk7gjf5hd.py
# Topologically Sorted Source Nodes: [x_in_214, x_350, x_351], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_350 => add_399, mul_517, mul_518, sub_172
#   x_351 => relu_172
#   x_in_214 => cat_107
# Graph fragment:
#   %cat_107 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_397, %cat_106], 1), kwargs = {})
#   %sub_172 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_107, %unsqueeze_1377), kwargs = {})
#   %mul_517 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_172, %unsqueeze_1379), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_517, %unsqueeze_1381), kwargs = {})
#   %add_399 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_518, %unsqueeze_1383), kwargs = {})
#   %relu_172 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_399,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12800
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1600
    x2 = xindex
    y1 = (yindex // 1600)
    y3 = yindex
    tmp32 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 1600, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1, 1], 512, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.load(in_ptr4 + (x2 + (196*((-1024) + y0)) + (100352*y1)), tmp21 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp17 >= tmp19
    tmp24 = tl.full([1, 1], 576, tl.int64)
    tmp25 = tmp17 < tmp24
    tmp26 = tmp23 & tmp14
    tmp27 = tl.load(in_ptr3 + (1024 + (1088*x2) + (213248*y1) + ((-512) + ((-1024) + y0))), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp20, tmp22, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp14, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp13, tmp30)
    tmp33 = tmp31 - tmp32
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tl.full([1, 1], 1, tl.int32)
    tmp39 = tmp38 / tmp37
    tmp40 = 1.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp33 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tl.full([1, 1], 0, tl.int32)
    tmp48 = triton_helpers.maximum(tmp47, tmp46)
    tl.store(out_ptr1 + (y0 + (1600*x2) + (313600*y1)), tmp48, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/r2/cr2vndbfuusyuxb4je2fxgykg4o7hqbkpkarjiliz7p3hlrefo3c.py
# Topologically Sorted Source Nodes: [resid_51, resid_52, resid_53, resid_54], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_51 => add_383
#   resid_52 => add_390
#   resid_53 => add_397
#   resid_54 => add_404
# Graph fragment:
#   %add_383 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_376, %slice_466), kwargs = {})
#   %add_390 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_383, %slice_474), kwargs = {})
#   %add_397 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_390, %slice_482), kwargs = {})
#   %add_404 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_397, %slice_490), kwargs = {})
triton_poi_fused_add_38 = async_compile.triton('triton_poi_fused_add_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (1088*x1)), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (1088*x1)), None)
    tmp5 = tl.load(in_ptr2 + (x0 + (1088*x1)), None)
    tmp7 = tl.load(in_ptr3 + (x0 + (1088*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vo/cvonreummtllmlslebxbjq2jfid6fzzuo6i375xb2s4ljhqt7ezi.py
# Topologically Sorted Source Nodes: [x_in_218, x_356, x_357], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_356 => add_406, mul_526, mul_527, sub_175
#   x_357 => relu_175
#   x_in_218 => cat_109
# Graph fragment:
#   %cat_109 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_404, %cat_108], 1), kwargs = {})
#   %sub_175 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_109, %unsqueeze_1401), kwargs = {})
#   %mul_526 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_175, %unsqueeze_1403), kwargs = {})
#   %mul_527 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_526, %unsqueeze_1405), kwargs = {})
#   %add_406 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_527, %unsqueeze_1407), kwargs = {})
#   %relu_175 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_406,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 13312
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1664
    x2 = xindex
    y1 = (yindex // 1664)
    y3 = yindex
    tmp33 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 1664, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp10 = tmp9 >= tmp1
    tmp11 = tl.full([1, 1], 576, tl.int64)
    tmp12 = tmp9 < tmp11
    tmp13 = tmp12 & tmp6
    tmp14 = tl.full([1, 1], 512, tl.int64)
    tmp15 = tmp9 < tmp14
    tmp16 = tmp15 & tmp13
    tmp17 = tl.load(in_ptr1 + (x2 + (196*((-1024) + y0)) + (100352*y1)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp9 >= tmp14
    tmp19 = tmp18 & tmp13
    tmp20 = tl.load(in_ptr2 + (1024 + (1088*x2) + (213248*y1) + ((-512) + ((-1024) + y0))), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp15, tmp17, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp13, tmp21, tmp22)
    tmp24 = tmp9 >= tmp11
    tmp25 = tl.full([1, 1], 640, tl.int64)
    tmp26 = tmp9 < tmp25
    tmp27 = tmp24 & tmp6
    tmp28 = tl.load(in_ptr3 + (1024 + (1088*x2) + (213248*y1) + ((-576) + ((-1024) + y0))), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.where(tmp12, tmp23, tmp28)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp6, tmp29, tmp30)
    tmp32 = tl.where(tmp4, tmp5, tmp31)
    tmp34 = tmp32 - tmp33
    tmp36 = 0.001
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.sqrt(tmp37)
    tmp39 = tl.full([1, 1], 1, tl.int32)
    tmp40 = tmp39 / tmp38
    tmp41 = 1.0
    tmp42 = tmp40 * tmp41
    tmp43 = tmp34 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tmp48 = tl.full([1, 1], 0, tl.int32)
    tmp49 = triton_helpers.maximum(tmp48, tmp47)
    tl.store(out_ptr1 + (y0 + (1664*x2) + (326144*y1)), tmp49, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qg/cqgu4jqmjw3djxcg42jf6euk4cnzlunz6gjn33ow4zsdwbjnkbqo.py
# Topologically Sorted Source Nodes: [dense_55], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_55 => cat_110
# Graph fragment:
#   %cat_110 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_108, %slice_502], 1), kwargs = {})
triton_poi_fused_cat_40 = async_compile.triton('triton_poi_fused_cat_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1103872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 704
    x0 = xindex % 196
    x2 = (xindex // 137984)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 640, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 576, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 512, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (x0 + (196*x1) + (100352*x2)), tmp10 & xmask, other=0.0)
    tmp12 = tmp0 >= tmp8
    tmp13 = tmp12 & tmp7
    tmp14 = tl.load(in_ptr1 + (1024 + (1088*x0) + (213248*x2) + ((-512) + x1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp11, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp7, tmp15, tmp16)
    tmp18 = tmp0 >= tmp5
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + (1024 + (1088*x0) + (213248*x2) + ((-576) + x1)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp6, tmp17, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 704, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (1024 + (1088*x0) + (213248*x2) + ((-640) + x1)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/de/cdeqemqxxtc3njbgcurxs5wsfw5h25aghtnjyuvfmvxs7jesshfq.py
# Topologically Sorted Source Nodes: [x_in_222, x_362, x_363], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_362 => add_413, mul_535, mul_536, sub_178
#   x_363 => relu_178
#   x_in_222 => cat_111
# Graph fragment:
#   %cat_111 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_411, %cat_110], 1), kwargs = {})
#   %sub_178 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_111, %unsqueeze_1425), kwargs = {})
#   %mul_535 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_178, %unsqueeze_1427), kwargs = {})
#   %mul_536 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_535, %unsqueeze_1429), kwargs = {})
#   %add_413 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_536, %unsqueeze_1431), kwargs = {})
#   %relu_178 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_413,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1728
    x3 = (xindex // 1728)
    x1 = (xindex // 1728) % 196
    x2 = (xindex // 338688)
    x4 = xindex
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x3) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x3) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1728, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (x1 + (196*((-1024) + x0)) + (137984*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = tl.full([1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp16 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(out_ptr0 + (x4), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rt/crthndkmw6q53wqlbescljdnzy6gg5s4hwim7cxpnraz75ata3p5.py
# Topologically Sorted Source Nodes: [x_in_226, x_368, x_369], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_368 => add_420, mul_544, mul_545, sub_181
#   x_369 => relu_181
#   x_in_226 => cat_113
# Graph fragment:
#   %cat_113 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_418, %cat_112], 1), kwargs = {})
#   %sub_181 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_113, %unsqueeze_1449), kwargs = {})
#   %mul_544 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_181, %unsqueeze_1451), kwargs = {})
#   %mul_545 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_544, %unsqueeze_1453), kwargs = {})
#   %add_420 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_545, %unsqueeze_1455), kwargs = {})
#   %relu_181 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_420,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 14336
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1792
    x2 = xindex
    y1 = (yindex // 1792)
    y3 = yindex
    tmp30 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 1792, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1, 1], 704, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.load(in_ptr3 + (x2 + (196*((-1024) + y0)) + (137984*y1)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp15 >= tmp17
    tmp22 = tl.full([1, 1], 768, tl.int64)
    tmp23 = tmp15 < tmp22
    tmp24 = tmp21 & tmp12
    tmp25 = tl.load(in_ptr2 + (1024 + (1088*x2) + (213248*y1) + ((-704) + ((-1024) + y0))), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.where(tmp18, tmp20, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp12, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp11, tmp28)
    tmp31 = tmp29 - tmp30
    tmp33 = 0.001
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1, 1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1, 1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tl.store(out_ptr1 + (y0 + (1792*x2) + (351232*y1)), tmp46, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/k3/ck3vcuiel3byiqe4ta2sckegs3vct5qou3evki7ya2vso2qitqs7.py
# Topologically Sorted Source Nodes: [x_in_230, x_374, x_375], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_374 => add_427, mul_553, mul_554, sub_184
#   x_375 => relu_184
#   x_in_230 => cat_115
# Graph fragment:
#   %cat_115 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_425, %cat_114], 1), kwargs = {})
#   %sub_184 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_115, %unsqueeze_1473), kwargs = {})
#   %mul_553 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_184, %unsqueeze_1475), kwargs = {})
#   %mul_554 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_553, %unsqueeze_1477), kwargs = {})
#   %add_427 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_554, %unsqueeze_1479), kwargs = {})
#   %relu_184 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_427,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 14848
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1856
    x2 = xindex
    y1 = (yindex // 1856)
    y3 = yindex
    tmp41 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 1856, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1, 1], 768, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.full([1, 1], 704, tl.int64)
    tmp23 = tmp17 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.load(in_ptr4 + (x2 + (196*((-1024) + y0)) + (137984*y1)), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp17 >= tmp22
    tmp27 = tmp26 & tmp21
    tmp28 = tl.load(in_ptr2 + (1024 + (1088*x2) + (213248*y1) + ((-704) + ((-1024) + y0))), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.where(tmp23, tmp25, tmp28)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp21, tmp29, tmp30)
    tmp32 = tmp17 >= tmp19
    tmp33 = tl.full([1, 1], 832, tl.int64)
    tmp34 = tmp17 < tmp33
    tmp35 = tmp32 & tmp14
    tmp36 = tl.load(in_ptr3 + (1024 + (1088*x2) + (213248*y1) + ((-768) + ((-1024) + y0))), tmp35 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.where(tmp20, tmp31, tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp14, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp13, tmp39)
    tmp42 = tmp40 - tmp41
    tmp44 = 0.001
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = tl.full([1, 1], 1, tl.int32)
    tmp48 = tmp47 / tmp46
    tmp49 = 1.0
    tmp50 = tmp48 * tmp49
    tmp51 = tmp42 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tl.full([1, 1], 0, tl.int32)
    tmp57 = triton_helpers.maximum(tmp56, tmp55)
    tl.store(out_ptr1 + (y0 + (1856*x2) + (363776*y1)), tmp57, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tz/ctz52nhf3wxtiutoesf6avwy6ldx223zedmhgom53dl57eiqobn7.py
# Topologically Sorted Source Nodes: [resid_55, resid_56, resid_57, resid_58], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_55 => add_411
#   resid_56 => add_418
#   resid_57 => add_425
#   resid_58 => add_432
# Graph fragment:
#   %add_411 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_404, %slice_498), kwargs = {})
#   %add_418 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_411, %slice_506), kwargs = {})
#   %add_425 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_418, %slice_514), kwargs = {})
#   %add_432 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_425, %slice_522), kwargs = {})
triton_poi_fused_add_44 = async_compile.triton('triton_poi_fused_add_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1088*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (1088*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (1088*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (1088*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (y0 + (196*x2) + (376320*y1)), tmp8, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fz/cfzds37agapqpbwo6poli4animtewqlf4xahoo2alcxvglnazsqa.py
# Topologically Sorted Source Nodes: [dense_58], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_58 => cat_116
# Graph fragment:
#   %cat_116 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_114, %slice_526], 1), kwargs = {})
triton_poi_fused_cat_45 = async_compile.triton('triton_poi_fused_cat_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 196) % 896
    x0 = xindex % 196
    x2 = (xindex // 175616)
    x3 = xindex % 175616
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 832, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 768, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 704, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (x0 + (196*x1) + (137984*x2)), tmp10, other=0.0)
    tmp12 = tmp0 >= tmp8
    tmp13 = tmp12 & tmp7
    tmp14 = tl.load(in_ptr1 + (1024 + (1088*x0) + (213248*x2) + ((-704) + x1)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp11, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp7, tmp15, tmp16)
    tmp18 = tmp0 >= tmp5
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + (1024 + (1088*x0) + (213248*x2) + ((-768) + x1)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp6, tmp17, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 896, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (1024 + (1088*x0) + (213248*x2) + ((-832) + x1)), tmp24, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3 + (376320*x2)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/e5/ce5x2d6524dbqlr665lynbqr5jzo3p5qy742km7i4pyhlwu4lkmz.py
# Topologically Sorted Source Nodes: [x_380, x_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_380 => add_434, mul_562, mul_563, sub_187
#   x_381 => relu_187
# Graph fragment:
#   %sub_187 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_117, %unsqueeze_1497), kwargs = {})
#   %mul_562 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_187, %unsqueeze_1499), kwargs = {})
#   %mul_563 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_562, %unsqueeze_1501), kwargs = {})
#   %add_434 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_563, %unsqueeze_1503), kwargs = {})
#   %relu_187 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_434,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_46', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15360
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1920
    y1 = (yindex // 1920)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (y0 + (1920*x2) + (376320*y1)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/aa/caaqkkgmhjtogwutf5mtcxirvtpepj7t4436chi7j2ced64ilxko.py
# Topologically Sorted Source Nodes: [x_in_238, x_386, x_387], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_386 => add_441, mul_571, mul_572, sub_190
#   x_387 => relu_190
#   x_in_238 => cat_119
# Graph fragment:
#   %cat_119 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_439, %cat_118], 1), kwargs = {})
#   %sub_190 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_119, %unsqueeze_1521), kwargs = {})
#   %mul_571 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_190, %unsqueeze_1523), kwargs = {})
#   %mul_572 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_571, %unsqueeze_1525), kwargs = {})
#   %add_441 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_572, %unsqueeze_1527), kwargs = {})
#   %relu_190 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_441,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15872
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1984
    x2 = xindex
    y1 = (yindex // 1984)
    y3 = yindex
    tmp28 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (376320*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 1984, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1, 1], 896, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr2 + (x2 + (196*((-1024) + y0)) + (376320*y1)), tmp17 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp13 >= tmp15
    tmp20 = tl.full([1, 1], 960, tl.int64)
    tmp21 = tmp13 < tmp20
    tmp22 = tmp19 & tmp10
    tmp23 = tl.load(in_ptr1 + (1024 + (1088*x2) + (213248*y1) + ((-896) + ((-1024) + y0))), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp16, tmp18, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp9, tmp26)
    tmp29 = tmp27 - tmp28
    tmp31 = 0.001
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tl.full([1, 1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full([1, 1], 0, tl.int32)
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tl.store(out_ptr1 + (y0 + (1984*x2) + (388864*y1)), tmp44, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qj/cqju4w6rrizgslarcw52cvq34xkgad7abjt2fawcvhi627bzoe4x.py
# Topologically Sorted Source Nodes: [x_in_242, x_392, x_393], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_392 => add_448, mul_580, mul_581, sub_193
#   x_393 => relu_193
#   x_in_242 => cat_121
# Graph fragment:
#   %cat_121 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_446, %cat_120], 1), kwargs = {})
#   %sub_193 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_121, %unsqueeze_1545), kwargs = {})
#   %mul_580 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_193, %unsqueeze_1547), kwargs = {})
#   %mul_581 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_580, %unsqueeze_1549), kwargs = {})
#   %add_448 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_581, %unsqueeze_1551), kwargs = {})
#   %relu_193 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_448,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2048
    x2 = xindex
    y1 = (yindex // 2048)
    y3 = yindex
    tmp38 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (376320*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 2048, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1, 1], 960, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1, 1], 896, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr3 + (x2 + (196*((-1024) + y0)) + (376320*y1)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp15 >= tmp20
    tmp25 = tmp24 & tmp19
    tmp26 = tl.load(in_ptr1 + (1024 + (1088*x2) + (213248*y1) + ((-896) + ((-1024) + y0))), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp21, tmp23, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp19, tmp27, tmp28)
    tmp30 = tmp15 >= tmp17
    tmp31 = tmp15 < tmp3
    tmp32 = tmp30 & tmp12
    tmp33 = tl.load(in_ptr2 + (1024 + (1088*x2) + (213248*y1) + ((-960) + ((-1024) + y0))), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp18, tmp29, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp12, tmp34, tmp35)
    tmp37 = tl.where(tmp4, tmp11, tmp36)
    tmp39 = tmp37 - tmp38
    tmp41 = 0.001
    tmp42 = tmp40 + tmp41
    tmp43 = libdevice.sqrt(tmp42)
    tmp44 = tl.full([1, 1], 1, tl.int32)
    tmp45 = tmp44 / tmp43
    tmp46 = 1.0
    tmp47 = tmp45 * tmp46
    tmp48 = tmp39 * tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = tl.full([1, 1], 0, tl.int32)
    tmp54 = triton_helpers.maximum(tmp53, tmp52)
    tl.store(out_ptr1 + (y0 + (2048*x2) + (401408*y1)), tmp54, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tp/ctpuepb2nvk5yb4wkx3emnrcikhafn6ay45gn35m3mrl46x23ziq.py
# Topologically Sorted Source Nodes: [dense_61], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_61 => cat_122
# Graph fragment:
#   %cat_122 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_120, %slice_550], 1), kwargs = {})
triton_poi_fused_cat_49 = async_compile.triton('triton_poi_fused_cat_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x0 = xindex % 196
    x2 = (xindex // 213248)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 960, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 896, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (x0 + (196*x1) + (376320*x2)), tmp10 & xmask, other=0.0)
    tmp12 = tmp0 >= tmp8
    tmp13 = tmp12 & tmp7
    tmp14 = tl.load(in_ptr1 + (1024 + (1088*x0) + (213248*x2) + ((-896) + x1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp11, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp7, tmp15, tmp16)
    tmp18 = tmp0 >= tmp5
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + (1024 + (1088*x0) + (213248*x2) + ((-960) + x1)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp6, tmp17, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 1088, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (1024 + (1088*x0) + (213248*x2) + ((-1024) + x1)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yw/cywtvvcr37urvobbdxffozemplvjhwvn6j4xetipel6pjd32mwlg.py
# Topologically Sorted Source Nodes: [x_in_246, x_398, x_399], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_398 => add_455, mul_589, mul_590, sub_196
#   x_399 => relu_196
#   x_in_246 => cat_123
# Graph fragment:
#   %cat_123 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_453, %cat_122], 1), kwargs = {})
#   %sub_196 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_123, %unsqueeze_1569), kwargs = {})
#   %mul_589 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_196, %unsqueeze_1571), kwargs = {})
#   %mul_590 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_589, %unsqueeze_1573), kwargs = {})
#   %add_455 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_590, %unsqueeze_1575), kwargs = {})
#   %relu_196 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_455,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_50', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2112
    x2 = xindex
    y1 = (yindex // 2112)
    y3 = yindex
    tmp19 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (376320*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 2112, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr4 + (x2 + (196*((-1024) + y0)) + (213248*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.where(tmp4, tmp13, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 0.001
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tl.full([1, 1], 1, tl.int32)
    tmp26 = tmp25 / tmp24
    tmp27 = 1.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp20 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1, 1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(out_ptr1 + (y0 + (2112*x2) + (413952*y1)), tmp35, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/u3/cu33g5jcrh3hjb53pxrjr3d5titqsicoqx2torxqz5grghynemfg.py
# Topologically Sorted Source Nodes: [resid_59, resid_60, resid_61, resid_62], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   resid_59 => add_439
#   resid_60 => add_446
#   resid_61 => add_453
#   resid_62 => add_460
# Graph fragment:
#   %add_439 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_432, %slice_530), kwargs = {})
#   %add_446 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_439, %slice_538), kwargs = {})
#   %add_453 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_446, %slice_546), kwargs = {})
#   %add_460 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_453, %slice_554), kwargs = {})
triton_poi_fused_add_51 = async_compile.triton('triton_poi_fused_add_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (376320*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1088*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (1088*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (1088*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (1088*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp8, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ri/crienritsxtbjaac4zensw6yuyuqom6oyic47pvhjso263c3luwd.py
# Topologically Sorted Source Nodes: [x_in_250, x_404, x_405], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_404 => add_462, mul_598, mul_599, sub_199
#   x_405 => relu_199
#   x_in_250 => cat_125
# Graph fragment:
#   %cat_125 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_460, %cat_124], 1), kwargs = {})
#   %sub_199 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_125, %unsqueeze_1593), kwargs = {})
#   %mul_598 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_199, %unsqueeze_1595), kwargs = {})
#   %mul_599 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_598, %unsqueeze_1597), kwargs = {})
#   %add_462 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_599, %unsqueeze_1599), kwargs = {})
#   %relu_199 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_462,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 17408
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2176
    x2 = xindex
    y1 = (yindex // 2176)
    y3 = yindex
    tmp24 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 2176, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp10 = tmp9 >= tmp1
    tmp11 = tl.full([1, 1], 1088, tl.int64)
    tmp12 = tmp9 < tmp11
    tmp13 = tmp12 & tmp6
    tmp14 = tl.load(in_ptr1 + (x2 + (196*((-1024) + y0)) + (213248*y1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp9 >= tmp11
    tmp16 = tl.full([1, 1], 1152, tl.int64)
    tmp17 = tmp9 < tmp16
    tmp18 = tmp15 & tmp6
    tmp19 = tl.load(in_ptr2 + (1024 + (1088*x2) + (213248*y1) + ((-1088) + ((-1024) + y0))), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp12, tmp14, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp6, tmp20, tmp21)
    tmp23 = tl.where(tmp4, tmp5, tmp22)
    tmp25 = tmp23 - tmp24
    tmp27 = 0.001
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tl.full([1, 1], 1, tl.int32)
    tmp31 = tmp30 / tmp29
    tmp32 = 1.0
    tmp33 = tmp31 * tmp32
    tmp34 = tmp25 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tl.full([1, 1], 0, tl.int32)
    tmp40 = triton_helpers.maximum(tmp39, tmp38)
    tl.store(out_ptr1 + (y0 + (2176*x2) + (426496*y1)), tmp40, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5e/c5etot5nw7sosy7z64wizsghb7ckv7kq52smgap5beefsdre4chh.py
# Topologically Sorted Source Nodes: [x_in_254, x_410, x_411], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_410 => add_469, mul_607, mul_608, sub_202
#   x_411 => relu_202
#   x_in_254 => cat_127
# Graph fragment:
#   %cat_127 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_467, %cat_126], 1), kwargs = {})
#   %sub_202 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_127, %unsqueeze_1617), kwargs = {})
#   %mul_607 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_202, %unsqueeze_1619), kwargs = {})
#   %mul_608 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_607, %unsqueeze_1621), kwargs = {})
#   %add_469 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_608, %unsqueeze_1623), kwargs = {})
#   %relu_202 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_469,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 17920
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2240
    x2 = xindex
    y1 = (yindex // 2240)
    y3 = yindex
    tmp37 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 2240, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1, 1], 1152, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.full([1, 1], 1088, tl.int64)
    tmp19 = tmp13 < tmp18
    tmp20 = tmp19 & tmp17
    tmp21 = tl.load(in_ptr2 + (x2 + (196*((-1024) + y0)) + (213248*y1)), tmp20 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp13 >= tmp18
    tmp23 = tmp22 & tmp17
    tmp24 = tl.load(in_ptr3 + (1024 + (1088*x2) + (213248*y1) + ((-1088) + ((-1024) + y0))), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.where(tmp19, tmp21, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp17, tmp25, tmp26)
    tmp28 = tmp13 >= tmp15
    tmp29 = tl.full([1, 1], 1216, tl.int64)
    tmp30 = tmp13 < tmp29
    tmp31 = tmp28 & tmp10
    tmp32 = tl.load(in_ptr1 + (1024 + (1088*x2) + (213248*y1) + ((-1152) + ((-1024) + y0))), tmp31 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp16, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp10, tmp33, tmp34)
    tmp36 = tl.where(tmp4, tmp9, tmp35)
    tmp38 = tmp36 - tmp37
    tmp40 = 0.001
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.sqrt(tmp41)
    tmp43 = tl.full([1, 1], 1, tl.int32)
    tmp44 = tmp43 / tmp42
    tmp45 = 1.0
    tmp46 = tmp44 * tmp45
    tmp47 = tmp38 * tmp46
    tmp49 = tmp47 * tmp48
    tmp51 = tmp49 + tmp50
    tmp52 = tl.full([1, 1], 0, tl.int32)
    tmp53 = triton_helpers.maximum(tmp52, tmp51)
    tl.store(out_ptr1 + (y0 + (2240*x2) + (439040*y1)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kg/ckgyt64fyebhy2syuciiu267l6kkwrqzgqnw4yyolm6sfhvztbrd.py
# Topologically Sorted Source Nodes: [dense_64], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_64 => cat_128
# Graph fragment:
#   %cat_128 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_126, %slice_574], 1), kwargs = {})
triton_poi_fused_cat_54 = async_compile.triton('triton_poi_fused_cat_54', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 196) % 1280
    x0 = xindex % 196
    x2 = (xindex // 250880)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1216, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 1152, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 1088, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (x0 + (196*x1) + (213248*x2)), tmp10, other=0.0)
    tmp12 = tmp0 >= tmp8
    tmp13 = tmp12 & tmp7
    tmp14 = tl.load(in_ptr1 + (1024 + (1088*x0) + (213248*x2) + ((-1088) + x1)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp11, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp7, tmp15, tmp16)
    tmp18 = tmp0 >= tmp5
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + (1024 + (1088*x0) + (213248*x2) + ((-1152) + x1)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp6, tmp17, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 1280, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (1024 + (1088*x0) + (213248*x2) + ((-1216) + x1)), tmp24, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mh/cmhh4sowgsyqypzjegylocruobqcvtvketbg6qkcreks3ymf6wh7.py
# Topologically Sorted Source Nodes: [x_in_258, x_416, x_417], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_416 => add_476, mul_616, mul_617, sub_205
#   x_417 => relu_205
#   x_in_258 => cat_129
# Graph fragment:
#   %cat_129 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_474, %cat_128], 1), kwargs = {})
#   %sub_205 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_129, %unsqueeze_1641), kwargs = {})
#   %mul_616 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_205, %unsqueeze_1643), kwargs = {})
#   %mul_617 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_616, %unsqueeze_1645), kwargs = {})
#   %add_476 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_617, %unsqueeze_1647), kwargs = {})
#   %relu_205 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_476,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2304
    x2 = xindex
    y1 = (yindex // 2304)
    y3 = yindex
    tmp17 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 2304, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr3 + (x2 + (196*((-1024) + y0)) + (250880*y1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp4, tmp11, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 0.001
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1, 1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1, 1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr1 + (y0 + (2304*x2) + (451584*y1)), tmp33, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3o/c3obrjut2acsld7x23uvwnh7lntucnucshjgc5nkmxw3cmudppnj.py
# Topologically Sorted Source Nodes: [x_in_262, x_422, x_423], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_422 => add_483, mul_625, mul_626, sub_208
#   x_423 => relu_208
#   x_in_262 => cat_131
# Graph fragment:
#   %cat_131 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_481, %cat_130], 1), kwargs = {})
#   %sub_208 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_131, %unsqueeze_1665), kwargs = {})
#   %mul_625 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_208, %unsqueeze_1667), kwargs = {})
#   %mul_626 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_625, %unsqueeze_1669), kwargs = {})
#   %add_483 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_626, %unsqueeze_1671), kwargs = {})
#   %relu_208 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_483,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_56', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18944
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2368
    x2 = xindex
    y1 = (yindex // 2368)
    y3 = yindex
    tmp32 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 2368, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1, 1], 1280, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.load(in_ptr4 + (x2 + (196*((-1024) + y0)) + (250880*y1)), tmp21 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp17 >= tmp19
    tmp24 = tl.full([1, 1], 1344, tl.int64)
    tmp25 = tmp17 < tmp24
    tmp26 = tmp23 & tmp14
    tmp27 = tl.load(in_ptr3 + (1024 + (1088*x2) + (213248*y1) + ((-1280) + ((-1024) + y0))), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp20, tmp22, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp14, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp13, tmp30)
    tmp33 = tmp31 - tmp32
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tl.full([1, 1], 1, tl.int32)
    tmp39 = tmp38 / tmp37
    tmp40 = 1.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp33 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tl.full([1, 1], 0, tl.int32)
    tmp48 = triton_helpers.maximum(tmp47, tmp46)
    tl.store(out_ptr1 + (y0 + (2368*x2) + (464128*y1)), tmp48, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ph/cph7nnpzduwf4fbhrx6xpg5hor5alcv4jhoh42psvyrawntclykl.py
# Topologically Sorted Source Nodes: [x_in_266, x_428, x_429, x_430, x_431], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_428 => add_490, mul_634, mul_635, sub_211
#   x_429 => relu_211
#   x_430 => add_492, mul_637, mul_638, sub_212
#   x_431 => relu_212
#   x_in_266 => cat_133
# Graph fragment:
#   %cat_133 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_488, %cat_132], 1), kwargs = {})
#   %sub_211 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_133, %unsqueeze_1689), kwargs = {})
#   %mul_634 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_211, %unsqueeze_1691), kwargs = {})
#   %mul_635 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_634, %unsqueeze_1693), kwargs = {})
#   %add_490 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_635, %unsqueeze_1695), kwargs = {})
#   %relu_211 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_490,), kwargs = {})
#   %sub_212 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_133, %unsqueeze_1697), kwargs = {})
#   %mul_637 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_212, %unsqueeze_1699), kwargs = {})
#   %mul_638 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_637, %unsqueeze_1701), kwargs = {})
#   %add_492 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_638, %unsqueeze_1703), kwargs = {})
#   %relu_212 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_492,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_57', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 19456
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2432
    x2 = xindex
    y1 = (yindex // 2432)
    y3 = yindex
    tmp43 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr10 + (y0), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr11 + (y0), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr12 + (y0), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr13 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr4 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1, 1], 2432, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp20 = tmp19 >= tmp1
    tmp21 = tl.full([1, 1], 1344, tl.int64)
    tmp22 = tmp19 < tmp21
    tmp23 = tmp22 & tmp16
    tmp24 = tl.full([1, 1], 1280, tl.int64)
    tmp25 = tmp19 < tmp24
    tmp26 = tmp25 & tmp23
    tmp27 = tl.load(in_ptr5 + (x2 + (196*((-1024) + y0)) + (250880*y1)), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp19 >= tmp24
    tmp29 = tmp28 & tmp23
    tmp30 = tl.load(in_ptr3 + (1024 + (1088*x2) + (213248*y1) + ((-1280) + ((-1024) + y0))), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.where(tmp25, tmp27, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp23, tmp31, tmp32)
    tmp34 = tmp19 >= tmp21
    tmp35 = tl.full([1, 1], 1408, tl.int64)
    tmp36 = tmp19 < tmp35
    tmp37 = tmp34 & tmp16
    tmp38 = tl.load(in_ptr4 + (1024 + (1088*x2) + (213248*y1) + ((-1344) + ((-1024) + y0))), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.where(tmp22, tmp33, tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp16, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp15, tmp41)
    tmp44 = tmp42 - tmp43
    tmp46 = 0.001
    tmp47 = tmp45 + tmp46
    tmp48 = libdevice.sqrt(tmp47)
    tmp49 = tl.full([1, 1], 1, tl.int32)
    tmp50 = tmp49 / tmp48
    tmp51 = 1.0
    tmp52 = tmp50 * tmp51
    tmp53 = tmp44 * tmp52
    tmp55 = tmp53 * tmp54
    tmp57 = tmp55 + tmp56
    tmp58 = tl.full([1, 1], 0, tl.int32)
    tmp59 = triton_helpers.maximum(tmp58, tmp57)
    tmp61 = tmp42 - tmp60
    tmp63 = tmp62 + tmp46
    tmp64 = libdevice.sqrt(tmp63)
    tmp65 = tmp49 / tmp64
    tmp66 = tmp65 * tmp51
    tmp67 = tmp61 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = triton_helpers.maximum(tmp58, tmp71)
    tl.store(out_ptr1 + (y0 + (2432*x2) + (476672*y1)), tmp59, xmask)
    tl.store(out_ptr2 + (y0 + (2432*x2) + (476672*y1)), tmp72, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vo/cvows2nv3ehdudmu54dqyyu6b7qiwjng2r3rs65ebg662zcyzlye.py
# Topologically Sorted Source Nodes: [x_432, x_433], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_432 => add_494, mul_640, mul_641, sub_213
#   x_433 => relu_213
# Graph fragment:
#   %sub_213 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_212, %unsqueeze_1705), kwargs = {})
#   %mul_640 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_213, %unsqueeze_1707), kwargs = {})
#   %mul_641 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_640, %unsqueeze_1709), kwargs = {})
#   %add_494 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_641, %unsqueeze_1711), kwargs = {})
#   %relu_213 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_494,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_58 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_58', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1600
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bg/cbg62iybyq2m6lviqvdnctjiap5gn66kplh72sruaapjhwnubifq.py
# Topologically Sorted Source Nodes: [x_432, x_433, x_in_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_432 => add_494, mul_640, mul_641, sub_213
#   x_433 => relu_213
#   x_in_268 => convolution_213
# Graph fragment:
#   %sub_213 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_212, %unsqueeze_1705), kwargs = {})
#   %mul_640 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_213, %unsqueeze_1707), kwargs = {})
#   %mul_641 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_640, %unsqueeze_1709), kwargs = {})
#   %add_494 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_641, %unsqueeze_1711), kwargs = {})
#   %relu_213 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_494,), kwargs = {})
#   %convolution_213 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_213, %arg515_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 50), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_59 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_59', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_59(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 51200
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (288*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nb/cnbfeeubjpclqbz2vukidrxjdc3twx43zyeq4ywdndkf54ucu7hi.py
# Topologically Sorted Source Nodes: [x_434, x_435], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_434 => add_496, mul_643, mul_644, sub_214
#   x_435 => relu_214
# Graph fragment:
#   %sub_214 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_213, %unsqueeze_1713), kwargs = {})
#   %mul_643 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_214, %unsqueeze_1715), kwargs = {})
#   %mul_644 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_643, %unsqueeze_1717), kwargs = {})
#   %add_496 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_644, %unsqueeze_1719), kwargs = {})
#   %relu_214 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_496,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_60 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_60', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_60', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_60(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 627200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1600
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dp/cdpjzwv3ne6mtsr6qjeczgapkjaimjnud6wxtpsqbplxwhhhua45.py
# Topologically Sorted Source Nodes: [x_in_270, x_436, x_437], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_436 => add_499, mul_646, mul_647, sub_215
#   x_437 => relu_215
#   x_in_270 => cat_135
# Graph fragment:
#   %cat_135 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_497, %cat_134], 1), kwargs = {})
#   %sub_215 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_135, %unsqueeze_1721), kwargs = {})
#   %mul_646 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_215, %unsqueeze_1723), kwargs = {})
#   %mul_647 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_646, %unsqueeze_1725), kwargs = {})
#   %add_499 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_647, %unsqueeze_1727), kwargs = {})
#   %relu_215 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_499,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 19456
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2432
    x2 = xindex
    y1 = (yindex // 2432)
    y3 = yindex
    tmp28 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2304*x2) + (112896*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((2176*x2) + (106624*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 2432, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.broadcast_to((-2048) + y0, [XBLOCK, YBLOCK])
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1, 1], 256, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr0 + (2048 + (2304*x2) + (112896*y1) + ((-2048) + y0)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp13 >= tmp15
    tmp20 = tl.full([1, 1], 384, tl.int64)
    tmp21 = tmp13 < tmp20
    tmp22 = tmp19 & tmp10
    tmp23 = tl.load(in_ptr1 + (2048 + (2176*x2) + (106624*y1) + ((-256) + ((-2048) + y0))), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp16, tmp18, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp9, tmp26)
    tmp29 = tmp27 - tmp28
    tmp31 = 0.001
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tl.full([1, 1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full([1, 1], 0, tl.int32)
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tl.store(out_ptr1 + (y0 + (2432*x2) + (119168*y1)), tmp44, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yx/cyxcakwm6sljdpko32uptuhxqj2qhxetb7pv3xyn2xip5woxyhoq.py
# Topologically Sorted Source Nodes: [x_in_274, x_442, x_443], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_442 => add_506, mul_655, mul_656, sub_218
#   x_443 => relu_218
#   x_in_274 => cat_137
# Graph fragment:
#   %cat_137 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_504, %cat_136], 1), kwargs = {})
#   %sub_218 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_137, %unsqueeze_1745), kwargs = {})
#   %mul_655 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_218, %unsqueeze_1747), kwargs = {})
#   %mul_656 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_655, %unsqueeze_1749), kwargs = {})
#   %add_506 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_656, %unsqueeze_1751), kwargs = {})
#   %relu_218 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_506,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2560
    x2 = xindex
    y1 = (yindex // 2560)
    y3 = yindex
    tmp39 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2304*x2) + (112896*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((2176*x2) + (106624*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((2176*x2) + (106624*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 2560, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.broadcast_to((-2048) + y0, [XBLOCK, YBLOCK])
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1, 1], 384, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1, 1], 256, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + (2048 + (2304*x2) + (112896*y1) + ((-2048) + y0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp15 >= tmp20
    tmp25 = tmp24 & tmp19
    tmp26 = tl.load(in_ptr1 + (2048 + (2176*x2) + (106624*y1) + ((-256) + ((-2048) + y0))), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp21, tmp23, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp19, tmp27, tmp28)
    tmp30 = tmp15 >= tmp17
    tmp31 = tl.full([1, 1], 512, tl.int64)
    tmp32 = tmp15 < tmp31
    tmp33 = tmp30 & tmp12
    tmp34 = tl.load(in_ptr2 + (2048 + (2176*x2) + (106624*y1) + ((-384) + ((-2048) + y0))), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp18, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp12, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp11, tmp37)
    tmp40 = tmp38 - tmp39
    tmp42 = 0.001
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.sqrt(tmp43)
    tmp45 = tl.full([1, 1], 1, tl.int32)
    tmp46 = tmp45 / tmp44
    tmp47 = 1.0
    tmp48 = tmp46 * tmp47
    tmp49 = tmp40 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tmp54 = tl.full([1, 1], 0, tl.int32)
    tmp55 = triton_helpers.maximum(tmp54, tmp53)
    tl.store(out_ptr1 + (y0 + (2560*x2) + (125440*y1)), tmp55, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cx/ccxxitviuwzkacmg2uxcqzfysqxvdmy7kibbfw7hdud3v6u6wnda.py
# Topologically Sorted Source Nodes: [dense_69], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   dense_69 => cat_138
# Graph fragment:
#   %cat_138 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_136, %slice_622], 1), kwargs = {})
triton_poi_fused_cat_63 = async_compile.triton('triton_poi_fused_cat_63', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_63(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 640
    x0 = xindex % 49
    x2 = (xindex // 31360)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 384, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 256, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (2048 + (2304*x0) + (112896*x2) + x1), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp0 >= tmp8
    tmp13 = tmp12 & tmp7
    tmp14 = tl.load(in_ptr1 + (2048 + (2176*x0) + (106624*x2) + ((-256) + x1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp11, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp7, tmp15, tmp16)
    tmp18 = tmp0 >= tmp5
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + (2048 + (2176*x0) + (106624*x2) + ((-384) + x1)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp6, tmp17, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 640, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr3 + (2048 + (2176*x0) + (106624*x2) + ((-512) + x1)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wa/cwayenptab5hia7yzwkbfobazel6fg5eex44zg4b3faunogtf2nl.py
# Topologically Sorted Source Nodes: [x_448, x_449, x_450, x_451], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   x_448 => cat_139
#   x_449 => add_513, mul_664, mul_665, sub_221
#   x_450 => relu_221
#   x_451 => mean_1
# Graph fragment:
#   %cat_139 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_511, %cat_138], 1), kwargs = {})
#   %sub_221 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_139, %unsqueeze_1769), kwargs = {})
#   %mul_664 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_221, %unsqueeze_1771), kwargs = {})
#   %mul_665 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_664, %unsqueeze_1773), kwargs = {})
#   %add_513 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_665, %unsqueeze_1775), kwargs = {})
#   %relu_221 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_513,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_221, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_cat_mean_relu_64 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_cat_mean_relu_64', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_mean_relu_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_cat_mean_relu_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 21504
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 2688
    r2 = rindex
    x1 = (xindex // 2688)
    x3 = xindex
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2304*r2) + (112896*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((2176*r2) + (106624*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((2176*r2) + (106624*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((2176*r2) + (106624*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 2688, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr4 + (r2 + (49*((-2048) + x0)) + (31360*x1)), rmask & tmp14 & xmask, other=0.0)
    tmp18 = tl.where(tmp4, tmp13, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 0.001
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tl.full([1, 1], 1, tl.int32)
    tmp26 = tmp25 / tmp24
    tmp27 = 1.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp20 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1, 1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp38 = tl.where(rmask & xmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None]
    tmp40 = 49.0
    tmp41 = tmp39 / tmp40
    tl.store(out_ptr2 + (x3), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ts/ctsh4uuqtchsqicolpdsk7fd6zqezwqbs3yskecvh3gseqojfsjm.py
# Topologically Sorted Source Nodes: [x_449, x_450, x_451, x_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_449 => add_513
#   x_450 => relu_221
#   x_451 => mean_1
#   x_452 => convolution_221
# Graph fragment:
#   %add_513 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_665, %unsqueeze_1775), kwargs = {})
#   %relu_221 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_513,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_221, [-1, -2], True), kwargs = {})
#   %convolution_221 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_1, %arg555_1, %arg556_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1000
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (296, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (200, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg16_1, (200, ), (1, ))
    assert_size_stride(arg17_1, (200, ), (1, ))
    assert_size_stride(arg18_1, (200, ), (1, ))
    assert_size_stride(arg19_1, (200, ), (1, ))
    assert_size_stride(arg20_1, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg21_1, (200, ), (1, ))
    assert_size_stride(arg22_1, (200, ), (1, ))
    assert_size_stride(arg23_1, (200, ), (1, ))
    assert_size_stride(arg24_1, (200, ), (1, ))
    assert_size_stride(arg25_1, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg26_1, (316, ), (1, ))
    assert_size_stride(arg27_1, (316, ), (1, ))
    assert_size_stride(arg28_1, (316, ), (1, ))
    assert_size_stride(arg29_1, (316, ), (1, ))
    assert_size_stride(arg30_1, (200, 316, 1, 1), (316, 1, 1, 1))
    assert_size_stride(arg31_1, (200, ), (1, ))
    assert_size_stride(arg32_1, (200, ), (1, ))
    assert_size_stride(arg33_1, (200, ), (1, ))
    assert_size_stride(arg34_1, (200, ), (1, ))
    assert_size_stride(arg35_1, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg36_1, (200, ), (1, ))
    assert_size_stride(arg37_1, (200, ), (1, ))
    assert_size_stride(arg38_1, (200, ), (1, ))
    assert_size_stride(arg39_1, (200, ), (1, ))
    assert_size_stride(arg40_1, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg41_1, (336, ), (1, ))
    assert_size_stride(arg42_1, (336, ), (1, ))
    assert_size_stride(arg43_1, (336, ), (1, ))
    assert_size_stride(arg44_1, (336, ), (1, ))
    assert_size_stride(arg45_1, (200, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg46_1, (200, ), (1, ))
    assert_size_stride(arg47_1, (200, ), (1, ))
    assert_size_stride(arg48_1, (200, ), (1, ))
    assert_size_stride(arg49_1, (200, ), (1, ))
    assert_size_stride(arg50_1, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg51_1, (200, ), (1, ))
    assert_size_stride(arg52_1, (200, ), (1, ))
    assert_size_stride(arg53_1, (200, ), (1, ))
    assert_size_stride(arg54_1, (200, ), (1, ))
    assert_size_stride(arg55_1, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg56_1, (356, ), (1, ))
    assert_size_stride(arg57_1, (356, ), (1, ))
    assert_size_stride(arg58_1, (356, ), (1, ))
    assert_size_stride(arg59_1, (356, ), (1, ))
    assert_size_stride(arg60_1, (200, 356, 1, 1), (356, 1, 1, 1))
    assert_size_stride(arg61_1, (200, ), (1, ))
    assert_size_stride(arg62_1, (200, ), (1, ))
    assert_size_stride(arg63_1, (200, ), (1, ))
    assert_size_stride(arg64_1, (200, ), (1, ))
    assert_size_stride(arg65_1, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg66_1, (200, ), (1, ))
    assert_size_stride(arg67_1, (200, ), (1, ))
    assert_size_stride(arg68_1, (200, ), (1, ))
    assert_size_stride(arg69_1, (200, ), (1, ))
    assert_size_stride(arg70_1, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg71_1, (376, ), (1, ))
    assert_size_stride(arg72_1, (376, ), (1, ))
    assert_size_stride(arg73_1, (376, ), (1, ))
    assert_size_stride(arg74_1, (376, ), (1, ))
    assert_size_stride(arg75_1, (640, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(arg76_1, (376, ), (1, ))
    assert_size_stride(arg77_1, (376, ), (1, ))
    assert_size_stride(arg78_1, (376, ), (1, ))
    assert_size_stride(arg79_1, (376, ), (1, ))
    assert_size_stride(arg80_1, (400, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(arg81_1, (400, ), (1, ))
    assert_size_stride(arg82_1, (400, ), (1, ))
    assert_size_stride(arg83_1, (400, ), (1, ))
    assert_size_stride(arg84_1, (400, ), (1, ))
    assert_size_stride(arg85_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg86_1, (400, ), (1, ))
    assert_size_stride(arg87_1, (400, ), (1, ))
    assert_size_stride(arg88_1, (400, ), (1, ))
    assert_size_stride(arg89_1, (400, ), (1, ))
    assert_size_stride(arg90_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg91_1, (704, ), (1, ))
    assert_size_stride(arg92_1, (704, ), (1, ))
    assert_size_stride(arg93_1, (704, ), (1, ))
    assert_size_stride(arg94_1, (704, ), (1, ))
    assert_size_stride(arg95_1, (400, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(arg96_1, (400, ), (1, ))
    assert_size_stride(arg97_1, (400, ), (1, ))
    assert_size_stride(arg98_1, (400, ), (1, ))
    assert_size_stride(arg99_1, (400, ), (1, ))
    assert_size_stride(arg100_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg101_1, (400, ), (1, ))
    assert_size_stride(arg102_1, (400, ), (1, ))
    assert_size_stride(arg103_1, (400, ), (1, ))
    assert_size_stride(arg104_1, (400, ), (1, ))
    assert_size_stride(arg105_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (400, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg111_1, (400, ), (1, ))
    assert_size_stride(arg112_1, (400, ), (1, ))
    assert_size_stride(arg113_1, (400, ), (1, ))
    assert_size_stride(arg114_1, (400, ), (1, ))
    assert_size_stride(arg115_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg116_1, (400, ), (1, ))
    assert_size_stride(arg117_1, (400, ), (1, ))
    assert_size_stride(arg118_1, (400, ), (1, ))
    assert_size_stride(arg119_1, (400, ), (1, ))
    assert_size_stride(arg120_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg121_1, (832, ), (1, ))
    assert_size_stride(arg122_1, (832, ), (1, ))
    assert_size_stride(arg123_1, (832, ), (1, ))
    assert_size_stride(arg124_1, (832, ), (1, ))
    assert_size_stride(arg125_1, (400, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg126_1, (400, ), (1, ))
    assert_size_stride(arg127_1, (400, ), (1, ))
    assert_size_stride(arg128_1, (400, ), (1, ))
    assert_size_stride(arg129_1, (400, ), (1, ))
    assert_size_stride(arg130_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg131_1, (400, ), (1, ))
    assert_size_stride(arg132_1, (400, ), (1, ))
    assert_size_stride(arg133_1, (400, ), (1, ))
    assert_size_stride(arg134_1, (400, ), (1, ))
    assert_size_stride(arg135_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg136_1, (896, ), (1, ))
    assert_size_stride(arg137_1, (896, ), (1, ))
    assert_size_stride(arg138_1, (896, ), (1, ))
    assert_size_stride(arg139_1, (896, ), (1, ))
    assert_size_stride(arg140_1, (400, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg141_1, (400, ), (1, ))
    assert_size_stride(arg142_1, (400, ), (1, ))
    assert_size_stride(arg143_1, (400, ), (1, ))
    assert_size_stride(arg144_1, (400, ), (1, ))
    assert_size_stride(arg145_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg146_1, (400, ), (1, ))
    assert_size_stride(arg147_1, (400, ), (1, ))
    assert_size_stride(arg148_1, (400, ), (1, ))
    assert_size_stride(arg149_1, (400, ), (1, ))
    assert_size_stride(arg150_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg151_1, (960, ), (1, ))
    assert_size_stride(arg152_1, (960, ), (1, ))
    assert_size_stride(arg153_1, (960, ), (1, ))
    assert_size_stride(arg154_1, (960, ), (1, ))
    assert_size_stride(arg155_1, (400, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg156_1, (400, ), (1, ))
    assert_size_stride(arg157_1, (400, ), (1, ))
    assert_size_stride(arg158_1, (400, ), (1, ))
    assert_size_stride(arg159_1, (400, ), (1, ))
    assert_size_stride(arg160_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg161_1, (400, ), (1, ))
    assert_size_stride(arg162_1, (400, ), (1, ))
    assert_size_stride(arg163_1, (400, ), (1, ))
    assert_size_stride(arg164_1, (400, ), (1, ))
    assert_size_stride(arg165_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (400, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg171_1, (400, ), (1, ))
    assert_size_stride(arg172_1, (400, ), (1, ))
    assert_size_stride(arg173_1, (400, ), (1, ))
    assert_size_stride(arg174_1, (400, ), (1, ))
    assert_size_stride(arg175_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg176_1, (400, ), (1, ))
    assert_size_stride(arg177_1, (400, ), (1, ))
    assert_size_stride(arg178_1, (400, ), (1, ))
    assert_size_stride(arg179_1, (400, ), (1, ))
    assert_size_stride(arg180_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg181_1, (1088, ), (1, ))
    assert_size_stride(arg182_1, (1088, ), (1, ))
    assert_size_stride(arg183_1, (1088, ), (1, ))
    assert_size_stride(arg184_1, (1088, ), (1, ))
    assert_size_stride(arg185_1, (400, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(arg186_1, (400, ), (1, ))
    assert_size_stride(arg187_1, (400, ), (1, ))
    assert_size_stride(arg188_1, (400, ), (1, ))
    assert_size_stride(arg189_1, (400, ), (1, ))
    assert_size_stride(arg190_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg191_1, (400, ), (1, ))
    assert_size_stride(arg192_1, (400, ), (1, ))
    assert_size_stride(arg193_1, (400, ), (1, ))
    assert_size_stride(arg194_1, (400, ), (1, ))
    assert_size_stride(arg195_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg196_1, (1152, ), (1, ))
    assert_size_stride(arg197_1, (1152, ), (1, ))
    assert_size_stride(arg198_1, (1152, ), (1, ))
    assert_size_stride(arg199_1, (1152, ), (1, ))
    assert_size_stride(arg200_1, (1152, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg201_1, (1152, ), (1, ))
    assert_size_stride(arg202_1, (1152, ), (1, ))
    assert_size_stride(arg203_1, (1152, ), (1, ))
    assert_size_stride(arg204_1, (1152, ), (1, ))
    assert_size_stride(arg205_1, (800, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg206_1, (800, ), (1, ))
    assert_size_stride(arg207_1, (800, ), (1, ))
    assert_size_stride(arg208_1, (800, ), (1, ))
    assert_size_stride(arg209_1, (800, ), (1, ))
    assert_size_stride(arg210_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg211_1, (800, ), (1, ))
    assert_size_stride(arg212_1, (800, ), (1, ))
    assert_size_stride(arg213_1, (800, ), (1, ))
    assert_size_stride(arg214_1, (800, ), (1, ))
    assert_size_stride(arg215_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg216_1, (1216, ), (1, ))
    assert_size_stride(arg217_1, (1216, ), (1, ))
    assert_size_stride(arg218_1, (1216, ), (1, ))
    assert_size_stride(arg219_1, (1216, ), (1, ))
    assert_size_stride(arg220_1, (800, 1216, 1, 1), (1216, 1, 1, 1))
    assert_size_stride(arg221_1, (800, ), (1, ))
    assert_size_stride(arg222_1, (800, ), (1, ))
    assert_size_stride(arg223_1, (800, ), (1, ))
    assert_size_stride(arg224_1, (800, ), (1, ))
    assert_size_stride(arg225_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg226_1, (800, ), (1, ))
    assert_size_stride(arg227_1, (800, ), (1, ))
    assert_size_stride(arg228_1, (800, ), (1, ))
    assert_size_stride(arg229_1, (800, ), (1, ))
    assert_size_stride(arg230_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg231_1, (1280, ), (1, ))
    assert_size_stride(arg232_1, (1280, ), (1, ))
    assert_size_stride(arg233_1, (1280, ), (1, ))
    assert_size_stride(arg234_1, (1280, ), (1, ))
    assert_size_stride(arg235_1, (800, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg236_1, (800, ), (1, ))
    assert_size_stride(arg237_1, (800, ), (1, ))
    assert_size_stride(arg238_1, (800, ), (1, ))
    assert_size_stride(arg239_1, (800, ), (1, ))
    assert_size_stride(arg240_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg241_1, (800, ), (1, ))
    assert_size_stride(arg242_1, (800, ), (1, ))
    assert_size_stride(arg243_1, (800, ), (1, ))
    assert_size_stride(arg244_1, (800, ), (1, ))
    assert_size_stride(arg245_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg246_1, (1344, ), (1, ))
    assert_size_stride(arg247_1, (1344, ), (1, ))
    assert_size_stride(arg248_1, (1344, ), (1, ))
    assert_size_stride(arg249_1, (1344, ), (1, ))
    assert_size_stride(arg250_1, (800, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(arg251_1, (800, ), (1, ))
    assert_size_stride(arg252_1, (800, ), (1, ))
    assert_size_stride(arg253_1, (800, ), (1, ))
    assert_size_stride(arg254_1, (800, ), (1, ))
    assert_size_stride(arg255_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg256_1, (800, ), (1, ))
    assert_size_stride(arg257_1, (800, ), (1, ))
    assert_size_stride(arg258_1, (800, ), (1, ))
    assert_size_stride(arg259_1, (800, ), (1, ))
    assert_size_stride(arg260_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg261_1, (1408, ), (1, ))
    assert_size_stride(arg262_1, (1408, ), (1, ))
    assert_size_stride(arg263_1, (1408, ), (1, ))
    assert_size_stride(arg264_1, (1408, ), (1, ))
    assert_size_stride(arg265_1, (800, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(arg266_1, (800, ), (1, ))
    assert_size_stride(arg267_1, (800, ), (1, ))
    assert_size_stride(arg268_1, (800, ), (1, ))
    assert_size_stride(arg269_1, (800, ), (1, ))
    assert_size_stride(arg270_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg271_1, (800, ), (1, ))
    assert_size_stride(arg272_1, (800, ), (1, ))
    assert_size_stride(arg273_1, (800, ), (1, ))
    assert_size_stride(arg274_1, (800, ), (1, ))
    assert_size_stride(arg275_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg276_1, (1472, ), (1, ))
    assert_size_stride(arg277_1, (1472, ), (1, ))
    assert_size_stride(arg278_1, (1472, ), (1, ))
    assert_size_stride(arg279_1, (1472, ), (1, ))
    assert_size_stride(arg280_1, (800, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(arg281_1, (800, ), (1, ))
    assert_size_stride(arg282_1, (800, ), (1, ))
    assert_size_stride(arg283_1, (800, ), (1, ))
    assert_size_stride(arg284_1, (800, ), (1, ))
    assert_size_stride(arg285_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg286_1, (800, ), (1, ))
    assert_size_stride(arg287_1, (800, ), (1, ))
    assert_size_stride(arg288_1, (800, ), (1, ))
    assert_size_stride(arg289_1, (800, ), (1, ))
    assert_size_stride(arg290_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg291_1, (1536, ), (1, ))
    assert_size_stride(arg292_1, (1536, ), (1, ))
    assert_size_stride(arg293_1, (1536, ), (1, ))
    assert_size_stride(arg294_1, (1536, ), (1, ))
    assert_size_stride(arg295_1, (800, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg296_1, (800, ), (1, ))
    assert_size_stride(arg297_1, (800, ), (1, ))
    assert_size_stride(arg298_1, (800, ), (1, ))
    assert_size_stride(arg299_1, (800, ), (1, ))
    assert_size_stride(arg300_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg301_1, (800, ), (1, ))
    assert_size_stride(arg302_1, (800, ), (1, ))
    assert_size_stride(arg303_1, (800, ), (1, ))
    assert_size_stride(arg304_1, (800, ), (1, ))
    assert_size_stride(arg305_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg306_1, (1600, ), (1, ))
    assert_size_stride(arg307_1, (1600, ), (1, ))
    assert_size_stride(arg308_1, (1600, ), (1, ))
    assert_size_stride(arg309_1, (1600, ), (1, ))
    assert_size_stride(arg310_1, (800, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(arg311_1, (800, ), (1, ))
    assert_size_stride(arg312_1, (800, ), (1, ))
    assert_size_stride(arg313_1, (800, ), (1, ))
    assert_size_stride(arg314_1, (800, ), (1, ))
    assert_size_stride(arg315_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg316_1, (800, ), (1, ))
    assert_size_stride(arg317_1, (800, ), (1, ))
    assert_size_stride(arg318_1, (800, ), (1, ))
    assert_size_stride(arg319_1, (800, ), (1, ))
    assert_size_stride(arg320_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg321_1, (1664, ), (1, ))
    assert_size_stride(arg322_1, (1664, ), (1, ))
    assert_size_stride(arg323_1, (1664, ), (1, ))
    assert_size_stride(arg324_1, (1664, ), (1, ))
    assert_size_stride(arg325_1, (800, 1664, 1, 1), (1664, 1, 1, 1))
    assert_size_stride(arg326_1, (800, ), (1, ))
    assert_size_stride(arg327_1, (800, ), (1, ))
    assert_size_stride(arg328_1, (800, ), (1, ))
    assert_size_stride(arg329_1, (800, ), (1, ))
    assert_size_stride(arg330_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg331_1, (800, ), (1, ))
    assert_size_stride(arg332_1, (800, ), (1, ))
    assert_size_stride(arg333_1, (800, ), (1, ))
    assert_size_stride(arg334_1, (800, ), (1, ))
    assert_size_stride(arg335_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg336_1, (1728, ), (1, ))
    assert_size_stride(arg337_1, (1728, ), (1, ))
    assert_size_stride(arg338_1, (1728, ), (1, ))
    assert_size_stride(arg339_1, (1728, ), (1, ))
    assert_size_stride(arg340_1, (800, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(arg341_1, (800, ), (1, ))
    assert_size_stride(arg342_1, (800, ), (1, ))
    assert_size_stride(arg343_1, (800, ), (1, ))
    assert_size_stride(arg344_1, (800, ), (1, ))
    assert_size_stride(arg345_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg346_1, (800, ), (1, ))
    assert_size_stride(arg347_1, (800, ), (1, ))
    assert_size_stride(arg348_1, (800, ), (1, ))
    assert_size_stride(arg349_1, (800, ), (1, ))
    assert_size_stride(arg350_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg351_1, (1792, ), (1, ))
    assert_size_stride(arg352_1, (1792, ), (1, ))
    assert_size_stride(arg353_1, (1792, ), (1, ))
    assert_size_stride(arg354_1, (1792, ), (1, ))
    assert_size_stride(arg355_1, (800, 1792, 1, 1), (1792, 1, 1, 1))
    assert_size_stride(arg356_1, (800, ), (1, ))
    assert_size_stride(arg357_1, (800, ), (1, ))
    assert_size_stride(arg358_1, (800, ), (1, ))
    assert_size_stride(arg359_1, (800, ), (1, ))
    assert_size_stride(arg360_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg361_1, (800, ), (1, ))
    assert_size_stride(arg362_1, (800, ), (1, ))
    assert_size_stride(arg363_1, (800, ), (1, ))
    assert_size_stride(arg364_1, (800, ), (1, ))
    assert_size_stride(arg365_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg366_1, (1856, ), (1, ))
    assert_size_stride(arg367_1, (1856, ), (1, ))
    assert_size_stride(arg368_1, (1856, ), (1, ))
    assert_size_stride(arg369_1, (1856, ), (1, ))
    assert_size_stride(arg370_1, (800, 1856, 1, 1), (1856, 1, 1, 1))
    assert_size_stride(arg371_1, (800, ), (1, ))
    assert_size_stride(arg372_1, (800, ), (1, ))
    assert_size_stride(arg373_1, (800, ), (1, ))
    assert_size_stride(arg374_1, (800, ), (1, ))
    assert_size_stride(arg375_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg376_1, (800, ), (1, ))
    assert_size_stride(arg377_1, (800, ), (1, ))
    assert_size_stride(arg378_1, (800, ), (1, ))
    assert_size_stride(arg379_1, (800, ), (1, ))
    assert_size_stride(arg380_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg381_1, (1920, ), (1, ))
    assert_size_stride(arg382_1, (1920, ), (1, ))
    assert_size_stride(arg383_1, (1920, ), (1, ))
    assert_size_stride(arg384_1, (1920, ), (1, ))
    assert_size_stride(arg385_1, (800, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg386_1, (800, ), (1, ))
    assert_size_stride(arg387_1, (800, ), (1, ))
    assert_size_stride(arg388_1, (800, ), (1, ))
    assert_size_stride(arg389_1, (800, ), (1, ))
    assert_size_stride(arg390_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg391_1, (800, ), (1, ))
    assert_size_stride(arg392_1, (800, ), (1, ))
    assert_size_stride(arg393_1, (800, ), (1, ))
    assert_size_stride(arg394_1, (800, ), (1, ))
    assert_size_stride(arg395_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg396_1, (1984, ), (1, ))
    assert_size_stride(arg397_1, (1984, ), (1, ))
    assert_size_stride(arg398_1, (1984, ), (1, ))
    assert_size_stride(arg399_1, (1984, ), (1, ))
    assert_size_stride(arg400_1, (800, 1984, 1, 1), (1984, 1, 1, 1))
    assert_size_stride(arg401_1, (800, ), (1, ))
    assert_size_stride(arg402_1, (800, ), (1, ))
    assert_size_stride(arg403_1, (800, ), (1, ))
    assert_size_stride(arg404_1, (800, ), (1, ))
    assert_size_stride(arg405_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg406_1, (800, ), (1, ))
    assert_size_stride(arg407_1, (800, ), (1, ))
    assert_size_stride(arg408_1, (800, ), (1, ))
    assert_size_stride(arg409_1, (800, ), (1, ))
    assert_size_stride(arg410_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg411_1, (2048, ), (1, ))
    assert_size_stride(arg412_1, (2048, ), (1, ))
    assert_size_stride(arg413_1, (2048, ), (1, ))
    assert_size_stride(arg414_1, (2048, ), (1, ))
    assert_size_stride(arg415_1, (800, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg416_1, (800, ), (1, ))
    assert_size_stride(arg417_1, (800, ), (1, ))
    assert_size_stride(arg418_1, (800, ), (1, ))
    assert_size_stride(arg419_1, (800, ), (1, ))
    assert_size_stride(arg420_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg421_1, (800, ), (1, ))
    assert_size_stride(arg422_1, (800, ), (1, ))
    assert_size_stride(arg423_1, (800, ), (1, ))
    assert_size_stride(arg424_1, (800, ), (1, ))
    assert_size_stride(arg425_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg426_1, (2112, ), (1, ))
    assert_size_stride(arg427_1, (2112, ), (1, ))
    assert_size_stride(arg428_1, (2112, ), (1, ))
    assert_size_stride(arg429_1, (2112, ), (1, ))
    assert_size_stride(arg430_1, (800, 2112, 1, 1), (2112, 1, 1, 1))
    assert_size_stride(arg431_1, (800, ), (1, ))
    assert_size_stride(arg432_1, (800, ), (1, ))
    assert_size_stride(arg433_1, (800, ), (1, ))
    assert_size_stride(arg434_1, (800, ), (1, ))
    assert_size_stride(arg435_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg436_1, (800, ), (1, ))
    assert_size_stride(arg437_1, (800, ), (1, ))
    assert_size_stride(arg438_1, (800, ), (1, ))
    assert_size_stride(arg439_1, (800, ), (1, ))
    assert_size_stride(arg440_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg441_1, (2176, ), (1, ))
    assert_size_stride(arg442_1, (2176, ), (1, ))
    assert_size_stride(arg443_1, (2176, ), (1, ))
    assert_size_stride(arg444_1, (2176, ), (1, ))
    assert_size_stride(arg445_1, (800, 2176, 1, 1), (2176, 1, 1, 1))
    assert_size_stride(arg446_1, (800, ), (1, ))
    assert_size_stride(arg447_1, (800, ), (1, ))
    assert_size_stride(arg448_1, (800, ), (1, ))
    assert_size_stride(arg449_1, (800, ), (1, ))
    assert_size_stride(arg450_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg451_1, (800, ), (1, ))
    assert_size_stride(arg452_1, (800, ), (1, ))
    assert_size_stride(arg453_1, (800, ), (1, ))
    assert_size_stride(arg454_1, (800, ), (1, ))
    assert_size_stride(arg455_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg456_1, (2240, ), (1, ))
    assert_size_stride(arg457_1, (2240, ), (1, ))
    assert_size_stride(arg458_1, (2240, ), (1, ))
    assert_size_stride(arg459_1, (2240, ), (1, ))
    assert_size_stride(arg460_1, (800, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(arg461_1, (800, ), (1, ))
    assert_size_stride(arg462_1, (800, ), (1, ))
    assert_size_stride(arg463_1, (800, ), (1, ))
    assert_size_stride(arg464_1, (800, ), (1, ))
    assert_size_stride(arg465_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg466_1, (800, ), (1, ))
    assert_size_stride(arg467_1, (800, ), (1, ))
    assert_size_stride(arg468_1, (800, ), (1, ))
    assert_size_stride(arg469_1, (800, ), (1, ))
    assert_size_stride(arg470_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg471_1, (2304, ), (1, ))
    assert_size_stride(arg472_1, (2304, ), (1, ))
    assert_size_stride(arg473_1, (2304, ), (1, ))
    assert_size_stride(arg474_1, (2304, ), (1, ))
    assert_size_stride(arg475_1, (800, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(arg476_1, (800, ), (1, ))
    assert_size_stride(arg477_1, (800, ), (1, ))
    assert_size_stride(arg478_1, (800, ), (1, ))
    assert_size_stride(arg479_1, (800, ), (1, ))
    assert_size_stride(arg480_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg481_1, (800, ), (1, ))
    assert_size_stride(arg482_1, (800, ), (1, ))
    assert_size_stride(arg483_1, (800, ), (1, ))
    assert_size_stride(arg484_1, (800, ), (1, ))
    assert_size_stride(arg485_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg486_1, (2368, ), (1, ))
    assert_size_stride(arg487_1, (2368, ), (1, ))
    assert_size_stride(arg488_1, (2368, ), (1, ))
    assert_size_stride(arg489_1, (2368, ), (1, ))
    assert_size_stride(arg490_1, (800, 2368, 1, 1), (2368, 1, 1, 1))
    assert_size_stride(arg491_1, (800, ), (1, ))
    assert_size_stride(arg492_1, (800, ), (1, ))
    assert_size_stride(arg493_1, (800, ), (1, ))
    assert_size_stride(arg494_1, (800, ), (1, ))
    assert_size_stride(arg495_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg496_1, (800, ), (1, ))
    assert_size_stride(arg497_1, (800, ), (1, ))
    assert_size_stride(arg498_1, (800, ), (1, ))
    assert_size_stride(arg499_1, (800, ), (1, ))
    assert_size_stride(arg500_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg501_1, (2432, ), (1, ))
    assert_size_stride(arg502_1, (2432, ), (1, ))
    assert_size_stride(arg503_1, (2432, ), (1, ))
    assert_size_stride(arg504_1, (2432, ), (1, ))
    assert_size_stride(arg505_1, (2304, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(arg506_1, (2432, ), (1, ))
    assert_size_stride(arg507_1, (2432, ), (1, ))
    assert_size_stride(arg508_1, (2432, ), (1, ))
    assert_size_stride(arg509_1, (2432, ), (1, ))
    assert_size_stride(arg510_1, (1600, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(arg511_1, (1600, ), (1, ))
    assert_size_stride(arg512_1, (1600, ), (1, ))
    assert_size_stride(arg513_1, (1600, ), (1, ))
    assert_size_stride(arg514_1, (1600, ), (1, ))
    assert_size_stride(arg515_1, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg516_1, (1600, ), (1, ))
    assert_size_stride(arg517_1, (1600, ), (1, ))
    assert_size_stride(arg518_1, (1600, ), (1, ))
    assert_size_stride(arg519_1, (1600, ), (1, ))
    assert_size_stride(arg520_1, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(arg521_1, (2432, ), (1, ))
    assert_size_stride(arg522_1, (2432, ), (1, ))
    assert_size_stride(arg523_1, (2432, ), (1, ))
    assert_size_stride(arg524_1, (2432, ), (1, ))
    assert_size_stride(arg525_1, (1600, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(arg526_1, (1600, ), (1, ))
    assert_size_stride(arg527_1, (1600, ), (1, ))
    assert_size_stride(arg528_1, (1600, ), (1, ))
    assert_size_stride(arg529_1, (1600, ), (1, ))
    assert_size_stride(arg530_1, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg531_1, (1600, ), (1, ))
    assert_size_stride(arg532_1, (1600, ), (1, ))
    assert_size_stride(arg533_1, (1600, ), (1, ))
    assert_size_stride(arg534_1, (1600, ), (1, ))
    assert_size_stride(arg535_1, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(arg536_1, (2560, ), (1, ))
    assert_size_stride(arg537_1, (2560, ), (1, ))
    assert_size_stride(arg538_1, (2560, ), (1, ))
    assert_size_stride(arg539_1, (2560, ), (1, ))
    assert_size_stride(arg540_1, (1600, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(arg541_1, (1600, ), (1, ))
    assert_size_stride(arg542_1, (1600, ), (1, ))
    assert_size_stride(arg543_1, (1600, ), (1, ))
    assert_size_stride(arg544_1, (1600, ), (1, ))
    assert_size_stride(arg545_1, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg546_1, (1600, ), (1, ))
    assert_size_stride(arg547_1, (1600, ), (1, ))
    assert_size_stride(arg548_1, (1600, ), (1, ))
    assert_size_stride(arg549_1, (1600, ), (1, ))
    assert_size_stride(arg550_1, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(arg551_1, (2688, ), (1, ))
    assert_size_stride(arg552_1, (2688, ), (1, ))
    assert_size_stride(arg553_1, (2688, ), (1, ))
    assert_size_stride(arg554_1, (2688, ), (1, ))
    assert_size_stride(arg555_1, (1000, 2688, 1, 1), (2688, 1, 1, 1))
    assert_size_stride(arg556_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_227], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((128, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_227], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 384, 49, grid=grid(384, 49), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 128, 112, 112), (1605632, 1, 14336, 128))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_228, x_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf5 = empty_strided_cuda((8, 128, 56, 56), (401408, 1, 7168, 128), torch.float32)
        buf7 = empty_strided_cuda((8, 128, 56, 56), (401408, 1, 7168, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_228, x_229, input_2, x_230, x_231, x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.run(buf3, arg6_1, arg7_1, arg8_1, arg9_1, arg11_1, arg12_1, arg13_1, arg14_1, buf5, buf7, 3211264, grid=grid(3211264), stream=stream0)
        del arg11_1
        del arg12_1
        del arg13_1
        del arg14_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
        del buf3
        # Topologically Sorted Source Nodes: [x_230, x_231, x_s_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg10_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 296, 56, 56), (928256, 1, 16576, 296))
        del arg10_1
        # Topologically Sorted Source Nodes: [x_232, x_233, x_in_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 200, 56, 56), (627200, 1, 11200, 200))
        del arg15_1
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf9, arg16_1, arg17_1, arg18_1, arg19_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        del arg19_1
        buf10 = empty_strided_cuda((200, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Topologically Sorted Source Nodes: [x_234, x_235, x_in_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg20_1, buf10, 800, 9, grid=grid(800, 9), stream=stream0)
        del arg20_1
        # Topologically Sorted Source Nodes: [x_234, x_235, x_in_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf11 = extern_kernels.convolution(buf9, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf11, (8, 200, 56, 56), (627200, 1, 11200, 200))
        del buf9
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_236, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf12, arg21_1, arg22_1, arg23_1, arg24_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg21_1
        del arg22_1
        del arg23_1
        del arg24_1
        # Topologically Sorted Source Nodes: [x_236, x_237, x_in_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg25_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 276, 56, 56), (865536, 1, 15456, 276))
        del arg25_1
        del buf12
        buf15 = empty_strided_cuda((8, 316, 56, 56), (990976, 1, 17696, 316), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_142, x_238, x_239], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6.run(buf6, buf13, arg26_1, arg27_1, arg28_1, arg29_1, buf15, 2528, 3136, grid=grid(2528, 3136), stream=stream0)
        del arg26_1
        del arg27_1
        del arg28_1
        del arg29_1
        # Topologically Sorted Source Nodes: [x_238, x_239, x_in_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 200, 56, 56), (627200, 1, 11200, 200))
        del arg30_1
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_240, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf17, arg31_1, arg32_1, arg33_1, arg34_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg31_1
        del arg32_1
        del arg33_1
        del arg34_1
        buf18 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_240, x_241, x_in_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg35_1, buf18, 800, 9, grid=grid(800, 9), stream=stream0)
        del arg35_1
        # Topologically Sorted Source Nodes: [x_240, x_241, x_in_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf19, (8, 200, 56, 56), (627200, 1, 11200, 200))
        del buf17
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_242, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf20, arg36_1, arg37_1, arg38_1, arg39_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg36_1
        del arg37_1
        del arg38_1
        del arg39_1
        # Topologically Sorted Source Nodes: [x_242, x_243, x_in_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg40_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 276, 56, 56), (865536, 1, 15456, 276))
        del arg40_1
        del buf20
        buf23 = empty_strided_cuda((8, 336, 56, 56), (1053696, 1, 18816, 336), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_146, x_244, x_245], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7.run(buf6, buf13, buf21, arg41_1, arg42_1, arg43_1, arg44_1, buf23, 2688, 3136, grid=grid(2688, 3136), stream=stream0)
        del arg41_1
        del arg42_1
        del arg43_1
        del arg44_1
        # Topologically Sorted Source Nodes: [x_244, x_245, x_in_147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg45_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 200, 56, 56), (627200, 1, 11200, 200))
        del arg45_1
        del buf23
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_246, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf25, arg46_1, arg47_1, arg48_1, arg49_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg46_1
        del arg47_1
        del arg48_1
        del arg49_1
        buf26 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_246, x_247, x_in_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg50_1, buf26, 800, 9, grid=grid(800, 9), stream=stream0)
        del arg50_1
        # Topologically Sorted Source Nodes: [x_246, x_247, x_in_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf27 = extern_kernels.convolution(buf25, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf27, (8, 200, 56, 56), (627200, 1, 11200, 200))
        del buf25
        buf28 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf28, arg51_1, arg52_1, arg53_1, arg54_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg51_1
        del arg52_1
        del arg53_1
        del arg54_1
        # Topologically Sorted Source Nodes: [x_248, x_249, x_in_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg55_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 276, 56, 56), (865536, 1, 15456, 276))
        del arg55_1
        del buf28
        buf30 = empty_strided_cuda((8, 100, 56, 56), (313600, 3136, 56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dense_37], Original ATen: [aten.cat]
        triton_poi_fused_cat_8.run(buf6, buf13, buf21, buf29, buf30, 2508800, grid=grid(2508800), stream=stream0)
        buf32 = empty_strided_cuda((8, 356, 56, 56), (1116416, 1, 19936, 356), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_150, x_250, x_251], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9.run(buf6, buf13, buf21, buf29, buf30, arg56_1, arg57_1, arg58_1, arg59_1, buf32, 2848, 3136, grid=grid(2848, 3136), stream=stream0)
        del arg56_1
        del arg57_1
        del arg58_1
        del arg59_1
        # Topologically Sorted Source Nodes: [x_250, x_251, x_in_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 200, 56, 56), (627200, 1, 11200, 200))
        del arg60_1
        del buf32
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_252, x_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf34, arg61_1, arg62_1, arg63_1, arg64_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        del arg64_1
        buf35 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_252, x_253, x_in_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg65_1, buf35, 800, 9, grid=grid(800, 9), stream=stream0)
        del arg65_1
        # Topologically Sorted Source Nodes: [x_252, x_253, x_in_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf36 = extern_kernels.convolution(buf34, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf36, (8, 200, 56, 56), (627200, 1, 11200, 200))
        del buf34
        del buf35
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_254, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf37, arg66_1, arg67_1, arg68_1, arg69_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg66_1
        del arg67_1
        del arg68_1
        del arg69_1
        # Topologically Sorted Source Nodes: [x_254, x_255, x_in_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg70_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 276, 56, 56), (865536, 1, 15456, 276))
        del arg70_1
        del buf37
        buf40 = empty_strided_cuda((8, 376, 56, 56), (1179136, 1, 21056, 376), torch.float32)
        buf42 = empty_strided_cuda((8, 376, 56, 56), (1179136, 1, 21056, 376), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_154, x_256, x_257, x_258, x_259], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10.run(buf6, buf13, buf21, buf29, buf38, buf30, arg71_1, arg72_1, arg73_1, arg74_1, arg76_1, arg77_1, arg78_1, arg79_1, buf40, buf42, 3008, 3136, grid=grid(3008, 3136), stream=stream0)
        del arg71_1
        del arg72_1
        del arg73_1
        del arg74_1
        del arg76_1
        del arg77_1
        del arg78_1
        del arg79_1
        del buf13
        del buf21
        del buf29
        del buf30
        del buf38
        del buf6
        # Topologically Sorted Source Nodes: [x_256, x_257, x_s_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg75_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 640, 28, 28), (501760, 1, 17920, 640))
        del arg75_1
        del buf40
        # Topologically Sorted Source Nodes: [x_258, x_259, x_in_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf43 = extern_kernels.convolution(buf42, arg80_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 400, 56, 56), (1254400, 1, 22400, 400))
        del arg80_1
        del buf42
        buf44 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_260, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf44, arg81_1, arg82_1, arg83_1, arg84_1, 10035200, grid=grid(10035200), stream=stream0)
        del arg81_1
        del arg82_1
        del arg83_1
        del arg84_1
        buf45 = empty_strided_cuda((400, 8, 3, 3), (72, 1, 24, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x_260, x_261, x_in_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(arg85_1, buf45, 3200, 9, grid=grid(3200, 9), stream=stream0)
        del arg85_1
        # Topologically Sorted Source Nodes: [x_260, x_261, x_in_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf46 = extern_kernels.convolution(buf44, buf45, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf46, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del buf44
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_262, x_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf47, arg86_1, arg87_1, arg88_1, arg89_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg86_1
        del arg87_1
        del arg88_1
        del arg89_1
        # Topologically Sorted Source Nodes: [x_262, x_263, x_in_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 576, 28, 28), (451584, 1, 16128, 576))
        del arg90_1
        del buf47
        buf50 = empty_strided_cuda((8, 704, 28, 28), (551936, 1, 19712, 704), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_158, x_264, x_265], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14.run(buf41, buf48, arg91_1, arg92_1, arg93_1, arg94_1, buf50, 5632, 784, grid=grid(5632, 784), stream=stream0)
        del arg91_1
        del arg92_1
        del arg93_1
        del arg94_1
        # Topologically Sorted Source Nodes: [x_264, x_265, x_in_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg95_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del arg95_1
        del buf50
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_266, x_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf52, arg96_1, arg97_1, arg98_1, arg99_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg96_1
        del arg97_1
        del arg98_1
        del arg99_1
        buf53 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [x_266, x_267, x_in_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(arg100_1, buf53, 3200, 9, grid=grid(3200, 9), stream=stream0)
        del arg100_1
        # Topologically Sorted Source Nodes: [x_266, x_267, x_in_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf54, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del buf52
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_268, x_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf55, arg101_1, arg102_1, arg103_1, arg104_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg101_1
        del arg102_1
        del arg103_1
        del arg104_1
        # Topologically Sorted Source Nodes: [x_268, x_269, x_in_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 576, 28, 28), (451584, 1, 16128, 576))
        del arg105_1
        del buf55
        buf58 = empty_strided_cuda((8, 768, 28, 28), (602112, 1, 21504, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_162, x_270, x_271], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15.run(buf41, buf48, buf56, arg106_1, arg107_1, arg108_1, arg109_1, buf58, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg106_1
        del arg107_1
        del arg108_1
        del arg109_1
        # Topologically Sorted Source Nodes: [x_270, x_271, x_in_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del arg110_1
        del buf58
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf60, arg111_1, arg112_1, arg113_1, arg114_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg111_1
        del arg112_1
        del arg113_1
        del arg114_1
        buf61 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [x_272, x_273, x_in_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(arg115_1, buf61, 3200, 9, grid=grid(3200, 9), stream=stream0)
        del arg115_1
        # Topologically Sorted Source Nodes: [x_272, x_273, x_in_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf62 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf62, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del buf60
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_274, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf63, arg116_1, arg117_1, arg118_1, arg119_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg116_1
        del arg117_1
        del arg118_1
        del arg119_1
        # Topologically Sorted Source Nodes: [x_274, x_275, x_in_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 576, 28, 28), (451584, 1, 16128, 576))
        del arg120_1
        del buf63
        buf65 = empty_strided_cuda((8, 320, 28, 28), (250880, 784, 28, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dense_41], Original ATen: [aten.cat]
        triton_poi_fused_cat_16.run(buf41, buf48, buf56, buf64, buf65, 2007040, grid=grid(2007040), stream=stream0)
        buf67 = empty_strided_cuda((8, 832, 28, 28), (652288, 1, 23296, 832), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_166, x_276, x_277], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17.run(buf41, buf48, buf56, buf64, buf65, arg121_1, arg122_1, arg123_1, arg124_1, buf67, 6656, 784, grid=grid(6656, 784), stream=stream0)
        del arg121_1
        del arg122_1
        del arg123_1
        del arg124_1
        # Topologically Sorted Source Nodes: [x_276, x_277, x_in_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del arg125_1
        del buf67
        buf69 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_278, x_279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf69, arg126_1, arg127_1, arg128_1, arg129_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg126_1
        del arg127_1
        del arg128_1
        del arg129_1
        buf70 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_278, x_279, x_in_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(arg130_1, buf70, 3200, 9, grid=grid(3200, 9), stream=stream0)
        del arg130_1
        # Topologically Sorted Source Nodes: [x_278, x_279, x_in_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf71 = extern_kernels.convolution(buf69, buf70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf71, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del buf69
        buf72 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_280, x_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf72, arg131_1, arg132_1, arg133_1, arg134_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg131_1
        del arg132_1
        del arg133_1
        del arg134_1
        # Topologically Sorted Source Nodes: [x_280, x_281, x_in_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 576, 28, 28), (451584, 1, 16128, 576))
        del arg135_1
        del buf72
        buf74 = reinterpret_tensor(buf7, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [resid_39, resid_40, resid_41, resid_42], Original ATen: [aten.add]
        triton_poi_fused_add_18.run(buf41, buf48, buf56, buf64, buf73, buf74, 3211264, grid=grid(3211264), stream=stream0)
        del buf41
        del buf48
        del buf56
        del buf64
        buf76 = empty_strided_cuda((8, 896, 28, 28), (702464, 1, 25088, 896), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_170, x_282, x_283], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19.run(buf74, buf65, buf73, arg136_1, arg137_1, arg138_1, arg139_1, buf76, 7168, 784, grid=grid(7168, 784), stream=stream0)
        del arg136_1
        del arg137_1
        del arg138_1
        del arg139_1
        # Topologically Sorted Source Nodes: [x_282, x_283, x_in_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del arg140_1
        del buf76
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [x_284, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf78, arg141_1, arg142_1, arg143_1, arg144_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg141_1
        del arg142_1
        del arg143_1
        del arg144_1
        buf79 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_284, x_285, x_in_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(arg145_1, buf79, 3200, 9, grid=grid(3200, 9), stream=stream0)
        del arg145_1
        # Topologically Sorted Source Nodes: [x_284, x_285, x_in_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf80 = extern_kernels.convolution(buf78, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf80, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del buf78
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_286, x_287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf81, arg146_1, arg147_1, arg148_1, arg149_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg146_1
        del arg147_1
        del arg148_1
        del arg149_1
        # Topologically Sorted Source Nodes: [x_286, x_287, x_in_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 576, 28, 28), (451584, 1, 16128, 576))
        del arg150_1
        del buf81
        buf84 = empty_strided_cuda((8, 960, 28, 28), (752640, 1, 26880, 960), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_174, x_288, x_289], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_20.run(buf74, buf82, buf65, buf73, arg151_1, arg152_1, arg153_1, arg154_1, buf84, 7680, 784, grid=grid(7680, 784), stream=stream0)
        del arg151_1
        del arg152_1
        del arg153_1
        del arg154_1
        # Topologically Sorted Source Nodes: [x_288, x_289, x_in_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf85 = extern_kernels.convolution(buf84, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del arg155_1
        del buf84
        buf86 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_290, x_291], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf86, arg156_1, arg157_1, arg158_1, arg159_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg156_1
        del arg157_1
        del arg158_1
        del arg159_1
        buf87 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [x_290, x_291, x_in_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(arg160_1, buf87, 3200, 9, grid=grid(3200, 9), stream=stream0)
        del arg160_1
        # Topologically Sorted Source Nodes: [x_290, x_291, x_in_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf88 = extern_kernels.convolution(buf86, buf87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf88, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del buf86
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf89, arg161_1, arg162_1, arg163_1, arg164_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg161_1
        del arg162_1
        del arg163_1
        del arg164_1
        # Topologically Sorted Source Nodes: [x_292, x_293, x_in_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 576, 28, 28), (451584, 1, 16128, 576))
        del arg165_1
        del buf89
        buf91 = reinterpret_tensor(buf5, (8, 512, 28, 28), (401408, 784, 28, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [dense_44], Original ATen: [aten.cat]
        triton_poi_fused_cat_21.run(buf65, buf73, buf82, buf90, buf91, 3211264, grid=grid(3211264), stream=stream0)
        del buf73
        buf93 = empty_strided_cuda((8, 1024, 28, 28), (802816, 1, 28672, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_178, x_294, x_295], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22.run(buf74, buf82, buf90, buf91, arg166_1, arg167_1, arg168_1, arg169_1, buf93, 8192, 784, grid=grid(8192, 784), stream=stream0)
        del arg166_1
        del arg167_1
        del arg168_1
        del arg169_1
        # Topologically Sorted Source Nodes: [x_295, x_in_179], Original ATen: [aten.relu, aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del arg170_1
        del buf93
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_296, x_297], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf95, arg171_1, arg172_1, arg173_1, arg174_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg171_1
        del arg172_1
        del arg173_1
        del arg174_1
        buf96 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_296, x_297, x_in_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(arg175_1, buf96, 3200, 9, grid=grid(3200, 9), stream=stream0)
        del arg175_1
        # Topologically Sorted Source Nodes: [x_296, x_297, x_in_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf97 = extern_kernels.convolution(buf95, buf96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf97, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del buf95
        buf98 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_298, x_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf98, arg176_1, arg177_1, arg178_1, arg179_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg176_1
        del arg177_1
        del arg178_1
        del arg179_1
        # Topologically Sorted Source Nodes: [x_298, x_299, x_in_181], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf99 = extern_kernels.convolution(buf98, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 576, 28, 28), (451584, 1, 16128, 576))
        del arg180_1
        del buf98
        buf101 = empty_strided_cuda((8, 1088, 28, 28), (852992, 1, 30464, 1088), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_182, x_300, x_301], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23.run(buf74, buf82, buf90, buf99, buf91, arg181_1, arg182_1, arg183_1, arg184_1, buf101, 8704, 784, grid=grid(8704, 784), stream=stream0)
        del arg181_1
        del arg182_1
        del arg183_1
        del arg184_1
        # Topologically Sorted Source Nodes: [x_300, x_301, x_in_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del arg185_1
        del buf101
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_302, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf103, arg186_1, arg187_1, arg188_1, arg189_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg186_1
        del arg187_1
        del arg188_1
        del arg189_1
        buf104 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_302, x_303, x_in_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(arg190_1, buf104, 3200, 9, grid=grid(3200, 9), stream=stream0)
        del arg190_1
        # Topologically Sorted Source Nodes: [x_302, x_303, x_in_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf105 = extern_kernels.convolution(buf103, buf104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf105, (8, 400, 28, 28), (313600, 1, 11200, 400))
        del buf103
        del buf104
        buf106 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_304, x_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf106, arg191_1, arg192_1, arg193_1, arg194_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg191_1
        del arg192_1
        del arg193_1
        del arg194_1
        # Topologically Sorted Source Nodes: [x_304, x_305, x_in_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf107 = extern_kernels.convolution(buf106, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 576, 28, 28), (451584, 1, 16128, 576))
        del arg195_1
        buf109 = empty_strided_cuda((8, 1152, 28, 28), (903168, 1, 32256, 1152), torch.float32)
        buf111 = empty_strided_cuda((8, 1152, 28, 28), (903168, 1, 32256, 1152), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_186, x_306, x_307, x_308, x_309], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24.run(buf74, buf82, buf90, buf99, buf107, buf91, arg196_1, arg197_1, arg198_1, arg199_1, arg201_1, arg202_1, arg203_1, arg204_1, buf109, buf111, 9216, 784, grid=grid(9216, 784), stream=stream0)
        del arg196_1
        del arg197_1
        del arg198_1
        del arg199_1
        del arg201_1
        del arg202_1
        del arg203_1
        del arg204_1
        del buf107
        del buf74
        del buf82
        del buf90
        # Topologically Sorted Source Nodes: [x_306, x_307, x_s_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg200_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 1152, 14, 14), (225792, 1, 16128, 1152))
        del arg200_1
        del buf109
        # Topologically Sorted Source Nodes: [x_308, x_309, x_in_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 800, 28, 28), (627200, 1, 22400, 800))
        del arg205_1
        del buf111
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_310, x_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf113, arg206_1, arg207_1, arg208_1, arg209_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg206_1
        del arg207_1
        del arg208_1
        del arg209_1
        buf114 = empty_strided_cuda((800, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_310, x_311, x_in_188], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg210_1, buf114, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg210_1
        # Topologically Sorted Source Nodes: [x_310, x_311, x_in_188], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf115 = extern_kernels.convolution(buf113, buf114, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf115, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf113
        buf116 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_312, x_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf116, arg211_1, arg212_1, arg213_1, arg214_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg211_1
        del arg212_1
        del arg213_1
        del arg214_1
        # Topologically Sorted Source Nodes: [x_312, x_313, x_in_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf117 = extern_kernels.convolution(buf116, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg215_1
        del buf116
        buf119 = empty_strided_cuda((8, 1216, 14, 14), (238336, 1, 17024, 1216), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_190, x_314, x_315], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28.run(buf110, buf117, arg216_1, arg217_1, arg218_1, arg219_1, buf119, 9728, 196, grid=grid(9728, 196), stream=stream0)
        del arg216_1
        del arg217_1
        del arg218_1
        del arg219_1
        # Topologically Sorted Source Nodes: [x_314, x_315, x_in_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg220_1
        del buf119
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [x_316, x_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf121, arg221_1, arg222_1, arg223_1, arg224_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg221_1
        del arg222_1
        del arg223_1
        del arg224_1
        buf122 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [x_316, x_317, x_in_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg225_1, buf122, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg225_1
        # Topologically Sorted Source Nodes: [x_316, x_317, x_in_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf123 = extern_kernels.convolution(buf121, buf122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf123, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf121
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_318, x_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf124, arg226_1, arg227_1, arg228_1, arg229_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg226_1
        del arg227_1
        del arg228_1
        del arg229_1
        # Topologically Sorted Source Nodes: [x_318, x_319, x_in_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf125 = extern_kernels.convolution(buf124, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg230_1
        del buf124
        buf127 = reinterpret_tensor(buf65, (8, 1280, 14, 14), (250880, 1, 17920, 1280), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_in_194, x_320, x_321], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29.run(buf110, buf117, buf125, arg231_1, arg232_1, arg233_1, arg234_1, buf127, 10240, 196, grid=grid(10240, 196), stream=stream0)
        del arg231_1
        del arg232_1
        del arg233_1
        del arg234_1
        # Topologically Sorted Source Nodes: [x_320, x_321, x_in_195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg235_1
        buf129 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_322, x_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf129, arg236_1, arg237_1, arg238_1, arg239_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg236_1
        del arg237_1
        del arg238_1
        del arg239_1
        buf130 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_322, x_323, x_in_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg240_1, buf130, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg240_1
        # Topologically Sorted Source Nodes: [x_322, x_323, x_in_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf131 = extern_kernels.convolution(buf129, buf130, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf131, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf129
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_324, x_325], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf132, arg241_1, arg242_1, arg243_1, arg244_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg241_1
        del arg242_1
        del arg243_1
        del arg244_1
        # Topologically Sorted Source Nodes: [x_324, x_325, x_in_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf133 = extern_kernels.convolution(buf132, arg245_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg245_1
        del buf132
        buf134 = empty_strided_cuda((8, 320, 14, 14), (62720, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dense_49], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf110, buf117, buf125, buf133, buf134, 501760, grid=grid(501760), stream=stream0)
        buf136 = empty_strided_cuda((8, 1344, 14, 14), (263424, 1, 18816, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_198, x_326, x_327], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31.run(buf110, buf117, buf125, buf133, buf134, arg246_1, arg247_1, arg248_1, arg249_1, buf136, 10752, 196, grid=grid(10752, 196), stream=stream0)
        del arg246_1
        del arg247_1
        del arg248_1
        del arg249_1
        # Topologically Sorted Source Nodes: [x_326, x_327, x_in_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf137 = extern_kernels.convolution(buf136, arg250_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg250_1
        del buf136
        buf138 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [x_328, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf138, arg251_1, arg252_1, arg253_1, arg254_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg251_1
        del arg252_1
        del arg253_1
        del arg254_1
        buf139 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_328, x_329, x_in_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg255_1, buf139, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg255_1
        # Topologically Sorted Source Nodes: [x_328, x_329, x_in_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf140 = extern_kernels.convolution(buf138, buf139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf140, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf138
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_330, x_331], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf141, arg256_1, arg257_1, arg258_1, arg259_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg256_1
        del arg257_1
        del arg258_1
        del arg259_1
        # Topologically Sorted Source Nodes: [x_330, x_331, x_in_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf142 = extern_kernels.convolution(buf141, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg260_1
        del buf141
        buf143 = empty_strided_cuda((8, 1024, 14, 14), (200704, 1, 14336, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [resid_47, resid_48, resid_49, resid_50], Original ATen: [aten.add]
        triton_poi_fused_add_32.run(buf110, buf117, buf125, buf133, buf142, buf143, 1605632, grid=grid(1605632), stream=stream0)
        del buf110
        del buf117
        del buf125
        del buf133
        buf145 = empty_strided_cuda((8, 1408, 14, 14), (275968, 1, 19712, 1408), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_202, x_332, x_333], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33.run(buf143, buf134, buf142, arg261_1, arg262_1, arg263_1, arg264_1, buf145, 11264, 196, grid=grid(11264, 196), stream=stream0)
        del arg261_1
        del arg262_1
        del arg263_1
        del arg264_1
        # Topologically Sorted Source Nodes: [x_332, x_333, x_in_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf146 = extern_kernels.convolution(buf145, arg265_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg265_1
        del buf145
        buf147 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_334, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf147, arg266_1, arg267_1, arg268_1, arg269_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg266_1
        del arg267_1
        del arg268_1
        del arg269_1
        buf148 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_334, x_335, x_in_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg270_1, buf148, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg270_1
        # Topologically Sorted Source Nodes: [x_334, x_335, x_in_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf149 = extern_kernels.convolution(buf147, buf148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf149, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf147
        buf150 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [x_336, x_337], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf150, arg271_1, arg272_1, arg273_1, arg274_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg271_1
        del arg272_1
        del arg273_1
        del arg274_1
        # Topologically Sorted Source Nodes: [x_336, x_337, x_in_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf151 = extern_kernels.convolution(buf150, arg275_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg275_1
        del buf150
        buf153 = empty_strided_cuda((8, 1472, 14, 14), (288512, 1, 20608, 1472), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_206, x_338, x_339], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34.run(buf143, buf151, buf134, buf142, arg276_1, arg277_1, arg278_1, arg279_1, buf153, 11776, 196, grid=grid(11776, 196), stream=stream0)
        del arg276_1
        del arg277_1
        del arg278_1
        del arg279_1
        # Topologically Sorted Source Nodes: [x_338, x_339, x_in_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf154 = extern_kernels.convolution(buf153, arg280_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg280_1
        del buf153
        buf155 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [x_340, x_341], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf155, arg281_1, arg282_1, arg283_1, arg284_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg281_1
        del arg282_1
        del arg283_1
        del arg284_1
        buf156 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_340, x_341, x_in_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg285_1, buf156, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg285_1
        # Topologically Sorted Source Nodes: [x_340, x_341, x_in_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf157 = extern_kernels.convolution(buf155, buf156, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf157, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf155
        buf158 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_342, x_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf158, arg286_1, arg287_1, arg288_1, arg289_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg286_1
        del arg287_1
        del arg288_1
        del arg289_1
        # Topologically Sorted Source Nodes: [x_342, x_343, x_in_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf159 = extern_kernels.convolution(buf158, arg290_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg290_1
        del buf158
        buf160 = empty_strided_cuda((8, 512, 14, 14), (100352, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dense_52], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf134, buf142, buf151, buf159, buf160, 802816, grid=grid(802816), stream=stream0)
        del buf134
        del buf142
        buf162 = empty_strided_cuda((8, 1536, 14, 14), (301056, 1, 21504, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_210, x_344, x_345], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36.run(buf143, buf151, buf159, buf160, arg291_1, arg292_1, arg293_1, arg294_1, buf162, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg291_1
        del arg292_1
        del arg293_1
        del arg294_1
        # Topologically Sorted Source Nodes: [x_345, x_in_211], Original ATen: [aten.relu, aten.convolution]
        buf163 = extern_kernels.convolution(buf162, arg295_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg295_1
        del buf162
        buf164 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_346, x_347], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf164, arg296_1, arg297_1, arg298_1, arg299_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg296_1
        del arg297_1
        del arg298_1
        del arg299_1
        buf165 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [x_346, x_347, x_in_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg300_1, buf165, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg300_1
        # Topologically Sorted Source Nodes: [x_346, x_347, x_in_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf166 = extern_kernels.convolution(buf164, buf165, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf166, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf164
        buf167 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [x_348, x_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf167, arg301_1, arg302_1, arg303_1, arg304_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg301_1
        del arg302_1
        del arg303_1
        del arg304_1
        # Topologically Sorted Source Nodes: [x_348, x_349, x_in_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf168 = extern_kernels.convolution(buf167, arg305_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg305_1
        del buf167
        buf170 = reinterpret_tensor(buf106, (8, 1600, 14, 14), (313600, 1, 22400, 1600), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_in_214, x_350, x_351], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37.run(buf143, buf151, buf159, buf168, buf160, arg306_1, arg307_1, arg308_1, arg309_1, buf170, 12800, 196, grid=grid(12800, 196), stream=stream0)
        del arg306_1
        del arg307_1
        del arg308_1
        del arg309_1
        # Topologically Sorted Source Nodes: [x_350, x_351, x_in_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf171 = extern_kernels.convolution(buf170, arg310_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg310_1
        del buf170
        buf172 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_352, x_353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf172, arg311_1, arg312_1, arg313_1, arg314_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg311_1
        del arg312_1
        del arg313_1
        del arg314_1
        buf173 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_352, x_353, x_in_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg315_1, buf173, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg315_1
        # Topologically Sorted Source Nodes: [x_352, x_353, x_in_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf174 = extern_kernels.convolution(buf172, buf173, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf174, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf172
        buf175 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [x_354, x_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf175, arg316_1, arg317_1, arg318_1, arg319_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg316_1
        del arg317_1
        del arg318_1
        del arg319_1
        # Topologically Sorted Source Nodes: [x_354, x_355, x_in_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg320_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg320_1
        del buf175
        buf177 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [resid_51, resid_52, resid_53, resid_54], Original ATen: [aten.add]
        triton_poi_fused_add_38.run(buf177, buf151, buf159, buf168, buf176, 1605632, grid=grid(1605632), stream=stream0)
        del buf151
        del buf159
        buf179 = empty_strided_cuda((8, 1664, 14, 14), (326144, 1, 23296, 1664), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_218, x_356, x_357], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39.run(buf177, buf160, buf168, buf176, arg321_1, arg322_1, arg323_1, arg324_1, buf179, 13312, 196, grid=grid(13312, 196), stream=stream0)
        del arg321_1
        del arg322_1
        del arg323_1
        del arg324_1
        # Topologically Sorted Source Nodes: [x_356, x_357, x_in_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf180 = extern_kernels.convolution(buf179, arg325_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg325_1
        del buf179
        buf181 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [x_358, x_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf181, arg326_1, arg327_1, arg328_1, arg329_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg326_1
        del arg327_1
        del arg328_1
        del arg329_1
        buf182 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [x_358, x_359, x_in_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg330_1, buf182, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg330_1
        # Topologically Sorted Source Nodes: [x_358, x_359, x_in_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf183 = extern_kernels.convolution(buf181, buf182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf183, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf181
        buf184 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [x_360, x_361], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf184, arg331_1, arg332_1, arg333_1, arg334_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg331_1
        del arg332_1
        del arg333_1
        del arg334_1
        # Topologically Sorted Source Nodes: [x_360, x_361, x_in_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf185 = extern_kernels.convolution(buf184, arg335_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg335_1
        del buf184
        buf186 = empty_strided_cuda((8, 704, 14, 14), (137984, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dense_55], Original ATen: [aten.cat]
        triton_poi_fused_cat_40.run(buf160, buf168, buf176, buf185, buf186, 1103872, grid=grid(1103872), stream=stream0)
        del buf160
        del buf168
        del buf176
        buf187 = empty_strided_cuda((8, 1728, 14, 14), (338688, 1, 24192, 1728), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_222, x_362, x_363], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_41.run(buf177, buf185, buf186, arg336_1, arg337_1, arg338_1, arg339_1, buf187, 2709504, grid=grid(2709504), stream=stream0)
        del arg336_1
        del arg337_1
        del arg338_1
        del arg339_1
        # Topologically Sorted Source Nodes: [x_in_222, x_362, x_363, x_in_223], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf188 = extern_kernels.convolution(buf187, arg340_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg340_1
        del buf187
        buf189 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_364, x_365], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf189, arg341_1, arg342_1, arg343_1, arg344_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg341_1
        del arg342_1
        del arg343_1
        del arg344_1
        buf190 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [x_364, x_365, x_in_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg345_1, buf190, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg345_1
        # Topologically Sorted Source Nodes: [x_364, x_365, x_in_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf191 = extern_kernels.convolution(buf189, buf190, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf191, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf189
        buf192 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [x_366, x_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf192, arg346_1, arg347_1, arg348_1, arg349_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg346_1
        del arg347_1
        del arg348_1
        del arg349_1
        # Topologically Sorted Source Nodes: [x_366, x_367, x_in_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf193 = extern_kernels.convolution(buf192, arg350_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg350_1
        del buf192
        buf195 = empty_strided_cuda((8, 1792, 14, 14), (351232, 1, 25088, 1792), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_226, x_368, x_369], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_42.run(buf177, buf185, buf193, buf186, arg351_1, arg352_1, arg353_1, arg354_1, buf195, 14336, 196, grid=grid(14336, 196), stream=stream0)
        del arg351_1
        del arg352_1
        del arg353_1
        del arg354_1
        # Topologically Sorted Source Nodes: [x_368, x_369, x_in_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf196 = extern_kernels.convolution(buf195, arg355_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg355_1
        del buf195
        buf197 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [x_370, x_371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf197, arg356_1, arg357_1, arg358_1, arg359_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg356_1
        del arg357_1
        del arg358_1
        del arg359_1
        buf198 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_370, x_371, x_in_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg360_1, buf198, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg360_1
        # Topologically Sorted Source Nodes: [x_370, x_371, x_in_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf199 = extern_kernels.convolution(buf197, buf198, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf199, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf197
        buf200 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [x_372, x_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf200, arg361_1, arg362_1, arg363_1, arg364_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg361_1
        del arg362_1
        del arg363_1
        del arg364_1
        # Topologically Sorted Source Nodes: [x_372, x_373, x_in_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf201 = extern_kernels.convolution(buf200, arg365_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg365_1
        del buf200
        buf203 = empty_strided_cuda((8, 1856, 14, 14), (363776, 1, 25984, 1856), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_230, x_374, x_375], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43.run(buf177, buf185, buf193, buf201, buf186, arg366_1, arg367_1, arg368_1, arg369_1, buf203, 14848, 196, grid=grid(14848, 196), stream=stream0)
        del arg366_1
        del arg367_1
        del arg368_1
        del arg369_1
        # Topologically Sorted Source Nodes: [x_374, x_375, x_in_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf204 = extern_kernels.convolution(buf203, arg370_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg370_1
        del buf203
        buf205 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_376, x_377], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf205, arg371_1, arg372_1, arg373_1, arg374_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg371_1
        del arg372_1
        del arg373_1
        del arg374_1
        buf206 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [x_376, x_377, x_in_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg375_1, buf206, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg375_1
        # Topologically Sorted Source Nodes: [x_376, x_377, x_in_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf207 = extern_kernels.convolution(buf205, buf206, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf207, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf205
        buf208 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_378, x_379], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf208, arg376_1, arg377_1, arg378_1, arg379_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg376_1
        del arg377_1
        del arg378_1
        del arg379_1
        # Topologically Sorted Source Nodes: [x_378, x_379, x_in_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf209 = extern_kernels.convolution(buf208, arg380_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg380_1
        del buf208
        buf212 = empty_strided_cuda((8, 1920, 14, 14), (376320, 196, 14, 1), torch.float32)
        buf210 = reinterpret_tensor(buf212, (8, 1024, 14, 14), (376320, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [resid_55, resid_56, resid_57, resid_58], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf177, buf185, buf193, buf201, buf209, buf210, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del buf185
        buf211 = reinterpret_tensor(buf212, (8, 896, 14, 14), (376320, 196, 14, 1), 200704)  # alias
        # Topologically Sorted Source Nodes: [dense_58], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf186, buf193, buf201, buf209, buf211, 1404928, grid=grid(1404928), stream=stream0)
        del buf186
        del buf193
        del buf201
        buf213 = empty_strided_cuda((8, 1920, 14, 14), (376320, 1, 26880, 1920), torch.float32)
        # Topologically Sorted Source Nodes: [x_380, x_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf212, arg381_1, arg382_1, arg383_1, arg384_1, buf213, 15360, 196, grid=grid(15360, 196), stream=stream0)
        del arg381_1
        del arg382_1
        del arg383_1
        del arg384_1
        # Topologically Sorted Source Nodes: [x_380, x_381, x_in_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf214 = extern_kernels.convolution(buf213, arg385_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg385_1
        del buf213
        buf215 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [x_382, x_383], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf215, arg386_1, arg387_1, arg388_1, arg389_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg386_1
        del arg387_1
        del arg388_1
        del arg389_1
        buf216 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [x_382, x_383, x_in_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg390_1, buf216, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg390_1
        # Topologically Sorted Source Nodes: [x_382, x_383, x_in_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf217 = extern_kernels.convolution(buf215, buf216, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf217, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf215
        buf218 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_384, x_385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf218, arg391_1, arg392_1, arg393_1, arg394_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg391_1
        del arg392_1
        del arg393_1
        del arg394_1
        # Topologically Sorted Source Nodes: [x_384, x_385, x_in_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf219 = extern_kernels.convolution(buf218, arg395_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg395_1
        del buf218
        buf221 = empty_strided_cuda((8, 1984, 14, 14), (388864, 1, 27776, 1984), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_238, x_386, x_387], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47.run(buf210, buf219, buf211, arg396_1, arg397_1, arg398_1, arg399_1, buf221, 15872, 196, grid=grid(15872, 196), stream=stream0)
        del arg396_1
        del arg397_1
        del arg398_1
        del arg399_1
        # Topologically Sorted Source Nodes: [x_386, x_387, x_in_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf222 = extern_kernels.convolution(buf221, arg400_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg400_1
        del buf221
        buf223 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [x_388, x_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf223, arg401_1, arg402_1, arg403_1, arg404_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg401_1
        del arg402_1
        del arg403_1
        del arg404_1
        buf224 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [x_388, x_389, x_in_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg405_1, buf224, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg405_1
        # Topologically Sorted Source Nodes: [x_388, x_389, x_in_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf225 = extern_kernels.convolution(buf223, buf224, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf225, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf223
        buf226 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [x_390, x_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf226, arg406_1, arg407_1, arg408_1, arg409_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg406_1
        del arg407_1
        del arg408_1
        del arg409_1
        # Topologically Sorted Source Nodes: [x_390, x_391, x_in_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf227 = extern_kernels.convolution(buf226, arg410_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg410_1
        del buf226
        buf229 = reinterpret_tensor(buf91, (8, 2048, 14, 14), (401408, 1, 28672, 2048), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_in_242, x_392, x_393], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48.run(buf210, buf219, buf227, buf211, arg411_1, arg412_1, arg413_1, arg414_1, buf229, 16384, 196, grid=grid(16384, 196), stream=stream0)
        del arg411_1
        del arg412_1
        del arg413_1
        del arg414_1
        # Topologically Sorted Source Nodes: [x_392, x_393, x_in_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf230 = extern_kernels.convolution(buf229, arg415_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg415_1
        del buf229
        buf231 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [x_394, x_395], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf231, arg416_1, arg417_1, arg418_1, arg419_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg416_1
        del arg417_1
        del arg418_1
        del arg419_1
        buf232 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [x_394, x_395, x_in_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg420_1, buf232, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg420_1
        # Topologically Sorted Source Nodes: [x_394, x_395, x_in_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf233 = extern_kernels.convolution(buf231, buf232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf233, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf231
        buf234 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_396, x_397], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf234, arg421_1, arg422_1, arg423_1, arg424_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg421_1
        del arg422_1
        del arg423_1
        del arg424_1
        # Topologically Sorted Source Nodes: [x_396, x_397, x_in_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf235 = extern_kernels.convolution(buf234, arg425_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg425_1
        del buf234
        buf236 = reinterpret_tensor(buf209, (8, 1088, 14, 14), (213248, 196, 14, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [dense_61], Original ATen: [aten.cat]
        triton_poi_fused_cat_49.run(buf211, buf219, buf227, buf235, buf236, 1705984, grid=grid(1705984), stream=stream0)
        del buf211
        buf238 = empty_strided_cuda((8, 2112, 14, 14), (413952, 1, 29568, 2112), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_246, x_398, x_399], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_50.run(buf210, buf219, buf227, buf235, buf236, arg426_1, arg427_1, arg428_1, arg429_1, buf238, 16896, 196, grid=grid(16896, 196), stream=stream0)
        del arg426_1
        del arg427_1
        del arg428_1
        del arg429_1
        # Topologically Sorted Source Nodes: [x_398, x_399, x_in_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf239 = extern_kernels.convolution(buf238, arg430_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg430_1
        del buf238
        buf240 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [x_400, x_401], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf240, arg431_1, arg432_1, arg433_1, arg434_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg431_1
        del arg432_1
        del arg433_1
        del arg434_1
        buf241 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_400, x_401, x_in_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg435_1, buf241, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg435_1
        # Topologically Sorted Source Nodes: [x_400, x_401, x_in_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf242 = extern_kernels.convolution(buf240, buf241, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf242, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf240
        buf243 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [x_402, x_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf243, arg436_1, arg437_1, arg438_1, arg439_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg436_1
        del arg437_1
        del arg438_1
        del arg439_1
        # Topologically Sorted Source Nodes: [x_402, x_403, x_in_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf244 = extern_kernels.convolution(buf243, arg440_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg440_1
        del buf243
        buf245 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [resid_59, resid_60, resid_61, resid_62], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(buf210, buf219, buf227, buf235, buf244, buf245, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del buf210
        del buf212
        del buf219
        del buf227
        del buf235
        buf247 = empty_strided_cuda((8, 2176, 14, 14), (426496, 1, 30464, 2176), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_250, x_404, x_405], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52.run(buf245, buf236, buf244, arg441_1, arg442_1, arg443_1, arg444_1, buf247, 17408, 196, grid=grid(17408, 196), stream=stream0)
        del arg441_1
        del arg442_1
        del arg443_1
        del arg444_1
        # Topologically Sorted Source Nodes: [x_404, x_405, x_in_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf248 = extern_kernels.convolution(buf247, arg445_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg445_1
        del buf247
        buf249 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [x_406, x_407], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf249, arg446_1, arg447_1, arg448_1, arg449_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg446_1
        del arg447_1
        del arg448_1
        del arg449_1
        buf250 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [x_406, x_407, x_in_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg450_1, buf250, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg450_1
        # Topologically Sorted Source Nodes: [x_406, x_407, x_in_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf251 = extern_kernels.convolution(buf249, buf250, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf251, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf249
        buf252 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [x_408, x_409], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf252, arg451_1, arg452_1, arg453_1, arg454_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg451_1
        del arg452_1
        del arg453_1
        del arg454_1
        # Topologically Sorted Source Nodes: [x_408, x_409, x_in_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf253 = extern_kernels.convolution(buf252, arg455_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg455_1
        del buf252
        buf255 = empty_strided_cuda((8, 2240, 14, 14), (439040, 1, 31360, 2240), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_254, x_410, x_411], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53.run(buf245, buf253, buf236, buf244, arg456_1, arg457_1, arg458_1, arg459_1, buf255, 17920, 196, grid=grid(17920, 196), stream=stream0)
        del arg456_1
        del arg457_1
        del arg458_1
        del arg459_1
        # Topologically Sorted Source Nodes: [x_410, x_411, x_in_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf256 = extern_kernels.convolution(buf255, arg460_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg460_1
        del buf255
        buf257 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [x_412, x_413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf257, arg461_1, arg462_1, arg463_1, arg464_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg461_1
        del arg462_1
        del arg463_1
        del arg464_1
        buf258 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [x_412, x_413, x_in_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg465_1, buf258, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg465_1
        # Topologically Sorted Source Nodes: [x_412, x_413, x_in_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf259 = extern_kernels.convolution(buf257, buf258, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf259, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf257
        buf260 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [x_414, x_415], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf260, arg466_1, arg467_1, arg468_1, arg469_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg466_1
        del arg467_1
        del arg468_1
        del arg469_1
        # Topologically Sorted Source Nodes: [x_414, x_415, x_in_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf261 = extern_kernels.convolution(buf260, arg470_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg470_1
        del buf260
        buf262 = reinterpret_tensor(buf127, (8, 1280, 14, 14), (250880, 196, 14, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [dense_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_54.run(buf236, buf244, buf253, buf261, buf262, 2007040, grid=grid(2007040), stream=stream0)
        del buf236
        del buf244
        buf264 = reinterpret_tensor(buf99, (8, 2304, 14, 14), (451584, 1, 32256, 2304), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [x_in_258, x_416, x_417], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55.run(buf245, buf253, buf261, buf262, arg471_1, arg472_1, arg473_1, arg474_1, buf264, 18432, 196, grid=grid(18432, 196), stream=stream0)
        del arg471_1
        del arg472_1
        del arg473_1
        del arg474_1
        # Topologically Sorted Source Nodes: [x_417, x_in_259], Original ATen: [aten.relu, aten.convolution]
        buf265 = extern_kernels.convolution(buf264, arg475_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg475_1
        del buf264
        buf266 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [x_418, x_419], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf266, arg476_1, arg477_1, arg478_1, arg479_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg476_1
        del arg477_1
        del arg478_1
        del arg479_1
        buf267 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [x_418, x_419, x_in_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg480_1, buf267, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg480_1
        # Topologically Sorted Source Nodes: [x_418, x_419, x_in_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf268 = extern_kernels.convolution(buf266, buf267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf268, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf266
        buf269 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [x_420, x_421], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf269, arg481_1, arg482_1, arg483_1, arg484_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg481_1
        del arg482_1
        del arg483_1
        del arg484_1
        # Topologically Sorted Source Nodes: [x_420, x_421, x_in_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf270 = extern_kernels.convolution(buf269, arg485_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg485_1
        del buf269
        buf272 = empty_strided_cuda((8, 2368, 14, 14), (464128, 1, 33152, 2368), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_262, x_422, x_423], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_56.run(buf245, buf253, buf261, buf270, buf262, arg486_1, arg487_1, arg488_1, arg489_1, buf272, 18944, 196, grid=grid(18944, 196), stream=stream0)
        del arg486_1
        del arg487_1
        del arg488_1
        del arg489_1
        # Topologically Sorted Source Nodes: [x_422, x_423, x_in_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf273 = extern_kernels.convolution(buf272, arg490_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del arg490_1
        del buf272
        buf274 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [x_424, x_425], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf274, arg491_1, arg492_1, arg493_1, arg494_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg491_1
        del arg492_1
        del arg493_1
        del arg494_1
        buf275 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [x_424, x_425, x_in_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(arg495_1, buf275, 12800, 9, grid=grid(12800, 9), stream=stream0)
        del arg495_1
        # Topologically Sorted Source Nodes: [x_424, x_425, x_in_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf276 = extern_kernels.convolution(buf274, buf275, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf276, (8, 800, 14, 14), (156800, 1, 11200, 800))
        del buf274
        del buf275
        buf277 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [x_426, x_427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf277, arg496_1, arg497_1, arg498_1, arg499_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg496_1
        del arg497_1
        del arg498_1
        del arg499_1
        # Topologically Sorted Source Nodes: [x_426, x_427, x_in_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf278 = extern_kernels.convolution(buf277, arg500_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
        del arg500_1
        del buf277
        buf280 = empty_strided_cuda((8, 2432, 14, 14), (476672, 1, 34048, 2432), torch.float32)
        buf282 = empty_strided_cuda((8, 2432, 14, 14), (476672, 1, 34048, 2432), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_266, x_428, x_429, x_430, x_431], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_57.run(buf245, buf253, buf261, buf270, buf278, buf262, arg501_1, arg502_1, arg503_1, arg504_1, arg506_1, arg507_1, arg508_1, arg509_1, buf280, buf282, 19456, 196, grid=grid(19456, 196), stream=stream0)
        del arg501_1
        del arg502_1
        del arg503_1
        del arg504_1
        del arg506_1
        del arg507_1
        del arg508_1
        del arg509_1
        del buf245
        del buf253
        del buf261
        del buf262
        del buf270
        del buf278
        # Topologically Sorted Source Nodes: [x_428, x_429, x_s_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf281 = extern_kernels.convolution(buf280, arg505_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (8, 2304, 7, 7), (112896, 1, 16128, 2304))
        del arg505_1
        del buf280
        # Topologically Sorted Source Nodes: [x_430, x_431, x_in_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf283 = extern_kernels.convolution(buf282, arg510_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (8, 1600, 14, 14), (313600, 1, 22400, 1600))
        del arg510_1
        del buf282
        buf284 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [x_432, x_433], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf284, arg511_1, arg512_1, arg513_1, arg514_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg511_1
        del arg512_1
        del arg513_1
        del arg514_1
        buf285 = empty_strided_cuda((1600, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_432, x_433, x_in_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_59.run(arg515_1, buf285, 51200, 9, grid=grid(51200, 9), stream=stream0)
        del arg515_1
        # Topologically Sorted Source Nodes: [x_432, x_433, x_in_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf286 = extern_kernels.convolution(buf284, buf285, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf286, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
        del buf284
        buf287 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [x_434, x_435], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_60.run(buf287, arg516_1, arg517_1, arg518_1, arg519_1, 627200, grid=grid(627200), stream=stream0)
        del arg516_1
        del arg517_1
        del arg518_1
        del arg519_1
        # Topologically Sorted Source Nodes: [x_434, x_435, x_in_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf288 = extern_kernels.convolution(buf287, arg520_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 2176, 7, 7), (106624, 1, 15232, 2176))
        del arg520_1
        del buf287
        buf290 = empty_strided_cuda((8, 2432, 7, 7), (119168, 1, 17024, 2432), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_270, x_436, x_437], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61.run(buf281, buf288, arg521_1, arg522_1, arg523_1, arg524_1, buf290, 19456, 49, grid=grid(19456, 49), stream=stream0)
        del arg521_1
        del arg522_1
        del arg523_1
        del arg524_1
        # Topologically Sorted Source Nodes: [x_436, x_437, x_in_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf291 = extern_kernels.convolution(buf290, arg525_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
        del arg525_1
        del buf290
        buf292 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [x_438, x_439], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_60.run(buf292, arg526_1, arg527_1, arg528_1, arg529_1, 627200, grid=grid(627200), stream=stream0)
        del arg526_1
        del arg527_1
        del arg528_1
        del arg529_1
        buf293 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [x_438, x_439, x_in_272], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_59.run(arg530_1, buf293, 51200, 9, grid=grid(51200, 9), stream=stream0)
        del arg530_1
        # Topologically Sorted Source Nodes: [x_438, x_439, x_in_272], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf294 = extern_kernels.convolution(buf292, buf293, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf294, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
        del buf292
        buf295 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [x_440, x_441], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_60.run(buf295, arg531_1, arg532_1, arg533_1, arg534_1, 627200, grid=grid(627200), stream=stream0)
        del arg531_1
        del arg532_1
        del arg533_1
        del arg534_1
        # Topologically Sorted Source Nodes: [x_440, x_441, x_in_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf296 = extern_kernels.convolution(buf295, arg535_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (8, 2176, 7, 7), (106624, 1, 15232, 2176))
        del arg535_1
        del buf295
        buf298 = empty_strided_cuda((8, 2560, 7, 7), (125440, 1, 17920, 2560), torch.float32)
        # Topologically Sorted Source Nodes: [x_in_274, x_442, x_443], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62.run(buf281, buf288, buf296, arg536_1, arg537_1, arg538_1, arg539_1, buf298, 20480, 49, grid=grid(20480, 49), stream=stream0)
        del arg536_1
        del arg537_1
        del arg538_1
        del arg539_1
        # Topologically Sorted Source Nodes: [x_442, x_443, x_in_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf299 = extern_kernels.convolution(buf298, arg540_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
        del arg540_1
        del buf298
        buf300 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [x_444, x_445], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_60.run(buf300, arg541_1, arg542_1, arg543_1, arg544_1, 627200, grid=grid(627200), stream=stream0)
        del arg541_1
        del arg542_1
        del arg543_1
        del arg544_1
        buf301 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [x_444, x_445, x_in_276], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_59.run(arg545_1, buf301, 51200, 9, grid=grid(51200, 9), stream=stream0)
        del arg545_1
        # Topologically Sorted Source Nodes: [x_444, x_445, x_in_276], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf302 = extern_kernels.convolution(buf300, buf301, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf302, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
        del buf300
        del buf301
        buf303 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [x_446, x_447], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_60.run(buf303, arg546_1, arg547_1, arg548_1, arg549_1, 627200, grid=grid(627200), stream=stream0)
        del arg546_1
        del arg547_1
        del arg548_1
        del arg549_1
        # Topologically Sorted Source Nodes: [x_446, x_447, x_in_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf304 = extern_kernels.convolution(buf303, arg550_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (8, 2176, 7, 7), (106624, 1, 15232, 2176))
        del arg550_1
        del buf303
        buf305 = empty_strided_cuda((8, 640, 7, 7), (31360, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dense_69], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf281, buf288, buf296, buf304, buf305, 250880, grid=grid(250880), stream=stream0)
        buf308 = empty_strided_cuda((8, 2688, 1, 1), (2688, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_448, x_449, x_450, x_451], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_relu_64.run(buf281, buf288, buf296, buf304, buf305, arg551_1, arg552_1, arg553_1, arg554_1, buf308, 21504, 49, grid=grid(21504), stream=stream0)
        del arg551_1
        del arg552_1
        del arg553_1
        del arg554_1
        del buf281
        del buf288
        del buf296
        del buf304
        del buf305
        # Topologically Sorted Source Nodes: [x_449, x_450, x_451, x_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean, aten.convolution]
        buf309 = extern_kernels.convolution(buf308, arg555_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (8, 1000, 1, 1), (1000, 1, 1, 1))
        del arg555_1
        del buf308
        buf310 = reinterpret_tensor(buf309, (8, 1000, 1, 1), (1000, 1, 8000, 8000), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [x_449, x_450, x_451, x_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65.run(buf310, arg556_1, 8000, grid=grid(8000), stream=stream0)
        del arg556_1
    return (reinterpret_tensor(buf310, (8, 1000), (1000, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((296, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((200, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((200, 316, 1, 1), (316, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((200, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((200, 356, 1, 1), (356, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((640, 376, 1, 1), (376, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((400, 376, 1, 1), (376, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((400, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((400, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((400, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((400, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((400, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((400, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((400, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1152, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((800, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((800, 1216, 1, 1), (1216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((800, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((800, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((800, 1408, 1, 1), (1408, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((800, 1472, 1, 1), (1472, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((800, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((800, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((800, 1664, 1, 1), (1664, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((800, 1728, 1, 1), (1728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((800, 1792, 1, 1), (1792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((800, 1856, 1, 1), (1856, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((800, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((800, 1984, 1, 1), (1984, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((800, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((800, 2112, 1, 1), (2112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((800, 2176, 1, 1), (2176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((800, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((800, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((800, 2368, 1, 1), (2368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((2304, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((1600, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((1600, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((1600, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((1000, 2688, 1, 1), (2688, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dpn107', benchmark_compiled_module)
