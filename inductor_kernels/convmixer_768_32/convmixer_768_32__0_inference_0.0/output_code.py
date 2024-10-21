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
# Topologically Sorted Source Nodes: [input_228], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_228 => convolution_65
# Graph fragment:
#   %convolution_65 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [7, 7], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/wv/cwvmhqygwemtvxnqy65otstfnciayil42ephdj2cykxkz4nvkwk6.py
# Topologically Sorted Source Nodes: [input_228], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_228 => convolution_65
# Graph fragment:
#   %convolution_65 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [7, 7], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
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


# kernel path: /tmp/torchinductor_sahanp/pe/cpeuw2l7duzj4jqzho3fkha7juehhxay63epmoafeidgx7awfapk.py
# Topologically Sorted Source Nodes: [input_228, input_229, input_230], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_228 => convolution_65
#   input_229 => relu_65
#   input_230 => add_163, mul_196, mul_197, sub_65
# Graph fragment:
#   %convolution_65 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [7, 7], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_65 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_65,), kwargs = {})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_65, %unsqueeze_521), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_196, %unsqueeze_525), kwargs = {})
#   %add_163 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_197, %unsqueeze_527), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ov/covnuj2e7hlzfjfbv3x2hbcvuq7goaljwqu2iajfwpgfyjjndjdy.py
# Topologically Sorted Source Nodes: [input_231, input_232, input_233, input_234], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_231 => convolution_66
#   input_232 => relu_66
#   input_233 => add_165, mul_199, mul_200, sub_66
#   input_234 => add_166
# Graph fragment:
#   %convolution_66 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_163, %arg7_1, %arg8_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768), kwargs = {})
#   %relu_66 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_66,), kwargs = {})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_66, %unsqueeze_529), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %unsqueeze_533), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %unsqueeze_535), kwargs = {})
#   %add_166 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_165, %add_163), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mo/cmoofqvrjgnpoe3fdiuxppi7nsz4fkuznpu2pknvqktpw5xsayd6.py
# Topologically Sorted Source Nodes: [input_399, input_400, input_401, input_402], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_399 => convolution_114
#   input_400 => relu_114
#   input_401 => add_285, mul_343, mul_344, sub_114
#   input_402 => add_286
# Graph fragment:
#   %convolution_114 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_283, %arg295_1, %arg296_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768), kwargs = {})
#   %relu_114 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_114,), kwargs = {})
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_114, %unsqueeze_913), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_915), kwargs = {})
#   %mul_344 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_343, %unsqueeze_917), kwargs = {})
#   %add_285 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_344, %unsqueeze_919), kwargs = {})
#   %add_286 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_285, %add_283), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tk/ctkjqnz2gr53jgph7r7inv372s4dqrpguarszuz2rhcwmi3grhs4.py
# Topologically Sorted Source Nodes: [input_448, input_449, input_450, input_451, input_452, input_453, input_454, x_4], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
# Source node to ATen node mapping:
#   input_448 => convolution_128
#   input_449 => relu_128
#   input_450 => add_320, mul_385, mul_386, sub_128
#   input_451 => add_321
#   input_452 => convolution_129
#   input_453 => relu_129
#   input_454 => add_323, mul_388, mul_389, sub_129
#   x_4 => mean_1
# Graph fragment:
#   %convolution_128 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_318, %arg379_1, %arg380_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768), kwargs = {})
#   %relu_128 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_128,), kwargs = {})
#   %sub_128 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_128, %unsqueeze_1025), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_128, %unsqueeze_1027), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_385, %unsqueeze_1029), kwargs = {})
#   %add_320 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_386, %unsqueeze_1031), kwargs = {})
#   %add_321 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_320, %add_318), kwargs = {})
#   %convolution_129 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_321, %arg385_1, %arg386_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_129 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_129,), kwargs = {})
#   %sub_129 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_129, %unsqueeze_1033), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %unsqueeze_1035), kwargs = {})
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_388, %unsqueeze_1037), kwargs = {})
#   %add_323 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_389, %unsqueeze_1039), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_323, [-1, -2], True), kwargs = {})
triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_5 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 0, tl.int32)
        tmp4 = triton_helpers.maximum(tmp3, tmp2)
        tmp6 = tmp4 - tmp5
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = libdevice.sqrt(tmp9)
        tmp11 = tl.full([1, 1], 1, tl.int32)
        tmp12 = tmp11 / tmp10
        tmp13 = 1.0
        tmp14 = tmp12 * tmp13
        tmp15 = tmp6 * tmp14
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 + tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hn/chngkv3pchnrsyjpu2rblqeps2mfnl6g5no4dysbh56nng4u4hpo.py
# Topologically Sorted Source Nodes: [input_448, input_449, input_450, input_451, input_452, input_453, input_454, x_4], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
# Source node to ATen node mapping:
#   input_448 => convolution_128
#   input_449 => relu_128
#   input_450 => add_320, mul_385, mul_386, sub_128
#   input_451 => add_321
#   input_452 => convolution_129
#   input_453 => relu_129
#   input_454 => add_323, mul_388, mul_389, sub_129
#   x_4 => mean_1
# Graph fragment:
#   %convolution_128 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_318, %arg379_1, %arg380_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768), kwargs = {})
#   %relu_128 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_128,), kwargs = {})
#   %sub_128 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_128, %unsqueeze_1025), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_128, %unsqueeze_1027), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_385, %unsqueeze_1029), kwargs = {})
#   %add_320 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_386, %unsqueeze_1031), kwargs = {})
#   %add_321 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_320, %add_318), kwargs = {})
#   %convolution_129 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_321, %arg385_1, %arg386_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_129 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_129,), kwargs = {})
#   %sub_129 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_129, %unsqueeze_1033), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %unsqueeze_1035), kwargs = {})
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_388, %unsqueeze_1037), kwargs = {})
#   %add_323 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_389, %unsqueeze_1039), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_323, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_6 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_6(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (6144*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 1024.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1 = args
    args.clear()
    assert_size_stride(arg0_1, (768, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (768, ), (1, ))
    assert_size_stride(arg2_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg152_1, (768, ), (1, ))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg176_1, (768, ), (1, ))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (768, ), (1, ))
    assert_size_stride(arg193_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (768, ), (1, ))
    assert_size_stride(arg199_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (768, ), (1, ))
    assert_size_stride(arg203_1, (768, ), (1, ))
    assert_size_stride(arg204_1, (768, ), (1, ))
    assert_size_stride(arg205_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (768, ), (1, ))
    assert_size_stride(arg209_1, (768, ), (1, ))
    assert_size_stride(arg210_1, (768, ), (1, ))
    assert_size_stride(arg211_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg212_1, (768, ), (1, ))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (768, ), (1, ))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (768, ), (1, ))
    assert_size_stride(arg217_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg218_1, (768, ), (1, ))
    assert_size_stride(arg219_1, (768, ), (1, ))
    assert_size_stride(arg220_1, (768, ), (1, ))
    assert_size_stride(arg221_1, (768, ), (1, ))
    assert_size_stride(arg222_1, (768, ), (1, ))
    assert_size_stride(arg223_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg224_1, (768, ), (1, ))
    assert_size_stride(arg225_1, (768, ), (1, ))
    assert_size_stride(arg226_1, (768, ), (1, ))
    assert_size_stride(arg227_1, (768, ), (1, ))
    assert_size_stride(arg228_1, (768, ), (1, ))
    assert_size_stride(arg229_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg230_1, (768, ), (1, ))
    assert_size_stride(arg231_1, (768, ), (1, ))
    assert_size_stride(arg232_1, (768, ), (1, ))
    assert_size_stride(arg233_1, (768, ), (1, ))
    assert_size_stride(arg234_1, (768, ), (1, ))
    assert_size_stride(arg235_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg236_1, (768, ), (1, ))
    assert_size_stride(arg237_1, (768, ), (1, ))
    assert_size_stride(arg238_1, (768, ), (1, ))
    assert_size_stride(arg239_1, (768, ), (1, ))
    assert_size_stride(arg240_1, (768, ), (1, ))
    assert_size_stride(arg241_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg242_1, (768, ), (1, ))
    assert_size_stride(arg243_1, (768, ), (1, ))
    assert_size_stride(arg244_1, (768, ), (1, ))
    assert_size_stride(arg245_1, (768, ), (1, ))
    assert_size_stride(arg246_1, (768, ), (1, ))
    assert_size_stride(arg247_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg248_1, (768, ), (1, ))
    assert_size_stride(arg249_1, (768, ), (1, ))
    assert_size_stride(arg250_1, (768, ), (1, ))
    assert_size_stride(arg251_1, (768, ), (1, ))
    assert_size_stride(arg252_1, (768, ), (1, ))
    assert_size_stride(arg253_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg254_1, (768, ), (1, ))
    assert_size_stride(arg255_1, (768, ), (1, ))
    assert_size_stride(arg256_1, (768, ), (1, ))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (768, ), (1, ))
    assert_size_stride(arg259_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg260_1, (768, ), (1, ))
    assert_size_stride(arg261_1, (768, ), (1, ))
    assert_size_stride(arg262_1, (768, ), (1, ))
    assert_size_stride(arg263_1, (768, ), (1, ))
    assert_size_stride(arg264_1, (768, ), (1, ))
    assert_size_stride(arg265_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg266_1, (768, ), (1, ))
    assert_size_stride(arg267_1, (768, ), (1, ))
    assert_size_stride(arg268_1, (768, ), (1, ))
    assert_size_stride(arg269_1, (768, ), (1, ))
    assert_size_stride(arg270_1, (768, ), (1, ))
    assert_size_stride(arg271_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg272_1, (768, ), (1, ))
    assert_size_stride(arg273_1, (768, ), (1, ))
    assert_size_stride(arg274_1, (768, ), (1, ))
    assert_size_stride(arg275_1, (768, ), (1, ))
    assert_size_stride(arg276_1, (768, ), (1, ))
    assert_size_stride(arg277_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg278_1, (768, ), (1, ))
    assert_size_stride(arg279_1, (768, ), (1, ))
    assert_size_stride(arg280_1, (768, ), (1, ))
    assert_size_stride(arg281_1, (768, ), (1, ))
    assert_size_stride(arg282_1, (768, ), (1, ))
    assert_size_stride(arg283_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg284_1, (768, ), (1, ))
    assert_size_stride(arg285_1, (768, ), (1, ))
    assert_size_stride(arg286_1, (768, ), (1, ))
    assert_size_stride(arg287_1, (768, ), (1, ))
    assert_size_stride(arg288_1, (768, ), (1, ))
    assert_size_stride(arg289_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg290_1, (768, ), (1, ))
    assert_size_stride(arg291_1, (768, ), (1, ))
    assert_size_stride(arg292_1, (768, ), (1, ))
    assert_size_stride(arg293_1, (768, ), (1, ))
    assert_size_stride(arg294_1, (768, ), (1, ))
    assert_size_stride(arg295_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg296_1, (768, ), (1, ))
    assert_size_stride(arg297_1, (768, ), (1, ))
    assert_size_stride(arg298_1, (768, ), (1, ))
    assert_size_stride(arg299_1, (768, ), (1, ))
    assert_size_stride(arg300_1, (768, ), (1, ))
    assert_size_stride(arg301_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg302_1, (768, ), (1, ))
    assert_size_stride(arg303_1, (768, ), (1, ))
    assert_size_stride(arg304_1, (768, ), (1, ))
    assert_size_stride(arg305_1, (768, ), (1, ))
    assert_size_stride(arg306_1, (768, ), (1, ))
    assert_size_stride(arg307_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg308_1, (768, ), (1, ))
    assert_size_stride(arg309_1, (768, ), (1, ))
    assert_size_stride(arg310_1, (768, ), (1, ))
    assert_size_stride(arg311_1, (768, ), (1, ))
    assert_size_stride(arg312_1, (768, ), (1, ))
    assert_size_stride(arg313_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg314_1, (768, ), (1, ))
    assert_size_stride(arg315_1, (768, ), (1, ))
    assert_size_stride(arg316_1, (768, ), (1, ))
    assert_size_stride(arg317_1, (768, ), (1, ))
    assert_size_stride(arg318_1, (768, ), (1, ))
    assert_size_stride(arg319_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg320_1, (768, ), (1, ))
    assert_size_stride(arg321_1, (768, ), (1, ))
    assert_size_stride(arg322_1, (768, ), (1, ))
    assert_size_stride(arg323_1, (768, ), (1, ))
    assert_size_stride(arg324_1, (768, ), (1, ))
    assert_size_stride(arg325_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg326_1, (768, ), (1, ))
    assert_size_stride(arg327_1, (768, ), (1, ))
    assert_size_stride(arg328_1, (768, ), (1, ))
    assert_size_stride(arg329_1, (768, ), (1, ))
    assert_size_stride(arg330_1, (768, ), (1, ))
    assert_size_stride(arg331_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg332_1, (768, ), (1, ))
    assert_size_stride(arg333_1, (768, ), (1, ))
    assert_size_stride(arg334_1, (768, ), (1, ))
    assert_size_stride(arg335_1, (768, ), (1, ))
    assert_size_stride(arg336_1, (768, ), (1, ))
    assert_size_stride(arg337_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg338_1, (768, ), (1, ))
    assert_size_stride(arg339_1, (768, ), (1, ))
    assert_size_stride(arg340_1, (768, ), (1, ))
    assert_size_stride(arg341_1, (768, ), (1, ))
    assert_size_stride(arg342_1, (768, ), (1, ))
    assert_size_stride(arg343_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg344_1, (768, ), (1, ))
    assert_size_stride(arg345_1, (768, ), (1, ))
    assert_size_stride(arg346_1, (768, ), (1, ))
    assert_size_stride(arg347_1, (768, ), (1, ))
    assert_size_stride(arg348_1, (768, ), (1, ))
    assert_size_stride(arg349_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg350_1, (768, ), (1, ))
    assert_size_stride(arg351_1, (768, ), (1, ))
    assert_size_stride(arg352_1, (768, ), (1, ))
    assert_size_stride(arg353_1, (768, ), (1, ))
    assert_size_stride(arg354_1, (768, ), (1, ))
    assert_size_stride(arg355_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg356_1, (768, ), (1, ))
    assert_size_stride(arg357_1, (768, ), (1, ))
    assert_size_stride(arg358_1, (768, ), (1, ))
    assert_size_stride(arg359_1, (768, ), (1, ))
    assert_size_stride(arg360_1, (768, ), (1, ))
    assert_size_stride(arg361_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg362_1, (768, ), (1, ))
    assert_size_stride(arg363_1, (768, ), (1, ))
    assert_size_stride(arg364_1, (768, ), (1, ))
    assert_size_stride(arg365_1, (768, ), (1, ))
    assert_size_stride(arg366_1, (768, ), (1, ))
    assert_size_stride(arg367_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg368_1, (768, ), (1, ))
    assert_size_stride(arg369_1, (768, ), (1, ))
    assert_size_stride(arg370_1, (768, ), (1, ))
    assert_size_stride(arg371_1, (768, ), (1, ))
    assert_size_stride(arg372_1, (768, ), (1, ))
    assert_size_stride(arg373_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg374_1, (768, ), (1, ))
    assert_size_stride(arg375_1, (768, ), (1, ))
    assert_size_stride(arg376_1, (768, ), (1, ))
    assert_size_stride(arg377_1, (768, ), (1, ))
    assert_size_stride(arg378_1, (768, ), (1, ))
    assert_size_stride(arg379_1, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg380_1, (768, ), (1, ))
    assert_size_stride(arg381_1, (768, ), (1, ))
    assert_size_stride(arg382_1, (768, ), (1, ))
    assert_size_stride(arg383_1, (768, ), (1, ))
    assert_size_stride(arg384_1, (768, ), (1, ))
    assert_size_stride(arg385_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg386_1, (768, ), (1, ))
    assert_size_stride(arg387_1, (768, ), (1, ))
    assert_size_stride(arg388_1, (768, ), (1, ))
    assert_size_stride(arg389_1, (768, ), (1, ))
    assert_size_stride(arg390_1, (768, ), (1, ))
    assert_size_stride(arg391_1, (1000, 768), (768, 1))
    assert_size_stride(arg392_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_228], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg2_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg2_1
        buf1 = empty_strided_cuda((768, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_228], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 2304, 49, grid=grid(2304, 49), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [input_228], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(7, 7), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_228, input_229, input_230], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf3, arg1_1, arg3_1, arg4_1, arg5_1, arg6_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg1_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        # Topologically Sorted Source Nodes: [input_231], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg7_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf4, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg7_1
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [input_231, input_232, input_233, input_234], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf5, buf4, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del arg8_1
        del arg9_1
        del buf4
        # Topologically Sorted Source Nodes: [input_231, input_232, input_233, input_234, input_235], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf6 = extern_kernels.convolution(buf5, arg13_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg13_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_231, input_232, input_233, input_234, input_235, input_236, input_237], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf7, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg14_1
        del arg15_1
        del arg16_1
        del arg17_1
        del arg18_1
        # Topologically Sorted Source Nodes: [input_238], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg19_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf8, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg19_1
        buf9 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [input_238, input_239, input_240, input_241], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf9, buf8, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg20_1
        del arg21_1
        del arg22_1
        del arg23_1
        del arg24_1
        del buf8
        # Topologically Sorted Source Nodes: [input_238, input_239, input_240, input_241, input_242], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf10 = extern_kernels.convolution(buf9, arg25_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg25_1
        del buf9
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_238, input_239, input_240, input_241, input_242, input_243, input_244], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf11, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg26_1
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        # Topologically Sorted Source Nodes: [input_245], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg31_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf12, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg31_1
        buf13 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [input_245, input_246, input_247, input_248], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf13, buf12, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        del arg36_1
        del buf12
        # Topologically Sorted Source Nodes: [input_245, input_246, input_247, input_248, input_249], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf14 = extern_kernels.convolution(buf13, arg37_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg37_1
        del buf13
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_245, input_246, input_247, input_248, input_249, input_250, input_251], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf15, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg38_1
        del arg39_1
        del arg40_1
        del arg41_1
        del arg42_1
        # Topologically Sorted Source Nodes: [input_252], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg43_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf16, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg43_1
        buf17 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [input_252, input_253, input_254, input_255], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf17, buf16, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg44_1
        del arg45_1
        del arg46_1
        del arg47_1
        del arg48_1
        del buf16
        # Topologically Sorted Source Nodes: [input_252, input_253, input_254, input_255, input_256], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf18 = extern_kernels.convolution(buf17, arg49_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg49_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_252, input_253, input_254, input_255, input_256, input_257, input_258], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf19, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg50_1
        del arg51_1
        del arg52_1
        del arg53_1
        del arg54_1
        # Topologically Sorted Source Nodes: [input_259], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg55_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf20, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg55_1
        buf21 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_259, input_260, input_261, input_262], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf21, buf20, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg56_1
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf20
        # Topologically Sorted Source Nodes: [input_259, input_260, input_261, input_262, input_263], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf22 = extern_kernels.convolution(buf21, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg61_1
        del buf21
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_259, input_260, input_261, input_262, input_263, input_264, input_265], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf23, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        del arg66_1
        # Topologically Sorted Source Nodes: [input_266], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg67_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf24, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg67_1
        buf25 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [input_266, input_267, input_268, input_269], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf25, buf24, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg68_1
        del arg69_1
        del arg70_1
        del arg71_1
        del arg72_1
        del buf24
        # Topologically Sorted Source Nodes: [input_266, input_267, input_268, input_269, input_270], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf26 = extern_kernels.convolution(buf25, arg73_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg73_1
        del buf25
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [input_266, input_267, input_268, input_269, input_270, input_271, input_272], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf27, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg74_1
        del arg75_1
        del arg76_1
        del arg77_1
        del arg78_1
        # Topologically Sorted Source Nodes: [input_273], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg79_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf28, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg79_1
        buf29 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [input_273, input_274, input_275, input_276], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf29, buf28, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg80_1
        del arg81_1
        del arg82_1
        del arg83_1
        del arg84_1
        del buf28
        # Topologically Sorted Source Nodes: [input_273, input_274, input_275, input_276, input_277], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf30 = extern_kernels.convolution(buf29, arg85_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg85_1
        del buf29
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [input_273, input_274, input_275, input_276, input_277, input_278, input_279], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf31, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg86_1
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        # Topologically Sorted Source Nodes: [input_280], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg91_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf32, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg91_1
        buf33 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [input_280, input_281, input_282, input_283], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf33, buf32, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del arg96_1
        del buf32
        # Topologically Sorted Source Nodes: [input_280, input_281, input_282, input_283, input_284], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf34 = extern_kernels.convolution(buf33, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg97_1
        del buf33
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [input_280, input_281, input_282, input_283, input_284, input_285, input_286], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf35, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        del arg98_1
        del arg99_1
        # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg103_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf36, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg103_1
        buf37 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [input_287, input_288, input_289, input_290], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf37, buf36, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg104_1
        del arg105_1
        del arg106_1
        del arg107_1
        del arg108_1
        del buf36
        # Topologically Sorted Source Nodes: [input_287, input_288, input_289, input_290, input_291], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf38 = extern_kernels.convolution(buf37, arg109_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg109_1
        del buf37
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [input_287, input_288, input_289, input_290, input_291, input_292, input_293], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf39, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg110_1
        del arg111_1
        del arg112_1
        del arg113_1
        del arg114_1
        # Topologically Sorted Source Nodes: [input_294], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg115_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf40, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg115_1
        buf41 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [input_294, input_295, input_296, input_297], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf41, buf40, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg116_1
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        del buf40
        # Topologically Sorted Source Nodes: [input_294, input_295, input_296, input_297, input_298], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf42 = extern_kernels.convolution(buf41, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg121_1
        del buf41
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [input_294, input_295, input_296, input_297, input_298, input_299, input_300], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf43, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        del arg126_1
        # Topologically Sorted Source Nodes: [input_301], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg127_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf44, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg127_1
        buf45 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [input_301, input_302, input_303, input_304], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf45, buf44, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg128_1
        del arg129_1
        del arg130_1
        del arg131_1
        del arg132_1
        del buf44
        # Topologically Sorted Source Nodes: [input_301, input_302, input_303, input_304, input_305], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf46 = extern_kernels.convolution(buf45, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg133_1
        del buf45
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [input_301, input_302, input_303, input_304, input_305, input_306, input_307], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf47, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg134_1
        del arg135_1
        del arg136_1
        del arg137_1
        del arg138_1
        # Topologically Sorted Source Nodes: [input_308], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg139_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf48, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg139_1
        buf49 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [input_308, input_309, input_310, input_311], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf49, buf48, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg140_1
        del arg141_1
        del arg142_1
        del arg143_1
        del arg144_1
        del buf48
        # Topologically Sorted Source Nodes: [input_308, input_309, input_310, input_311, input_312], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf50 = extern_kernels.convolution(buf49, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg145_1
        del buf49
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [input_308, input_309, input_310, input_311, input_312, input_313, input_314], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf51, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg146_1
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        # Topologically Sorted Source Nodes: [input_315], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg151_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf52, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg151_1
        buf53 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [input_315, input_316, input_317, input_318], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf53, buf52, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        del arg156_1
        del buf52
        # Topologically Sorted Source Nodes: [input_315, input_316, input_317, input_318, input_319], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf54 = extern_kernels.convolution(buf53, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg157_1
        del buf53
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [input_315, input_316, input_317, input_318, input_319, input_320, input_321], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf55, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg158_1
        del arg159_1
        del arg160_1
        del arg161_1
        del arg162_1
        # Topologically Sorted Source Nodes: [input_322], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg163_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf56, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg163_1
        buf57 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [input_322, input_323, input_324, input_325], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf57, buf56, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg164_1
        del arg165_1
        del arg166_1
        del arg167_1
        del arg168_1
        del buf56
        # Topologically Sorted Source Nodes: [input_322, input_323, input_324, input_325, input_326], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf58 = extern_kernels.convolution(buf57, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg169_1
        del buf57
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [input_322, input_323, input_324, input_325, input_326, input_327, input_328], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf59, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg170_1
        del arg171_1
        del arg172_1
        del arg173_1
        del arg174_1
        # Topologically Sorted Source Nodes: [input_329], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg175_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf60, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg175_1
        buf61 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [input_329, input_330, input_331, input_332], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf61, buf60, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg176_1
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        del buf60
        # Topologically Sorted Source Nodes: [input_329, input_330, input_331, input_332, input_333], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf62 = extern_kernels.convolution(buf61, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg181_1
        del buf61
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [input_329, input_330, input_331, input_332, input_333, input_334, input_335], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf63, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        del arg186_1
        # Topologically Sorted Source Nodes: [input_336], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg187_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf64, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg187_1
        buf65 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [input_336, input_337, input_338, input_339], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf65, buf64, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg188_1
        del arg189_1
        del arg190_1
        del arg191_1
        del arg192_1
        del buf64
        # Topologically Sorted Source Nodes: [input_336, input_337, input_338, input_339, input_340], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf66 = extern_kernels.convolution(buf65, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg193_1
        del buf65
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [input_336, input_337, input_338, input_339, input_340, input_341, input_342], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf67, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg194_1
        del arg195_1
        del arg196_1
        del arg197_1
        del arg198_1
        # Topologically Sorted Source Nodes: [input_343], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg199_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf68, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg199_1
        buf69 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [input_343, input_344, input_345, input_346], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf69, buf68, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg200_1
        del arg201_1
        del arg202_1
        del arg203_1
        del arg204_1
        del buf68
        # Topologically Sorted Source Nodes: [input_343, input_344, input_345, input_346, input_347], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf70 = extern_kernels.convolution(buf69, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg205_1
        del buf69
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [input_343, input_344, input_345, input_346, input_347, input_348, input_349], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf71, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg206_1
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        # Topologically Sorted Source Nodes: [input_350], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg211_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf72, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg211_1
        buf73 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [input_350, input_351, input_352, input_353], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf73, buf72, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        del arg216_1
        del buf72
        # Topologically Sorted Source Nodes: [input_350, input_351, input_352, input_353, input_354], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf74 = extern_kernels.convolution(buf73, arg217_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg217_1
        del buf73
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [input_350, input_351, input_352, input_353, input_354, input_355, input_356], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf75, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg218_1
        del arg219_1
        del arg220_1
        del arg221_1
        del arg222_1
        # Topologically Sorted Source Nodes: [input_357], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg223_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf76, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg223_1
        buf77 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [input_357, input_358, input_359, input_360], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf77, buf76, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg224_1
        del arg225_1
        del arg226_1
        del arg227_1
        del arg228_1
        del buf76
        # Topologically Sorted Source Nodes: [input_357, input_358, input_359, input_360, input_361], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf78 = extern_kernels.convolution(buf77, arg229_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg229_1
        del buf77
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [input_357, input_358, input_359, input_360, input_361, input_362, input_363], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf79, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg230_1
        del arg231_1
        del arg232_1
        del arg233_1
        del arg234_1
        # Topologically Sorted Source Nodes: [input_364], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg235_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf80, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg235_1
        buf81 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [input_364, input_365, input_366, input_367], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf81, buf80, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg236_1
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        del buf80
        # Topologically Sorted Source Nodes: [input_364, input_365, input_366, input_367, input_368], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf82 = extern_kernels.convolution(buf81, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg241_1
        del buf81
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [input_364, input_365, input_366, input_367, input_368, input_369, input_370], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf83, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        del arg246_1
        # Topologically Sorted Source Nodes: [input_371], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, arg247_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf84, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg247_1
        buf85 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [input_371, input_372, input_373, input_374], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf85, buf84, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg248_1
        del arg249_1
        del arg250_1
        del arg251_1
        del arg252_1
        del buf84
        # Topologically Sorted Source Nodes: [input_371, input_372, input_373, input_374, input_375], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf86 = extern_kernels.convolution(buf85, arg253_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg253_1
        del buf85
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [input_371, input_372, input_373, input_374, input_375, input_376, input_377], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf87, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg254_1
        del arg255_1
        del arg256_1
        del arg257_1
        del arg258_1
        # Topologically Sorted Source Nodes: [input_378], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg259_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf88, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg259_1
        buf89 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [input_378, input_379, input_380, input_381], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf89, buf88, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg260_1
        del arg261_1
        del arg262_1
        del arg263_1
        del arg264_1
        del buf88
        # Topologically Sorted Source Nodes: [input_378, input_379, input_380, input_381, input_382], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf90 = extern_kernels.convolution(buf89, arg265_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg265_1
        del buf89
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [input_378, input_379, input_380, input_381, input_382, input_383, input_384], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf91, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg266_1
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        # Topologically Sorted Source Nodes: [input_385], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg271_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf92, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg271_1
        buf93 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [input_385, input_386, input_387, input_388], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf93, buf92, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        del arg276_1
        del buf92
        # Topologically Sorted Source Nodes: [input_385, input_386, input_387, input_388, input_389], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf94 = extern_kernels.convolution(buf93, arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg277_1
        del buf93
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [input_385, input_386, input_387, input_388, input_389, input_390, input_391], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf95, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg278_1
        del arg279_1
        del arg280_1
        del arg281_1
        del arg282_1
        # Topologically Sorted Source Nodes: [input_392], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg283_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf96, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg283_1
        buf97 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [input_392, input_393, input_394, input_395], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf97, buf96, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg284_1
        del arg285_1
        del arg286_1
        del arg287_1
        del arg288_1
        del buf96
        # Topologically Sorted Source Nodes: [input_392, input_393, input_394, input_395, input_396], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf98 = extern_kernels.convolution(buf97, arg289_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg289_1
        del buf97
        buf99 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [input_392, input_393, input_394, input_395, input_396, input_397, input_398], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf99, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg290_1
        del arg291_1
        del arg292_1
        del arg293_1
        del arg294_1
        # Topologically Sorted Source Nodes: [input_399], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg295_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf100, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg295_1
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [input_399, input_400, input_401, input_402], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4.run(buf101, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, buf99, 6291456, grid=grid(6291456), stream=stream0)
        del arg296_1
        del arg297_1
        del arg298_1
        del arg299_1
        del arg300_1
        del buf99
        # Topologically Sorted Source Nodes: [input_399, input_400, input_401, input_402, input_403], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf102 = extern_kernels.convolution(buf101, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg301_1
        del buf101
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [input_399, input_400, input_401, input_402, input_403, input_404, input_405], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf103, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        del arg306_1
        # Topologically Sorted Source Nodes: [input_406], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg307_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf104, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg307_1
        buf105 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [input_406, input_407, input_408, input_409], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf105, buf104, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg308_1
        del arg309_1
        del arg310_1
        del arg311_1
        del arg312_1
        del buf104
        # Topologically Sorted Source Nodes: [input_406, input_407, input_408, input_409, input_410], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf106 = extern_kernels.convolution(buf105, arg313_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg313_1
        del buf105
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [input_406, input_407, input_408, input_409, input_410, input_411, input_412], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf107, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg314_1
        del arg315_1
        del arg316_1
        del arg317_1
        del arg318_1
        # Topologically Sorted Source Nodes: [input_413], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg319_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf108, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg319_1
        buf109 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [input_413, input_414, input_415, input_416], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf109, buf108, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg320_1
        del arg321_1
        del arg322_1
        del arg323_1
        del arg324_1
        del buf108
        # Topologically Sorted Source Nodes: [input_413, input_414, input_415, input_416, input_417], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf110 = extern_kernels.convolution(buf109, arg325_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg325_1
        del buf109
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [input_413, input_414, input_415, input_416, input_417, input_418, input_419], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf111, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg326_1
        del arg327_1
        del arg328_1
        del arg329_1
        del arg330_1
        # Topologically Sorted Source Nodes: [input_420], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg331_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf112, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg331_1
        buf113 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [input_420, input_421, input_422, input_423], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf113, buf112, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg332_1
        del arg333_1
        del arg334_1
        del arg335_1
        del arg336_1
        del buf112
        # Topologically Sorted Source Nodes: [input_420, input_421, input_422, input_423, input_424], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf114 = extern_kernels.convolution(buf113, arg337_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg337_1
        del buf113
        buf115 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [input_420, input_421, input_422, input_423, input_424, input_425, input_426], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf115, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg338_1
        del arg339_1
        del arg340_1
        del arg341_1
        del arg342_1
        # Topologically Sorted Source Nodes: [input_427], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, arg343_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf116, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg343_1
        buf117 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [input_427, input_428, input_429, input_430], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf117, buf116, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg344_1
        del arg345_1
        del arg346_1
        del arg347_1
        del arg348_1
        del buf116
        # Topologically Sorted Source Nodes: [input_427, input_428, input_429, input_430, input_431], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf118 = extern_kernels.convolution(buf117, arg349_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg349_1
        del buf117
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [input_427, input_428, input_429, input_430, input_431, input_432, input_433], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf119, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg350_1
        del arg351_1
        del arg352_1
        del arg353_1
        del arg354_1
        # Topologically Sorted Source Nodes: [input_434], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg355_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf120, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg355_1
        buf121 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [input_434, input_435, input_436, input_437], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf121, buf120, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg356_1
        del arg357_1
        del arg358_1
        del arg359_1
        del arg360_1
        del buf120
        # Topologically Sorted Source Nodes: [input_434, input_435, input_436, input_437, input_438], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf122 = extern_kernels.convolution(buf121, arg361_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg361_1
        del buf121
        buf123 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [input_434, input_435, input_436, input_437, input_438, input_439, input_440], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf123, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg362_1
        del arg363_1
        del arg364_1
        del arg365_1
        del arg366_1
        # Topologically Sorted Source Nodes: [input_441], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg367_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf124, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg367_1
        buf125 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [input_441, input_442, input_443, input_444], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf125, buf124, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg368_1
        del arg369_1
        del arg370_1
        del arg371_1
        del arg372_1
        del buf124
        # Topologically Sorted Source Nodes: [input_441, input_442, input_443, input_444, input_445], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf126 = extern_kernels.convolution(buf125, arg373_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg373_1
        del buf125
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [input_441, input_442, input_443, input_444, input_445, input_446, input_447], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf127, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg374_1
        del arg375_1
        del arg376_1
        del arg377_1
        del arg378_1
        # Topologically Sorted Source Nodes: [input_448], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg379_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf128, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg379_1
        buf129 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [input_448, input_449, input_450, input_451], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf129, buf128, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg380_1
        del arg381_1
        del arg382_1
        del arg383_1
        del arg384_1
        del buf128
        # Topologically Sorted Source Nodes: [input_448, input_449, input_450, input_451, input_452], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add]
        buf130 = extern_kernels.convolution(buf129, arg385_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 768, 32, 32), (786432, 1, 24576, 768))
        del arg385_1
        del buf129
        buf131 = empty_strided_cuda((8, 768, 1, 1, 8), (6144, 1, 49152, 49152, 768), torch.float32)
        # Topologically Sorted Source Nodes: [input_448, input_449, input_450, input_451, input_452, input_453, input_454, x_4], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
        triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_5.run(buf130, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, buf131, 49152, 128, grid=grid(49152), stream=stream0)
        del arg386_1
        del arg387_1
        del arg388_1
        del arg389_1
        del arg390_1
        del buf130
        buf133 = empty_strided_cuda((8, 768, 1, 1), (768, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [input_448, input_449, input_450, input_451, input_452, input_453, input_454, x_4], Original ATen: [aten.convolution, aten.relu, aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_6.run(buf131, buf133, 6144, 8, grid=grid(6144), stream=stream0)
        del buf131
        buf134 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg392_1, reinterpret_tensor(buf133, (8, 768), (768, 1), 0), reinterpret_tensor(arg391_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf134)
        del arg391_1
        del arg392_1
        del buf133
    return (buf134, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((768, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convmixer_768_32', benchmark_compiled_module)
