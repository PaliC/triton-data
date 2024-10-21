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
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_4 => convolution_57
# Graph fragment:
#   %convolution_57 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/qd/cqdbaky7q4ab4v5b2vexynleduwq3ndyow5stnazorbwox66cx4i.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_4 => convolution_57
# Graph fragment:
#   %convolution_57 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[128, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
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


# kernel path: /tmp/torchinductor_sahanp/ek/cekujtroe4lmfynngo4l6377j52uwrdlvgggr4iljkvhusbdr7ty.py
# Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_5 => add_105, mul_159, mul_160, sub_36
#   input_6 => relu_1
# Graph fragment:
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_57, %unsqueeze_225), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_227), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_159, %unsqueeze_229), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_160, %unsqueeze_231), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_105,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_sahanp/l4/cl4uutvjvcxfy3a6lhag7gv2fs2645kqk2hzzivnt3zrbpm5fzrd.py
# Topologically Sorted Source Nodes: [input_5, input_6, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_5 => add_105, mul_159, mul_160, sub_36
#   input_6 => relu_1
#   x_166 => convolution_58
# Graph fragment:
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_57, %unsqueeze_225), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_227), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_159, %unsqueeze_229), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_160, %unsqueeze_231), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_105,), kwargs = {})
#   %convolution_58 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg6_1, %arg7_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (512*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ds/cds772kcvooopfdy3yft36jirjrmvv3gbe3bdw3e4yuvbg5e4lsz.py
# Topologically Sorted Source Nodes: [input_5, input_6, x_166, x_167, add_26, batch_norm_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_26 => add_108
#   batch_norm_30 => add_110, mul_165, mul_166, sub_38
#   input_5 => add_105, mul_159, mul_160, sub_36
#   input_6 => relu_1
#   x_166 => convolution_58
#   x_167 => add_107, mul_162, mul_163, sub_37
# Graph fragment:
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_57, %unsqueeze_225), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_227), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_159, %unsqueeze_229), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_160, %unsqueeze_231), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_105,), kwargs = {})
#   %convolution_58 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg6_1, %arg7_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_58, %unsqueeze_233), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_235), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_162, %unsqueeze_237), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_163, %unsqueeze_239), kwargs = {})
#   %add_108 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_107, %arg12_1), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_108, %unsqueeze_241), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_243), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_165, %unsqueeze_245), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_166, %unsqueeze_247), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2 + (784*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr10 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 - tmp20
    tmp23 = tmp22 + tmp6
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tmp9 / tmp24
    tmp26 = tmp25 * tmp11
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp19, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (192*x2) + (150528*y1)), tmp31, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5c/c5c7wsrzu2y4d7n2cd5tbhxxffscc7sdxshr4vkbri5annvgfnm6.py
# Topologically Sorted Source Nodes: [x_170], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_170 => add_111, erf_22, mul_167, mul_168, mul_169
# Graph fragment:
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_59, 0.5), kwargs = {})
#   %mul_168 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_59, 0.7071067811865476), kwargs = {})
#   %erf_22 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_168,), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_22, 1), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_167, %add_111), kwargs = {})
triton_poi_fused_gelu_5 = async_compile.triton('triton_poi_fused_gelu_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_5(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kf/ckf4tu4m4o6j7boazz3yuyxsj4zn4ezayq7uvqirs5kncdyvh23z.py
# Topologically Sorted Source Nodes: [x_170, x_172], Original ATen: [aten.gelu, aten.convolution]
# Source node to ATen node mapping:
#   x_170 => add_111, erf_22, mul_167, mul_168, mul_169
#   x_172 => convolution_60
# Graph fragment:
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_59, 0.5), kwargs = {})
#   %mul_168 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_59, 0.7071067811865476), kwargs = {})
#   %erf_22 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_168,), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_22, 1), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_167, %add_111), kwargs = {})
#   %convolution_60 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_169, %arg18_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8), kwargs = {})
triton_poi_fused_convolution_gelu_6 = async_compile.triton('triton_poi_fused_convolution_gelu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 48
    y1 = (yindex // 48)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (48*x2) + (432*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/az/cazibebasamf72r4z5dkyasuufntpgkxe7tihzwewl2rm6wbydsd.py
# Topologically Sorted Source Nodes: [x_176, batch_norm_31], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_31 => add_115, mul_174, mul_175, sub_39
#   x_176 => add_113
# Graph fragment:
#   %add_113 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_108, %convolution_61), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_113, %unsqueeze_249), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_251), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_174, %unsqueeze_253), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_175, %unsqueeze_255), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6k/c6kdeasxal4natvoe6yepq5li2xxpw6geihgq3lfx7qlntxp4og7.py
# Topologically Sorted Source Nodes: [x_176, x_184, batch_norm_32], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_32 => add_120, mul_183, mul_184, sub_40
#   x_176 => add_113
#   x_184 => add_118
# Graph fragment:
#   %add_113 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_108, %convolution_61), kwargs = {})
#   %add_118 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_113, %convolution_64), kwargs = {})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_118, %unsqueeze_257), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_259), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_183, %unsqueeze_261), kwargs = {})
#   %add_120 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_184, %unsqueeze_263), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xx/cxxnxfno7q4vtyjamgbse4rcoxan2tj5jvw475h3a6ojukkqalwb.py
# Topologically Sorted Source Nodes: [x_176, x_184, x_192, batch_norm_33], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_33 => add_125, mul_192, mul_193, sub_41
#   x_176 => add_113
#   x_184 => add_118
#   x_192 => add_123
# Graph fragment:
#   %add_113 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_108, %convolution_61), kwargs = {})
#   %add_118 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_113, %convolution_64), kwargs = {})
#   %add_123 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_118, %convolution_67), kwargs = {})
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_123, %unsqueeze_265), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_267), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_192, %unsqueeze_269), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_193, %unsqueeze_271), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1, 1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp21, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6t/c6t24xum7si4vvj3i546icthjpqfgh43ehjmdkgga54imunyf2dv.py
# Topologically Sorted Source Nodes: [x_176, x_184, x_192, x_200, batch_norm_34], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_34 => add_130, mul_201, mul_202, sub_42
#   x_176 => add_113
#   x_184 => add_118
#   x_192 => add_123
#   x_200 => add_128
# Graph fragment:
#   %add_113 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_108, %convolution_61), kwargs = {})
#   %add_118 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_113, %convolution_64), kwargs = {})
#   %add_123 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_118, %convolution_67), kwargs = {})
#   %add_128 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_123, %convolution_70), kwargs = {})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_128, %unsqueeze_273), kwargs = {})
#   %mul_201 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_275), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_201, %unsqueeze_277), kwargs = {})
#   %add_130 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_202, %unsqueeze_279), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1, 1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (192*y3)), tmp8, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp23, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/or/corbeqes4wnezxyjdrhnh4i53y7p2vtalodnbhize4hlpdrgo2vs.py
# Topologically Sorted Source Nodes: [x_208, batch_norm_35], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_35 => add_135, mul_210, mul_211, sub_43
#   x_208 => add_133
# Graph fragment:
#   %add_133 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_128, %convolution_73), kwargs = {})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_133, %unsqueeze_281), kwargs = {})
#   %mul_210 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_283), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_210, %unsqueeze_285), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_211, %unsqueeze_287), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4h/c4hs7l7b6mo2x474tdzj4qzclsgu3fuxklo7b75lsvomonim5gav.py
# Topologically Sorted Source Nodes: [x_208, x_216, batch_norm_36], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_36 => add_140, mul_219, mul_220, sub_44
#   x_208 => add_133
#   x_216 => add_138
# Graph fragment:
#   %add_133 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_128, %convolution_73), kwargs = {})
#   %add_138 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_133, %convolution_76), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_138, %unsqueeze_289), kwargs = {})
#   %mul_219 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %unsqueeze_291), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_219, %unsqueeze_293), kwargs = {})
#   %add_140 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_220, %unsqueeze_295), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xm/cxme5z7zkd72gcz76nwtylkww3bjvef5jra5zd6zwrg3z3pqlme4.py
# Topologically Sorted Source Nodes: [x_208, x_216, x_224], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_208 => add_133
#   x_216 => add_138
#   x_224 => add_143
# Graph fragment:
#   %add_133 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_128, %convolution_73), kwargs = {})
#   %add_138 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_133, %convolution_76), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_138, %convolution_79), kwargs = {})
triton_poi_fused_add_13 = async_compile.triton('triton_poi_fused_add_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bt/cbtz75catnr3yr65cr3gahfhoegqpqo26vu45oxh6cu5lpmg3ffj.py
# Topologically Sorted Source Nodes: [x_208, x_216, x_224, x_225], Original ATen: [aten.add, aten.convolution]
# Source node to ATen node mapping:
#   x_208 => add_133
#   x_216 => add_138
#   x_224 => add_143
#   x_225 => convolution_80
# Graph fragment:
#   %add_133 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_128, %convolution_73), kwargs = {})
#   %add_138 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_133, %convolution_76), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_138, %convolution_79), kwargs = {})
#   %convolution_80 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_143, %arg62_1, %arg63_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_14 = async_compile.triton('triton_poi_fused_add_convolution_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (768*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/l2/cl2k4zo7ravm6pc44wg6z5ommhdivzymhswmru7voddmthahklbf.py
# Topologically Sorted Source Nodes: [x_208, x_216, x_224, x_225, x_226, add_34, batch_norm_38], Original ATen: [aten.add, aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   add_34 => add_146
#   batch_norm_38 => add_148, mul_231, mul_232, sub_46
#   x_208 => add_133
#   x_216 => add_138
#   x_224 => add_143
#   x_225 => convolution_80
#   x_226 => add_145, mul_228, mul_229, sub_45
# Graph fragment:
#   %add_133 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_128, %convolution_73), kwargs = {})
#   %add_138 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_133, %convolution_76), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_138, %convolution_79), kwargs = {})
#   %convolution_80 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_143, %arg62_1, %arg63_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_80, %unsqueeze_297), kwargs = {})
#   %mul_228 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_299), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_228, %unsqueeze_301), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_229, %unsqueeze_303), kwargs = {})
#   %add_146 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_145, %arg68_1), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_146, %unsqueeze_305), kwargs = {})
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_307), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_231, %unsqueeze_309), kwargs = {})
#   %add_148 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_232, %unsqueeze_311), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2 + (196*y0)), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr10 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 - tmp20
    tmp23 = tmp22 + tmp6
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tmp9 / tmp24
    tmp26 = tmp25 * tmp11
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp19, xmask)
    tl.store(out_ptr1 + (y0 + (384*x2) + (75264*y1)), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pn/cpngs7yumvf4hmkggv2z44ghm7myo7tosqows23b22xgc26yyq6d.py
# Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_16 => clone_98
# Graph fragment:
#   %clone_98 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_32,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_16 = async_compile.triton('triton_poi_fused_clone_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 196
    x2 = (xindex // 12544) % 6
    x3 = (xindex // 75264)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1152*x1) + (225792*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7x/c7xtfwgx3vtez2zn5uq5bes5zjsln3ksfq7tchulghomgrzgxtku.py
# Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_16 => clone_99
# Graph fragment:
#   %clone_99 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_33,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_17 = async_compile.triton('triton_poi_fused_clone_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (384 + y0 + (1152*x2) + (225792*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6j/c6jz7carbccufoz3kdc4tkts32l2sk5blc3p7rktcbz5ahoqmpv3.py
# Topologically Sorted Source Nodes: [attn_25], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_25 => div_8, exp_8, sum_9
# Graph fragment:
#   %mul_tensor_14 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_68, 1), kwargs = {})
#   %amax_default_7 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_14, [-1], True), kwargs = {})
#   %sub_tensor_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_14, %amax_default_7), kwargs = {})
#   %mul_tensor_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_7, 0.125), kwargs = {})
#   %exp_8 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_15,), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_8, [-1], True), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_8, %sum_9), kwargs = {})
triton_per_fused__softmax_18 = async_compile.triton('triton_per_fused__softmax_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_18(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9408
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
    tmp8 = 0.125
    tmp9 = tmp7 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r1 + (196*x0)), tmp15, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nh/cnhlk4qfcwaw5r3l57ywk6w5ddevqj2ws7dujhtcgirri4yvg26b.py
# Topologically Sorted Source Nodes: [x_229], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_229 => clone_101
# Graph fragment:
#   %clone_101 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_35,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_19 = async_compile.triton('triton_poi_fused_clone_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 196
    x2 = (xindex // 12544) % 6
    x3 = (xindex // 75264)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (64*x2) + (1152*x1) + (225792*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gx/cgxe2ea67b35uao5keer2dgifhzweo5jrygsccbunce7eufo2fkc.py
# Topologically Sorted Source Nodes: [x_230, x_231], Original ATen: [aten.clone, aten.convolution]
# Source node to ATen node mapping:
#   x_230 => clone_102
#   x_231 => convolution_82
# Graph fragment:
#   %clone_102 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_27,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_82 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_72, %arg74_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_clone_convolution_20 = async_compile.triton('triton_poi_fused_clone_convolution_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_20(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y5 = yindex
    y3 = yindex % 384
    y4 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (12544*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr1 + (y3 + (384*x2) + (75264*y4)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pp/cpphgqgqnt6u4t2ck3myh4ok5sjg5pnj5nm2cjxttvnun5oth4fv.py
# Topologically Sorted Source Nodes: [x_233, batch_norm_39], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_39 => add_151, mul_235, mul_236, sub_48
#   x_233 => add_149
# Graph fragment:
#   %add_149 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_146, %convolution_82), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_149, %unsqueeze_313), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_315), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_235, %unsqueeze_317), kwargs = {})
#   %add_151 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_236, %unsqueeze_319), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gi/cginajbexq5ukrbsmxknu2w4a6i52o7luge55x22kmzm4s64cn47.py
# Topologically Sorted Source Nodes: [x_233, x_239, batch_norm_40], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_40 => add_155, mul_241, mul_242, sub_49
#   x_233 => add_149
#   x_239 => add_153
# Graph fragment:
#   %add_149 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_146, %convolution_82), kwargs = {})
#   %add_153 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_149, %convolution_84), kwargs = {})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_153, %unsqueeze_321), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_323), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_241, %unsqueeze_325), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_242, %unsqueeze_327), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2k/c2khxmtutxls6cmmk6zqetzl3i6don34dwcrv7a2k3cqh5j2ppko.py
# Topologically Sorted Source Nodes: [x_233, x_239, x_245, batch_norm_41], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_41 => add_158, mul_245, mul_246, sub_51
#   x_233 => add_149
#   x_239 => add_153
#   x_245 => add_156
# Graph fragment:
#   %add_149 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_146, %convolution_82), kwargs = {})
#   %add_153 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_149, %convolution_84), kwargs = {})
#   %add_156 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_153, %convolution_86), kwargs = {})
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_156, %unsqueeze_329), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_331), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_245, %unsqueeze_333), kwargs = {})
#   %add_158 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_246, %unsqueeze_335), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1, 1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp21, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hp/chpmhyewhhi4nqmmu4yj6dttueussmcw2ikteyvxxharu5aac2to.py
# Topologically Sorted Source Nodes: [x_233, x_239, x_245, x_251, batch_norm_42], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_42 => add_162, mul_251, mul_252, sub_52
#   x_233 => add_149
#   x_239 => add_153
#   x_245 => add_156
#   x_251 => add_160
# Graph fragment:
#   %add_149 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_146, %convolution_82), kwargs = {})
#   %add_153 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_149, %convolution_84), kwargs = {})
#   %add_156 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_153, %convolution_86), kwargs = {})
#   %add_160 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_156, %convolution_88), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_160, %unsqueeze_337), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_339), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_251, %unsqueeze_341), kwargs = {})
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_252, %unsqueeze_343), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr1 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1, 1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (384*y3)), tmp8, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp23, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bb/cbbwvlmzzvq6kgjxo7vvl26hyg2symjhbunzcodf6n3wxbww7j2y.py
# Topologically Sorted Source Nodes: [x_257, batch_norm_43], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_43 => add_165, mul_255, mul_256, sub_54
#   x_257 => add_163
# Graph fragment:
#   %add_163 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_160, %convolution_90), kwargs = {})
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_163, %unsqueeze_345), kwargs = {})
#   %mul_255 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_347), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_255, %unsqueeze_349), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_256, %unsqueeze_351), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zh/czh3fyzbexuat3r4vqqng43bue4jqkd33xy7k6vthvmjiihwf3an.py
# Topologically Sorted Source Nodes: [x_257, x_263, batch_norm_44], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_44 => add_169, mul_261, mul_262, sub_55
#   x_257 => add_163
#   x_263 => add_167
# Graph fragment:
#   %add_163 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_160, %convolution_90), kwargs = {})
#   %add_167 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_163, %convolution_92), kwargs = {})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_167, %unsqueeze_353), kwargs = {})
#   %mul_261 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_355), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_261, %unsqueeze_357), kwargs = {})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_262, %unsqueeze_359), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2m/c2mdyurk4elefjpkxwjp5r3yuwuj5uyncjewdlnrvxsnsbube7bt.py
# Topologically Sorted Source Nodes: [x_257, x_263, x_269, batch_norm_45], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_45 => add_172, mul_265, mul_266, sub_57
#   x_257 => add_163
#   x_263 => add_167
#   x_269 => add_170
# Graph fragment:
#   %add_163 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_160, %convolution_90), kwargs = {})
#   %add_167 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_163, %convolution_92), kwargs = {})
#   %add_170 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_167, %convolution_94), kwargs = {})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_170, %unsqueeze_361), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_363), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_265, %unsqueeze_365), kwargs = {})
#   %add_172 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_266, %unsqueeze_367), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uv/cuvv3nnsastrsekpb3hp3sbmfn7kj3fq62rhahihoe7xmscglrdn.py
# Topologically Sorted Source Nodes: [x_257, x_263, x_269, x_275], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_257 => add_163
#   x_263 => add_167
#   x_269 => add_170
#   x_275 => add_174
# Graph fragment:
#   %add_163 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_160, %convolution_90), kwargs = {})
#   %add_167 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_163, %convolution_92), kwargs = {})
#   %add_170 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_167, %convolution_94), kwargs = {})
#   %add_174 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_170, %convolution_96), kwargs = {})
triton_poi_fused_add_28 = async_compile.triton('triton_poi_fused_add_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ro/crop6frq7u6yvgrdzu6eiygggjznpfdz5ayactzhp3t5s6nd33cd.py
# Topologically Sorted Source Nodes: [x_257, x_263, x_269, x_275, x_276], Original ATen: [aten.add, aten.convolution]
# Source node to ATen node mapping:
#   x_257 => add_163
#   x_263 => add_167
#   x_269 => add_170
#   x_275 => add_174
#   x_276 => convolution_97
# Graph fragment:
#   %add_163 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_160, %convolution_90), kwargs = {})
#   %add_167 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_163, %convolution_92), kwargs = {})
#   %add_170 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_167, %convolution_94), kwargs = {})
#   %add_174 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_170, %convolution_96), kwargs = {})
#   %convolution_97 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_174, %arg117_1, %arg118_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_29 = async_compile.triton('triton_poi_fused_add_convolution_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 294912
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (1536*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2o/c2ojjvyqoztte55f3ussofwvqyljifz2fcxerun4pdtb5vkyfmk6.py
# Topologically Sorted Source Nodes: [x_257, x_263, x_269, x_275, x_276, x_277, add_43, batch_norm_47], Original ATen: [aten.add, aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   add_43 => add_177
#   batch_norm_47 => add_179, mul_274, mul_275, sub_59
#   x_257 => add_163
#   x_263 => add_167
#   x_269 => add_170
#   x_275 => add_174
#   x_276 => convolution_97
#   x_277 => add_176, mul_271, mul_272, sub_58
# Graph fragment:
#   %add_163 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_160, %convolution_90), kwargs = {})
#   %add_167 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_163, %convolution_92), kwargs = {})
#   %add_170 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_167, %convolution_94), kwargs = {})
#   %add_174 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_170, %convolution_96), kwargs = {})
#   %convolution_97 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_174, %arg117_1, %arg118_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_97, %unsqueeze_369), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_371), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %unsqueeze_373), kwargs = {})
#   %add_176 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_272, %unsqueeze_375), kwargs = {})
#   %add_177 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_176, %arg123_1), kwargs = {})
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_177, %unsqueeze_377), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_379), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_274, %unsqueeze_381), kwargs = {})
#   %add_179 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_275, %unsqueeze_383), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2 + (49*y0)), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr10 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 - tmp20
    tmp23 = tmp22 + tmp6
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tmp9 / tmp24
    tmp26 = tmp25 * tmp11
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp19, xmask)
    tl.store(out_ptr1 + (y0 + (768*x2) + (37632*y1)), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/56/c56qokof7wl4e2tbyuxtuwyskpmwvksdvd3szbnfzhz7qixehb2f.py
# Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_24 => clone_131
# Graph fragment:
#   %clone_131 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_48,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_31 = async_compile.triton('triton_poi_fused_clone_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 49
    x2 = (xindex // 6272) % 6
    x3 = (xindex // 37632)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x2) + (2304*x1) + (112896*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mo/cmovg5pzr22ttegbv7efo2olo4acthtqztdvy2r7sdzgwmyekjme.py
# Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_24 => clone_132
# Graph fragment:
#   %clone_132 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_49,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_32 = async_compile.triton('triton_poi_fused_clone_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (768 + y0 + (2304*x2) + (112896*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/my/cmyl46cato5dhklpzifxrjr26fu3mhyplh33mlxzmivq46y7j35l.py
# Topologically Sorted Source Nodes: [attn_37], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_37 => div_12, exp_12, sum_13
# Graph fragment:
#   %mul_tensor_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_100, 1), kwargs = {})
#   %amax_default_3 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_6, [-1], True), kwargs = {})
#   %sub_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_6, %amax_default_3), kwargs = {})
#   %mul_tensor_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_3, 0.08838834764831845), kwargs = {})
#   %exp_12 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_7,), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_12, [-1], True), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_12, %sum_13), kwargs = {})
triton_per_fused__softmax_33 = async_compile.triton('triton_per_fused__softmax_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_33(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = 0.08838834764831845
    tmp9 = tmp7 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r1 + (49*x0)), tmp15, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/df/cdfd7sebxnb3cq5c5tirdsqdnapl6ysv7hb3fnus56zxviswm745.py
# Topologically Sorted Source Nodes: [x_280], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_280 => clone_134
# Graph fragment:
#   %clone_134 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_51,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_34 = async_compile.triton('triton_poi_fused_clone_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 49
    x2 = (xindex // 6272) % 6
    x3 = (xindex // 37632)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + (128*x2) + (2304*x1) + (112896*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kf/ckf2f2id4hnoz6tofy2pa5z55jdnkj72c4uqqlch2e5pahpfbi56.py
# Topologically Sorted Source Nodes: [x_281, x_282], Original ATen: [aten.clone, aten.convolution]
# Source node to ATen node mapping:
#   x_281 => clone_135
#   x_282 => convolution_99
# Graph fragment:
#   %clone_135 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_39,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_99 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_104, %arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_clone_convolution_35 = async_compile.triton('triton_poi_fused_clone_convolution_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_35(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y5 = yindex
    y3 = yindex % 768
    y4 = (yindex // 768)
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (6272*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr1 + (y3 + (768*x2) + (37632*y4)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jm/cjm5ivwptomo53glohaksh56zadk4cjz4qxzgdtbsgq2qsduxx2g.py
# Topologically Sorted Source Nodes: [x_284, batch_norm_48], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_48 => add_182, mul_278, mul_279, sub_61
#   x_284 => add_180
# Graph fragment:
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_177, %convolution_99), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_180, %unsqueeze_385), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_387), kwargs = {})
#   %mul_279 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_278, %unsqueeze_389), kwargs = {})
#   %add_182 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_279, %unsqueeze_391), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (37632*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/l3/cl3hxzkvqpe2iymugh54ytizwhadewefzds56cnwqizv3ve5jcu3.py
# Topologically Sorted Source Nodes: [x_286], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_286 => add_183, erf_40, mul_280, mul_281, mul_282
# Graph fragment:
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_100, 0.5), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_100, 0.7071067811865476), kwargs = {})
#   %erf_40 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_281,), kwargs = {})
#   %add_183 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_40, 1), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_280, %add_183), kwargs = {})
triton_poi_fused_gelu_37 = async_compile.triton('triton_poi_fused_gelu_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_37(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xg/cxgx6jvzn2iuyml65oucs44xgka3eehsiqepjmq74qyslacnlpeh.py
# Topologically Sorted Source Nodes: [x_284, x_290, batch_norm_49], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_49 => add_186, mul_284, mul_285, sub_62
#   x_284 => add_180
#   x_290 => add_184
# Graph fragment:
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_177, %convolution_99), kwargs = {})
#   %add_184 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_180, %convolution_101), kwargs = {})
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_184, %unsqueeze_393), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %unsqueeze_395), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_284, %unsqueeze_397), kwargs = {})
#   %add_186 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_285, %unsqueeze_399), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (37632*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ar/carf7gezl5faxarkh5brwvzk6al7cjzi756x5zoktdglbpmjcila.py
# Topologically Sorted Source Nodes: [x_284, x_290, x_296, batch_norm_50], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_50 => add_189, mul_288, mul_289, sub_64
#   x_284 => add_180
#   x_290 => add_184
#   x_296 => add_187
# Graph fragment:
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_177, %convolution_99), kwargs = {})
#   %add_184 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_180, %convolution_101), kwargs = {})
#   %add_187 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_184, %convolution_103), kwargs = {})
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_187, %unsqueeze_401), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_403), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_288, %unsqueeze_405), kwargs = {})
#   %add_189 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_289, %unsqueeze_407), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (37632*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1, 1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp21, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/j2/cj2mvmovsxa3xagfh7xgjnesxz4rmrtagzlubmfgrz4zspmj5lxo.py
# Topologically Sorted Source Nodes: [x_284, x_290, x_296, x_302, batch_norm_51], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_51 => add_193, mul_294, mul_295, sub_65
#   x_284 => add_180
#   x_290 => add_184
#   x_296 => add_187
#   x_302 => add_191
# Graph fragment:
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_177, %convolution_99), kwargs = {})
#   %add_184 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_180, %convolution_101), kwargs = {})
#   %add_187 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_184, %convolution_103), kwargs = {})
#   %add_191 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_187, %convolution_105), kwargs = {})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_191, %unsqueeze_409), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_411), kwargs = {})
#   %mul_295 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_294, %unsqueeze_413), kwargs = {})
#   %add_193 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_295, %unsqueeze_415), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_40', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (37632*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1, 1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (768*y3)), tmp8, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp23, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/en/cenvxdfoeuwfpv732yc34zfjy6kupuk2rxfpcaztjobisagg2aky.py
# Topologically Sorted Source Nodes: [x_308, batch_norm_52], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_52 => add_196, mul_298, mul_299, sub_67
#   x_308 => add_194
# Graph fragment:
#   %add_194 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_191, %convolution_107), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_194, %unsqueeze_417), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_419), kwargs = {})
#   %mul_299 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_298, %unsqueeze_421), kwargs = {})
#   %add_196 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_299, %unsqueeze_423), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/go/cgobyic7iodefu7vrxl3edeta22sopsa4fjuzxiz4pkshzcnwfcg.py
# Topologically Sorted Source Nodes: [x_308, x_314, batch_norm_53], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_53 => add_200, mul_304, mul_305, sub_68
#   x_308 => add_194
#   x_314 => add_198
# Graph fragment:
#   %add_194 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_191, %convolution_107), kwargs = {})
#   %add_198 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_194, %convolution_109), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_198, %unsqueeze_425), kwargs = {})
#   %mul_304 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_427), kwargs = {})
#   %mul_305 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_304, %unsqueeze_429), kwargs = {})
#   %add_200 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_305, %unsqueeze_431), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7s/c7smwersekq6rjd3bpox2y4iurmszh7cgbnu5sk6cuddfp7iihoh.py
# Topologically Sorted Source Nodes: [x_308, x_314, x_320, batch_norm_54], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_54 => add_203, mul_308, mul_309, sub_70
#   x_308 => add_194
#   x_314 => add_198
#   x_320 => add_201
# Graph fragment:
#   %add_194 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_191, %convolution_107), kwargs = {})
#   %add_198 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_194, %convolution_109), kwargs = {})
#   %add_201 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_198, %convolution_111), kwargs = {})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_201, %unsqueeze_433), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_435), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_308, %unsqueeze_437), kwargs = {})
#   %add_203 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_309, %unsqueeze_439), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp5 = tl.load(in_ptr3 + (x2), xmask)
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qq/cqq4vrtcuszqlfzqrip67wfcep54jdzt4dwk2lzhm5gnmjgfiw26.py
# Topologically Sorted Source Nodes: [x_308, x_314, x_320, x_326, x_327], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_308 => add_194
#   x_314 => add_198
#   x_320 => add_201
#   x_326 => add_205
#   x_327 => add_207, mul_314, mul_315, sub_71
# Graph fragment:
#   %add_194 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_191, %convolution_107), kwargs = {})
#   %add_198 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_194, %convolution_109), kwargs = {})
#   %add_201 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_198, %convolution_111), kwargs = {})
#   %add_205 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_201, %convolution_113), kwargs = {})
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_205, %unsqueeze_441), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_443), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_314, %unsqueeze_445), kwargs = {})
#   %add_207 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_315, %unsqueeze_447), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp5 = tl.load(in_ptr2 + (x2), xmask)
    tmp7 = tl.load(in_ptr3 + (x2), xmask)
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tl.store(in_out_ptr0 + (x2), tmp23, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4b/c4byvft7lmzijzxnhnfvbomot76ez3fuuvbg6jhdxcdyb2hzdk23.py
# Topologically Sorted Source Nodes: [x_328], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_328 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_207, [-1, -2], True), kwargs = {})
triton_per_fused_mean_45 = async_compile.triton('triton_per_fused_mean_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_45(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    x0 = xindex % 768
    x1 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (37632*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (192, 32, 4, 4), (512, 16, 4, 1))
    assert_size_stride(arg7_1, (192, ), (1, ))
    assert_size_stride(arg8_1, (192, ), (1, ))
    assert_size_stride(arg9_1, (192, ), (1, ))
    assert_size_stride(arg10_1, (192, ), (1, ))
    assert_size_stride(arg11_1, (192, ), (1, ))
    assert_size_stride(arg12_1, (1, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(arg13_1, (192, ), (1, ))
    assert_size_stride(arg14_1, (192, ), (1, ))
    assert_size_stride(arg15_1, (192, ), (1, ))
    assert_size_stride(arg16_1, (192, ), (1, ))
    assert_size_stride(arg17_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg18_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg19_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg20_1, (192, ), (1, ))
    assert_size_stride(arg21_1, (192, ), (1, ))
    assert_size_stride(arg22_1, (192, ), (1, ))
    assert_size_stride(arg23_1, (192, ), (1, ))
    assert_size_stride(arg24_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg25_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg26_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg27_1, (192, ), (1, ))
    assert_size_stride(arg28_1, (192, ), (1, ))
    assert_size_stride(arg29_1, (192, ), (1, ))
    assert_size_stride(arg30_1, (192, ), (1, ))
    assert_size_stride(arg31_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg32_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg33_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg34_1, (192, ), (1, ))
    assert_size_stride(arg35_1, (192, ), (1, ))
    assert_size_stride(arg36_1, (192, ), (1, ))
    assert_size_stride(arg37_1, (192, ), (1, ))
    assert_size_stride(arg38_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg39_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg40_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg41_1, (192, ), (1, ))
    assert_size_stride(arg42_1, (192, ), (1, ))
    assert_size_stride(arg43_1, (192, ), (1, ))
    assert_size_stride(arg44_1, (192, ), (1, ))
    assert_size_stride(arg45_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg46_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg47_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg48_1, (192, ), (1, ))
    assert_size_stride(arg49_1, (192, ), (1, ))
    assert_size_stride(arg50_1, (192, ), (1, ))
    assert_size_stride(arg51_1, (192, ), (1, ))
    assert_size_stride(arg52_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg53_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg54_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg55_1, (192, ), (1, ))
    assert_size_stride(arg56_1, (192, ), (1, ))
    assert_size_stride(arg57_1, (192, ), (1, ))
    assert_size_stride(arg58_1, (192, ), (1, ))
    assert_size_stride(arg59_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg60_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg61_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg62_1, (384, 192, 2, 2), (768, 4, 2, 1))
    assert_size_stride(arg63_1, (384, ), (1, ))
    assert_size_stride(arg64_1, (384, ), (1, ))
    assert_size_stride(arg65_1, (384, ), (1, ))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (384, ), (1, ))
    assert_size_stride(arg68_1, (1, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(arg69_1, (384, ), (1, ))
    assert_size_stride(arg70_1, (384, ), (1, ))
    assert_size_stride(arg71_1, (384, ), (1, ))
    assert_size_stride(arg72_1, (384, ), (1, ))
    assert_size_stride(arg73_1, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg74_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (384, ), (1, ))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (384, ), (1, ))
    assert_size_stride(arg79_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg80_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg81_1, (384, ), (1, ))
    assert_size_stride(arg82_1, (384, ), (1, ))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (384, ), (1, ))
    assert_size_stride(arg85_1, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg86_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg87_1, (384, ), (1, ))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg92_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg93_1, (384, ), (1, ))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (384, ), (1, ))
    assert_size_stride(arg97_1, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg98_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (384, ), (1, ))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (384, ), (1, ))
    assert_size_stride(arg103_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg104_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (384, ), (1, ))
    assert_size_stride(arg107_1, (384, ), (1, ))
    assert_size_stride(arg108_1, (384, ), (1, ))
    assert_size_stride(arg109_1, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg110_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (384, ), (1, ))
    assert_size_stride(arg114_1, (384, ), (1, ))
    assert_size_stride(arg115_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg116_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg117_1, (768, 384, 2, 2), (1536, 4, 2, 1))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (1, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg129_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg135_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg141_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg147_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg153_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg159_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg165_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg171_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (1000, 768), (768, 1))
    assert_size_stride(arg177_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((32, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 96, 49, grid=grid(96, 49), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((192, 32, 4, 4), (512, 1, 128, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg6_1, buf4, 6144, 16, grid=grid(6144, 16), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [input_5, input_6, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del buf3
        del buf4
        buf6 = reinterpret_tensor(buf0, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf0  # reuse
        buf7 = empty_strided_cuda((8, 192, 28, 28), (150528, 1, 5376, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6, x_166, x_167, add_26, batch_norm_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_4.run(buf5, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, buf6, buf7, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del arg16_1
        del arg7_1
        del arg8_1
        del arg9_1
        del buf5
        # Topologically Sorted Source Nodes: [batch_norm_30, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg17_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del arg17_1
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_170], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf9, 2408448, grid=grid(2408448), stream=stream0)
        buf10 = empty_strided_cuda((384, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_170, x_172], Original ATen: [aten.gelu, aten.convolution]
        triton_poi_fused_convolution_gelu_6.run(arg18_1, buf10, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg18_1
        # Topologically Sorted Source Nodes: [x_170, x_172], Original ATen: [aten.gelu, aten.convolution]
        buf11 = extern_kernels.convolution(buf9, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf11, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del buf9
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_173], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf12, 2408448, grid=grid(2408448), stream=stream0)
        # Topologically Sorted Source Nodes: [x_173, x_174], Original ATen: [aten.gelu, aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg19_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg19_1
        del buf12
        buf14 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_176, batch_norm_31], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_7.run(buf6, buf13, arg20_1, arg21_1, arg22_1, arg23_1, buf14, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg20_1
        del arg21_1
        del arg22_1
        del arg23_1
        # Topologically Sorted Source Nodes: [x_176, batch_norm_31, x_177], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del arg24_1
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_178], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf16, 2408448, grid=grid(2408448), stream=stream0)
        buf17 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_178, x_180], Original ATen: [aten.gelu, aten.convolution]
        triton_poi_fused_convolution_gelu_6.run(arg25_1, buf17, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg25_1
        # Topologically Sorted Source Nodes: [x_178, x_180], Original ATen: [aten.gelu, aten.convolution]
        buf18 = extern_kernels.convolution(buf16, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf18, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del buf16
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_181], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf19, 2408448, grid=grid(2408448), stream=stream0)
        # Topologically Sorted Source Nodes: [x_181, x_182], Original ATen: [aten.gelu, aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg26_1
        del buf19
        buf21 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_176, x_184, batch_norm_32], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf6, buf13, buf20, arg27_1, arg28_1, arg29_1, arg30_1, buf21, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        # Topologically Sorted Source Nodes: [x_176, x_184, batch_norm_32, x_185], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del arg31_1
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_186], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf23, 2408448, grid=grid(2408448), stream=stream0)
        buf24 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [x_186, x_188], Original ATen: [aten.gelu, aten.convolution]
        triton_poi_fused_convolution_gelu_6.run(arg32_1, buf24, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg32_1
        # Topologically Sorted Source Nodes: [x_186, x_188], Original ATen: [aten.gelu, aten.convolution]
        buf25 = extern_kernels.convolution(buf23, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf25, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del buf23
        buf26 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_189], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf26, 2408448, grid=grid(2408448), stream=stream0)
        # Topologically Sorted Source Nodes: [x_189, x_190], Original ATen: [aten.gelu, aten.convolution]
        buf27 = extern_kernels.convolution(buf26, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg33_1
        del buf26
        buf28 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_176, x_184, x_192, batch_norm_33], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_9.run(buf6, buf13, buf20, buf27, arg34_1, arg35_1, arg36_1, arg37_1, buf28, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        del arg37_1
        # Topologically Sorted Source Nodes: [x_176, x_184, x_192, batch_norm_33, x_193], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg38_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del arg38_1
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_194], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf30, 2408448, grid=grid(2408448), stream=stream0)
        buf31 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_194, x_196], Original ATen: [aten.gelu, aten.convolution]
        triton_poi_fused_convolution_gelu_6.run(arg39_1, buf31, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg39_1
        # Topologically Sorted Source Nodes: [x_194, x_196], Original ATen: [aten.gelu, aten.convolution]
        buf32 = extern_kernels.convolution(buf30, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf32, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del buf30
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_197], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf33, 2408448, grid=grid(2408448), stream=stream0)
        # Topologically Sorted Source Nodes: [x_197, x_198], Original ATen: [aten.gelu, aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg40_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg40_1
        del buf33
        buf35 = buf13; del buf13  # reuse
        buf36 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_176, x_184, x_192, x_200, batch_norm_34], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf35, buf6, buf20, buf27, buf34, arg41_1, arg42_1, arg43_1, arg44_1, buf36, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg41_1
        del arg42_1
        del arg43_1
        del arg44_1
        del buf20
        del buf27
        del buf34
        del buf6
        # Topologically Sorted Source Nodes: [batch_norm_34, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg45_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del arg45_1
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_202], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf38, 2408448, grid=grid(2408448), stream=stream0)
        buf39 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_202, x_204], Original ATen: [aten.gelu, aten.convolution]
        triton_poi_fused_convolution_gelu_6.run(arg46_1, buf39, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg46_1
        # Topologically Sorted Source Nodes: [x_202, x_204], Original ATen: [aten.gelu, aten.convolution]
        buf40 = extern_kernels.convolution(buf38, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf40, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del buf38
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_205], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf41, 2408448, grid=grid(2408448), stream=stream0)
        # Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten.gelu, aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg47_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg47_1
        del buf41
        buf43 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_208, batch_norm_35], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf35, buf42, arg48_1, arg49_1, arg50_1, arg51_1, buf43, 1204224, grid=grid(1204224), stream=stream0)
        del arg48_1
        del arg49_1
        del arg50_1
        del arg51_1
        # Topologically Sorted Source Nodes: [x_208, batch_norm_35, x_209], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg52_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del arg52_1
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [x_210], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf45, 2408448, grid=grid(2408448), stream=stream0)
        buf46 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_210, x_212], Original ATen: [aten.gelu, aten.convolution]
        triton_poi_fused_convolution_gelu_6.run(arg53_1, buf46, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg53_1
        # Topologically Sorted Source Nodes: [x_210, x_212], Original ATen: [aten.gelu, aten.convolution]
        buf47 = extern_kernels.convolution(buf45, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf47, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del buf45
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_213], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf48, 2408448, grid=grid(2408448), stream=stream0)
        # Topologically Sorted Source Nodes: [x_213, x_214], Original ATen: [aten.gelu, aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg54_1
        del buf48
        buf50 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_208, x_216, batch_norm_36], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_12.run(buf35, buf42, buf49, arg55_1, arg56_1, arg57_1, arg58_1, buf50, 1204224, grid=grid(1204224), stream=stream0)
        del arg55_1
        del arg56_1
        del arg57_1
        del arg58_1
        # Topologically Sorted Source Nodes: [x_208, x_216, batch_norm_36, x_217], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg59_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del arg59_1
        del buf50
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_218], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf52, 2408448, grid=grid(2408448), stream=stream0)
        buf53 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_218, x_220], Original ATen: [aten.gelu, aten.convolution]
        triton_poi_fused_convolution_gelu_6.run(arg60_1, buf53, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg60_1
        # Topologically Sorted Source Nodes: [x_218, x_220], Original ATen: [aten.gelu, aten.convolution]
        buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf54, (8, 384, 28, 28), (301056, 1, 10752, 384))
        del buf52
        del buf53
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf55, 2408448, grid=grid(2408448), stream=stream0)
        # Topologically Sorted Source Nodes: [x_221, x_222], Original ATen: [aten.gelu, aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg61_1
        del buf55
        buf57 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [x_208, x_216, x_224], Original ATen: [aten.add]
        triton_poi_fused_add_13.run(buf57, buf42, buf49, buf56, 1204224, grid=grid(1204224), stream=stream0)
        del buf42
        del buf49
        del buf56
        buf58 = empty_strided_cuda((384, 192, 2, 2), (768, 1, 384, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_208, x_216, x_224, x_225], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_add_convolution_14.run(arg62_1, buf58, 73728, 4, grid=grid(73728, 4), stream=stream0)
        del arg62_1
        # Topologically Sorted Source Nodes: [x_208, x_216, x_224, x_225], Original ATen: [aten.add, aten.convolution]
        buf59 = extern_kernels.convolution(buf57, buf58, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del buf57
        del buf58
        buf60 = empty_strided_cuda((8, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
        buf61 = empty_strided_cuda((8, 384, 14, 14), (75264, 1, 5376, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_208, x_216, x_224, x_225, x_226, add_34, batch_norm_38], Original ATen: [aten.add, aten.convolution, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_15.run(buf59, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, buf60, buf61, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg63_1
        del arg64_1
        del arg65_1
        del arg66_1
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        del arg71_1
        del arg72_1
        # Topologically Sorted Source Nodes: [batch_norm_38, conv2d_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg73_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 1152, 14, 14), (225792, 1, 16128, 1152))
        del arg73_1
        buf63 = reinterpret_tensor(buf61, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf62, buf63, 602112, grid=grid(602112), stream=stream0)
        buf64 = reinterpret_tensor(buf59, (8, 6, 64, 196), (75264, 12544, 196, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf62, buf64, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf65 = empty_strided_cuda((48, 196, 196), (38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf63, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf64, (48, 64, 196), (12544, 196, 1), 0), out=buf65)
        buf68 = empty_strided_cuda((8, 6, 196, 196), (230496, 38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_25], Original ATen: [aten._softmax]
        triton_per_fused__softmax_18.run(buf65, buf68, 9408, 196, grid=grid(9408), stream=stream0)
        buf69 = reinterpret_tensor(buf64, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_229], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf62, buf69, 602112, grid=grid(602112), stream=stream0)
        del buf62
        buf70 = reinterpret_tensor(buf63, (48, 196, 64), (12544, 64, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [x_229], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf69, (48, 196, 64), (12544, 64, 1), 0), out=buf70)
        buf72 = reinterpret_tensor(buf69, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_230, x_231], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_20.run(buf70, buf72, 3072, 196, grid=grid(3072, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [x_231], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg74_1
        buf74 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_233, batch_norm_39], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf60, buf73, arg75_1, arg76_1, arg77_1, arg78_1, buf74, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg75_1
        del arg76_1
        del arg77_1
        del arg78_1
        # Topologically Sorted Source Nodes: [x_233, batch_norm_39, x_234], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg79_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg79_1
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_235], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf76, 2408448, grid=grid(2408448), stream=stream0)
        # Topologically Sorted Source Nodes: [x_235, x_237], Original ATen: [aten.gelu, aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg80_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg80_1
        del buf76
        buf78 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_233, x_239, batch_norm_40], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_22.run(buf60, buf73, buf77, arg81_1, arg82_1, arg83_1, arg84_1, buf78, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg81_1
        del arg82_1
        del arg83_1
        del arg84_1
        # Topologically Sorted Source Nodes: [x_233, x_239, batch_norm_40, conv2d_85], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf79 = extern_kernels.convolution(buf78, arg85_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 1152, 14, 14), (225792, 1, 16128, 1152))
        del arg85_1
        buf80 = reinterpret_tensor(buf78, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf79, buf80, 602112, grid=grid(602112), stream=stream0)
        buf81 = reinterpret_tensor(buf70, (8, 6, 64, 196), (75264, 12544, 196, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf79, buf81, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf82 = reinterpret_tensor(buf68, (48, 196, 196), (38416, 196, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf80, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf81, (48, 64, 196), (12544, 196, 1), 0), out=buf82)
        buf85 = reinterpret_tensor(buf65, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [attn_28], Original ATen: [aten._softmax]
        triton_per_fused__softmax_18.run(buf82, buf85, 9408, 196, grid=grid(9408), stream=stream0)
        buf86 = reinterpret_tensor(buf81, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_241], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf79, buf86, 602112, grid=grid(602112), stream=stream0)
        del buf79
        buf87 = reinterpret_tensor(buf80, (48, 196, 64), (12544, 64, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_241], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf85, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf86, (48, 196, 64), (12544, 64, 1), 0), out=buf87)
        buf89 = reinterpret_tensor(buf86, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_242, x_243], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_20.run(buf87, buf89, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del buf87
        # Topologically Sorted Source Nodes: [x_243], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg86_1
        buf91 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [x_233, x_239, x_245, batch_norm_41], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf60, buf73, buf77, buf90, arg87_1, arg88_1, arg89_1, arg90_1, buf91, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        # Topologically Sorted Source Nodes: [x_233, x_239, x_245, batch_norm_41, x_246], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg91_1
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_247], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf93, 2408448, grid=grid(2408448), stream=stream0)
        # Topologically Sorted Source Nodes: [x_247, x_249], Original ATen: [aten.gelu, aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg92_1
        del buf93
        buf95 = buf73; del buf73  # reuse
        buf96 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_233, x_239, x_245, x_251, batch_norm_42], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_24.run(buf95, buf60, buf77, buf90, buf94, arg93_1, arg94_1, arg95_1, arg96_1, buf96, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg93_1
        del arg94_1
        del arg95_1
        del arg96_1
        del buf60
        del buf77
        del buf90
        # Topologically Sorted Source Nodes: [batch_norm_42, conv2d_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 1152, 14, 14), (225792, 1, 16128, 1152))
        del arg97_1
        buf98 = reinterpret_tensor(buf96, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf97, buf98, 602112, grid=grid(602112), stream=stream0)
        buf99 = reinterpret_tensor(buf94, (8, 6, 64, 196), (75264, 12544, 196, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf97, buf99, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf100 = reinterpret_tensor(buf85, (48, 196, 196), (38416, 196, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf98, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf99, (48, 64, 196), (12544, 196, 1), 0), out=buf100)
        buf103 = reinterpret_tensor(buf82, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [attn_31], Original ATen: [aten._softmax]
        triton_per_fused__softmax_18.run(buf100, buf103, 9408, 196, grid=grid(9408), stream=stream0)
        buf104 = reinterpret_tensor(buf99, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [x_253], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf97, buf104, 602112, grid=grid(602112), stream=stream0)
        del buf97
        buf105 = reinterpret_tensor(buf98, (48, 196, 64), (12544, 64, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [x_253], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf103, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf104, (48, 196, 64), (12544, 64, 1), 0), out=buf105)
        buf107 = reinterpret_tensor(buf104, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_254, x_255], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_20.run(buf105, buf107, 3072, 196, grid=grid(3072, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [x_255], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg98_1
        buf109 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [x_257, batch_norm_43], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_25.run(buf95, buf108, arg99_1, arg100_1, arg101_1, arg102_1, buf109, 602112, grid=grid(602112), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        del arg99_1
        # Topologically Sorted Source Nodes: [x_257, batch_norm_43, x_258], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg103_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg103_1
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_259], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf111, 2408448, grid=grid(2408448), stream=stream0)
        # Topologically Sorted Source Nodes: [x_259, x_261], Original ATen: [aten.gelu, aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg104_1
        del buf111
        buf113 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_257, x_263, batch_norm_44], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf95, buf108, buf112, arg105_1, arg106_1, arg107_1, arg108_1, buf113, 602112, grid=grid(602112), stream=stream0)
        del arg105_1
        del arg106_1
        del arg107_1
        del arg108_1
        # Topologically Sorted Source Nodes: [x_257, x_263, batch_norm_44, conv2d_93], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf114 = extern_kernels.convolution(buf113, arg109_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 1152, 14, 14), (225792, 1, 16128, 1152))
        del arg109_1
        buf115 = reinterpret_tensor(buf113, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf114, buf115, 602112, grid=grid(602112), stream=stream0)
        buf116 = reinterpret_tensor(buf105, (8, 6, 64, 196), (75264, 12544, 196, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf114, buf116, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf117 = reinterpret_tensor(buf103, (48, 196, 196), (38416, 196, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf115, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf116, (48, 64, 196), (12544, 196, 1), 0), out=buf117)
        buf120 = reinterpret_tensor(buf100, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [attn_34], Original ATen: [aten._softmax]
        triton_per_fused__softmax_18.run(buf117, buf120, 9408, 196, grid=grid(9408), stream=stream0)
        del buf117
        buf121 = reinterpret_tensor(buf116, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_265], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf114, buf121, 602112, grid=grid(602112), stream=stream0)
        del buf114
        buf122 = reinterpret_tensor(buf115, (48, 196, 64), (12544, 64, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_265], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf121, (48, 196, 64), (12544, 64, 1), 0), out=buf122)
        del buf120
        buf124 = reinterpret_tensor(buf121, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [x_266, x_267], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_20.run(buf122, buf124, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del buf122
        # Topologically Sorted Source Nodes: [x_267], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg110_1
        buf126 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_257, x_263, x_269, batch_norm_45], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_27.run(buf95, buf108, buf112, buf125, arg111_1, arg112_1, arg113_1, arg114_1, buf126, 602112, grid=grid(602112), stream=stream0)
        del arg111_1
        del arg112_1
        del arg113_1
        del arg114_1
        # Topologically Sorted Source Nodes: [x_257, x_263, x_269, batch_norm_45, x_270], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf127 = extern_kernels.convolution(buf126, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
        del arg115_1
        del buf126
        buf128 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_271], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf128, 2408448, grid=grid(2408448), stream=stream0)
        # Topologically Sorted Source Nodes: [x_271, x_273], Original ATen: [aten.gelu, aten.convolution]
        buf129 = extern_kernels.convolution(buf128, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg116_1
        del buf128
        buf130 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_257, x_263, x_269, x_275], Original ATen: [aten.add]
        triton_poi_fused_add_28.run(buf130, buf95, buf112, buf125, buf129, 602112, grid=grid(602112), stream=stream0)
        del buf112
        del buf125
        del buf129
        del buf95
        buf131 = empty_strided_cuda((768, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_257, x_263, x_269, x_275, x_276], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_add_convolution_29.run(arg117_1, buf131, 294912, 4, grid=grid(294912, 4), stream=stream0)
        del arg117_1
        # Topologically Sorted Source Nodes: [x_257, x_263, x_269, x_275, x_276], Original ATen: [aten.add, aten.convolution]
        buf132 = extern_kernels.convolution(buf130, buf131, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del buf130
        del buf131
        buf133 = empty_strided_cuda((8, 768, 7, 7), (37632, 49, 7, 1), torch.float32)
        buf134 = empty_strided_cuda((8, 768, 7, 7), (37632, 1, 5376, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_257, x_263, x_269, x_275, x_276, x_277, add_43, batch_norm_47], Original ATen: [aten.add, aten.convolution, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_30.run(buf132, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, buf133, buf134, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg118_1
        del arg119_1
        del arg120_1
        del arg121_1
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        del arg126_1
        del arg127_1
        # Topologically Sorted Source Nodes: [batch_norm_47, conv2d_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf135 = extern_kernels.convolution(buf134, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (8, 2304, 7, 7), (112896, 1, 16128, 2304))
        del arg128_1
        buf136 = reinterpret_tensor(buf134, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf135, buf136, 301056, grid=grid(301056), stream=stream0)
        buf137 = reinterpret_tensor(buf132, (8, 6, 128, 49), (37632, 6272, 49, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf135, buf137, 6144, 49, grid=grid(6144, 49), stream=stream0)
        buf138 = empty_strided_cuda((48, 49, 49), (2401, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf136, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf137, (48, 128, 49), (6272, 49, 1), 0), out=buf138)
        buf141 = empty_strided_cuda((8, 6, 49, 49), (14406, 2401, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_37], Original ATen: [aten._softmax]
        triton_per_fused__softmax_33.run(buf138, buf141, 2352, 49, grid=grid(2352), stream=stream0)
        buf142 = reinterpret_tensor(buf137, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [x_280], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf135, buf142, 301056, grid=grid(301056), stream=stream0)
        del buf135
        buf143 = reinterpret_tensor(buf136, (48, 49, 128), (6272, 128, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_280], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf141, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf142, (48, 49, 128), (6272, 128, 1), 0), out=buf143)
        buf145 = reinterpret_tensor(buf142, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [x_281, x_282], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_35.run(buf143, buf145, 6144, 49, grid=grid(6144, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [x_282], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg129_1
        buf147 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [x_284, batch_norm_48], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_36.run(buf133, buf146, arg130_1, arg131_1, arg132_1, arg133_1, buf147, 392, 768, grid=grid(392, 768), stream=stream0)
        del arg130_1
        del arg131_1
        del arg132_1
        del arg133_1
        # Topologically Sorted Source Nodes: [x_284, batch_norm_48, x_285], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf148 = extern_kernels.convolution(buf147, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
        del arg134_1
        buf149 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_286], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf149, 1204224, grid=grid(1204224), stream=stream0)
        # Topologically Sorted Source Nodes: [x_286, x_288], Original ATen: [aten.gelu, aten.convolution]
        buf150 = extern_kernels.convolution(buf149, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg135_1
        del buf149
        buf151 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_284, x_290, batch_norm_49], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf133, buf146, buf150, arg136_1, arg137_1, arg138_1, arg139_1, buf151, 392, 768, grid=grid(392, 768), stream=stream0)
        del arg136_1
        del arg137_1
        del arg138_1
        del arg139_1
        # Topologically Sorted Source Nodes: [x_284, x_290, batch_norm_49, conv2d_102], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 2304, 7, 7), (112896, 1, 16128, 2304))
        del arg140_1
        buf153 = reinterpret_tensor(buf151, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf152, buf153, 301056, grid=grid(301056), stream=stream0)
        buf154 = reinterpret_tensor(buf143, (8, 6, 128, 49), (37632, 6272, 49, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf152, buf154, 6144, 49, grid=grid(6144, 49), stream=stream0)
        buf155 = reinterpret_tensor(buf141, (48, 49, 49), (2401, 49, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf153, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf154, (48, 128, 49), (6272, 49, 1), 0), out=buf155)
        buf158 = reinterpret_tensor(buf138, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [attn_40], Original ATen: [aten._softmax]
        triton_per_fused__softmax_33.run(buf155, buf158, 2352, 49, grid=grid(2352), stream=stream0)
        buf159 = reinterpret_tensor(buf154, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [x_292], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf152, buf159, 301056, grid=grid(301056), stream=stream0)
        del buf152
        buf160 = reinterpret_tensor(buf153, (48, 49, 128), (6272, 128, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [x_292], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf158, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf159, (48, 49, 128), (6272, 128, 1), 0), out=buf160)
        buf162 = reinterpret_tensor(buf159, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_293, x_294], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_35.run(buf160, buf162, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del buf160
        # Topologically Sorted Source Nodes: [x_294], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg141_1
        buf164 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_284, x_290, x_296, batch_norm_50], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_39.run(buf133, buf146, buf150, buf163, arg142_1, arg143_1, arg144_1, arg145_1, buf164, 392, 768, grid=grid(392, 768), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        # Topologically Sorted Source Nodes: [x_284, x_290, x_296, batch_norm_50, x_297], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf165 = extern_kernels.convolution(buf164, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
        del arg146_1
        buf166 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_298], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf166, 1204224, grid=grid(1204224), stream=stream0)
        # Topologically Sorted Source Nodes: [x_298, x_300], Original ATen: [aten.gelu, aten.convolution]
        buf167 = extern_kernels.convolution(buf166, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg147_1
        del buf166
        buf168 = buf146; del buf146  # reuse
        buf169 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [x_284, x_290, x_296, x_302, batch_norm_51], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf168, buf133, buf150, buf163, buf167, arg148_1, arg149_1, arg150_1, arg151_1, buf169, 392, 768, grid=grid(392, 768), stream=stream0)
        del arg148_1
        del arg149_1
        del arg150_1
        del arg151_1
        del buf133
        del buf150
        del buf163
        # Topologically Sorted Source Nodes: [batch_norm_51, conv2d_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf170 = extern_kernels.convolution(buf169, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (8, 2304, 7, 7), (112896, 1, 16128, 2304))
        del arg152_1
        buf171 = reinterpret_tensor(buf169, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf170, buf171, 301056, grid=grid(301056), stream=stream0)
        buf172 = reinterpret_tensor(buf167, (8, 6, 128, 49), (37632, 6272, 49, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf170, buf172, 6144, 49, grid=grid(6144, 49), stream=stream0)
        buf173 = reinterpret_tensor(buf158, (48, 49, 49), (2401, 49, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf171, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf172, (48, 128, 49), (6272, 49, 1), 0), out=buf173)
        buf176 = reinterpret_tensor(buf155, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [attn_43], Original ATen: [aten._softmax]
        triton_per_fused__softmax_33.run(buf173, buf176, 2352, 49, grid=grid(2352), stream=stream0)
        buf177 = reinterpret_tensor(buf172, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [x_304], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf170, buf177, 301056, grid=grid(301056), stream=stream0)
        del buf170
        buf178 = reinterpret_tensor(buf171, (48, 49, 128), (6272, 128, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_304], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf176, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf177, (48, 49, 128), (6272, 128, 1), 0), out=buf178)
        buf180 = reinterpret_tensor(buf177, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_305, x_306], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_35.run(buf178, buf180, 6144, 49, grid=grid(6144, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [x_306], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg153_1
        buf182 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [x_308, batch_norm_52], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_41.run(buf168, buf181, arg154_1, arg155_1, arg156_1, arg157_1, buf182, 301056, grid=grid(301056), stream=stream0)
        del arg154_1
        del arg155_1
        del arg156_1
        del arg157_1
        # Topologically Sorted Source Nodes: [x_308, batch_norm_52, x_309], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf183 = extern_kernels.convolution(buf182, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
        del arg158_1
        buf184 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [x_310], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf184, 1204224, grid=grid(1204224), stream=stream0)
        # Topologically Sorted Source Nodes: [x_310, x_312], Original ATen: [aten.gelu, aten.convolution]
        buf185 = extern_kernels.convolution(buf184, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg159_1
        del buf184
        buf186 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [x_308, x_314, batch_norm_53], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_42.run(buf168, buf181, buf185, arg160_1, arg161_1, arg162_1, arg163_1, buf186, 301056, grid=grid(301056), stream=stream0)
        del arg160_1
        del arg161_1
        del arg162_1
        del arg163_1
        # Topologically Sorted Source Nodes: [x_308, x_314, batch_norm_53, conv2d_110], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf187 = extern_kernels.convolution(buf186, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (8, 2304, 7, 7), (112896, 1, 16128, 2304))
        del arg164_1
        buf188 = reinterpret_tensor(buf186, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf187, buf188, 301056, grid=grid(301056), stream=stream0)
        buf189 = reinterpret_tensor(buf178, (8, 6, 128, 49), (37632, 6272, 49, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf187, buf189, 6144, 49, grid=grid(6144, 49), stream=stream0)
        buf190 = reinterpret_tensor(buf176, (48, 49, 49), (2401, 49, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf188, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf189, (48, 128, 49), (6272, 49, 1), 0), out=buf190)
        buf193 = reinterpret_tensor(buf173, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [attn_46], Original ATen: [aten._softmax]
        triton_per_fused__softmax_33.run(buf190, buf193, 2352, 49, grid=grid(2352), stream=stream0)
        del buf190
        buf194 = reinterpret_tensor(buf189, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_316], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf187, buf194, 301056, grid=grid(301056), stream=stream0)
        del buf187
        buf195 = reinterpret_tensor(buf188, (48, 49, 128), (6272, 128, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_316], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf193, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf194, (48, 49, 128), (6272, 128, 1), 0), out=buf195)
        del buf193
        buf197 = reinterpret_tensor(buf194, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [x_317, x_318], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_35.run(buf195, buf197, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del buf195
        # Topologically Sorted Source Nodes: [x_318], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg165_1
        buf199 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [x_308, x_314, x_320, batch_norm_54], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_43.run(buf168, buf181, buf185, buf198, arg166_1, arg167_1, arg168_1, arg169_1, buf199, 301056, grid=grid(301056), stream=stream0)
        del arg166_1
        del arg167_1
        del arg168_1
        del arg169_1
        # Topologically Sorted Source Nodes: [x_308, x_314, x_320, batch_norm_54, x_321], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        buf200 = extern_kernels.convolution(buf199, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
        del arg170_1
        del buf199
        buf201 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [x_322], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf201, 1204224, grid=grid(1204224), stream=stream0)
        # Topologically Sorted Source Nodes: [x_322, x_324], Original ATen: [aten.gelu, aten.convolution]
        buf202 = extern_kernels.convolution(buf201, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg171_1
        del buf201
        buf203 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_308, x_314, x_320, x_326, x_327], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_add_44.run(buf203, buf181, buf185, buf198, buf202, arg172_1, arg173_1, arg174_1, arg175_1, 301056, grid=grid(301056), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        del buf181
        del buf185
        del buf198
        del buf202
        buf205 = empty_strided_cuda((8, 768, 1, 1), (768, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [x_328], Original ATen: [aten.mean]
        triton_per_fused_mean_45.run(buf203, buf205, 6144, 49, grid=grid(6144), stream=stream0)
        del buf203
        buf206 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_331], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg177_1, reinterpret_tensor(buf205, (8, 768), (768, 1), 0), reinterpret_tensor(arg176_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf206)
        del arg176_1
        del arg177_1
        del buf205
    return (buf206, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((192, 32, 4, 4), (512, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, 192, 2, 2), (768, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 384, 2, 2), (1536, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('visformer_small', benchmark_compiled_module)
