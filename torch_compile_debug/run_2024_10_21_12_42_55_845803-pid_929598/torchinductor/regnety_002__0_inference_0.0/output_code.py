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
# Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_158 => convolution_70
# Graph fragment:
#   %convolution_70 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/fn/cfnqc5t7ruox32h42p5hkjg336a6auwfmlumtz3g3aru5iaouonc.py
# Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_158 => convolution_70
# Graph fragment:
#   %convolution_70 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[128, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ek/cekujtroe4lmfynngo4l6377j52uwrdlvgggr4iljkvhusbdr7ty.py
# Topologically Sorted Source Nodes: [x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_159 => add_102, mul_146, mul_147, sub_44
#   x_160 => relu_53
# Graph fragment:
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_70, %unsqueeze_353), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %unsqueeze_355), kwargs = {})
#   %mul_147 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_146, %unsqueeze_357), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_147, %unsqueeze_359), kwargs = {})
#   %relu_53 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_102,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/4v/c4vo32qqdurqczrwr5uhfis3gseycrmjgplqlqalp26t6bob6xv3.py
# Topologically Sorted Source Nodes: [x_162, x_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_162 => add_104, mul_149, mul_150, sub_45
#   x_163 => relu_54
# Graph fragment:
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_361), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_149, %unsqueeze_365), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_150, %unsqueeze_367), kwargs = {})
#   %relu_54 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_104,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 24
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


# kernel path: /tmp/torchinductor_sahanp/ql/cql5xh3fabrbejscpct4v7j7yctrjmsbfc53is6eelfkrhxwuoc2.py
# Topologically Sorted Source Nodes: [x_162, x_163, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_162 => add_104, mul_149, mul_150, sub_45
#   x_163 => relu_54
#   x_164 => convolution_72
# Graph fragment:
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_361), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_149, %unsqueeze_365), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_150, %unsqueeze_367), kwargs = {})
#   %relu_54 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_104,), kwargs = {})
#   %convolution_72 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_54, %arg11_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 3), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
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


# kernel path: /tmp/torchinductor_sahanp/s2/cs2b4ipcec5hl6h6tkullnau2r35w5sjzzoeesirqeso47szdgof.py
# Topologically Sorted Source Nodes: [x_165, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_165 => add_106, mul_152, mul_153, sub_46
#   x_166 => relu_55
# Graph fragment:
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_72, %unsqueeze_369), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_371), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_152, %unsqueeze_373), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_153, %unsqueeze_375), kwargs = {})
#   %relu_55 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_106,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 24
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


# kernel path: /tmp/torchinductor_sahanp/cm/ccmefrcjmwrch6l7mq5vrxwrzx5b327h7x6vm7wmx5g2ca3b4zjw.py
# Topologically Sorted Source Nodes: [x_se_52], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_52 => mean_14
# Graph fragment:
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_55, [2, 3], True), kwargs = {})
triton_red_fused_mean_6 = async_compile.triton('triton_red_fused_mean_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_6(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 24) % 25
    x0 = xindex % 24
    x2 = (xindex // 600)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (24*((r3 + (126*x1)) % 3136)) + (75264*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bx/cbxklmskgrmu5h4mgxmqqimmwpffi56g3qdecjk5tngmavwsoehk.py
# Topologically Sorted Source Nodes: [x_se_52], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_52 => mean_14
# Graph fragment:
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_55, [2, 3], True), kwargs = {})
triton_per_fused_mean_7 = async_compile.triton('triton_per_fused_mean_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[256, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_7(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 24
    x1 = (xindex // 24)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (24*r2) + (600*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 3136.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/z6/cz67mv7je6xgq45mgvelf4hjtfazmucujyusvbf7wnhtqgqnsb4h.py
# Topologically Sorted Source Nodes: [x_se_52, x_se_53, x_se_54], Original ATen: [aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_se_52 => mean_14
#   x_se_53 => convolution_73
#   x_se_54 => relu_56
# Graph fragment:
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_55, [2, 3], True), kwargs = {})
#   %convolution_73 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_14, %arg16_1, %arg17_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_56 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_73,), kwargs = {})
triton_poi_fused_convolution_mean_relu_8 = async_compile.triton('triton_poi_fused_convolution_mean_relu_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pq/cpqr4j7aknma5n7xcfaybodbymrhu2im6haaotducxvo5yqg7i3i.py
# Topologically Sorted Source Nodes: [x_se_52, x_se_53, x_se_54, x_se_55, sigmoid_13, x_167], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_13 => sigmoid_13
#   x_167 => mul_154
#   x_se_52 => mean_14
#   x_se_53 => convolution_73
#   x_se_54 => relu_56
#   x_se_55 => convolution_74
# Graph fragment:
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_55, [2, 3], True), kwargs = {})
#   %convolution_73 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_14, %arg16_1, %arg17_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_56 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_73,), kwargs = {})
#   %convolution_74 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_56, %arg18_1, %arg19_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_74,), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_55, %sigmoid_13), kwargs = {})
triton_poi_fused_convolution_mean_mul_relu_sigmoid_9 = async_compile.triton('triton_poi_fused_convolution_mean_mul_relu_sigmoid_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_relu_sigmoid_9(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 24
    x2 = (xindex // 75264)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (24*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4j/c4jvyeea7xytumjqsuijc4yunr2jkct6gkzzbnlu2issunzuncyc.py
# Topologically Sorted Source Nodes: [x_169, x_171, x_172, x_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_169 => add_108, mul_156, mul_157, sub_47
#   x_171 => add_110, mul_159, mul_160, sub_48
#   x_172 => add_111
#   x_173 => relu_57
# Graph fragment:
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_75, %unsqueeze_377), kwargs = {})
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %unsqueeze_379), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_156, %unsqueeze_381), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_157, %unsqueeze_383), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_76, %unsqueeze_385), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_387), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_159, %unsqueeze_389), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_160, %unsqueeze_391), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_108, %add_110), kwargs = {})
#   %relu_57 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_111,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3p/c3pew2hbvmmntxieajrwxivwktz3hxiaxb7pdtivb5cwpy66qwzc.py
# Topologically Sorted Source Nodes: [x_175, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_175 => add_113, mul_162, mul_163, sub_49
#   x_176 => relu_58
# Graph fragment:
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_393), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_162, %unsqueeze_397), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_163, %unsqueeze_399), kwargs = {})
#   %relu_58 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_113,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 56
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


# kernel path: /tmp/torchinductor_sahanp/nw/cnwy6bdbmcfel2wftxookhvb2ft6px5a7wrpaasb7lckfm3b6r7x.py
# Topologically Sorted Source Nodes: [x_175, x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_175 => add_113, mul_162, mul_163, sub_49
#   x_176 => relu_58
#   x_177 => convolution_78
# Graph fragment:
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_393), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_162, %unsqueeze_397), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_163, %unsqueeze_399), kwargs = {})
#   %relu_58 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_113,), kwargs = {})
#   %convolution_78 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_58, %arg35_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 7), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
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


# kernel path: /tmp/torchinductor_sahanp/3d/c3ddmuc3vgtegwf26cljxl5ufhcy6ilu5etc7jgtqi7rpfdysrma.py
# Topologically Sorted Source Nodes: [x_178, x_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_178 => add_115, mul_165, mul_166, sub_50
#   x_179 => relu_59
# Graph fragment:
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_78, %unsqueeze_401), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_403), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_165, %unsqueeze_405), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_166, %unsqueeze_407), kwargs = {})
#   %relu_59 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_115,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 56
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/l2/cl2rclau4ljwjuxst3dvwcnakd4gpvjjzs23cux4xqmqoakndqs6.py
# Topologically Sorted Source Nodes: [x_se_56], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_56 => mean_15
# Graph fragment:
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_59, [2, 3], True), kwargs = {})
triton_red_fused_mean_14 = async_compile.triton('triton_red_fused_mean_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_14(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 56
    x1 = (xindex // 56)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (56*r2) + (6272*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ag/cagkxdgw6ij765iu24quhn6g3tbxk3ggelwbxbdjvzmyixlf5bmj.py
# Topologically Sorted Source Nodes: [x_se_56], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_56 => mean_15
# Graph fragment:
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_59, [2, 3], True), kwargs = {})
triton_per_fused_mean_15 = async_compile.triton('triton_per_fused_mean_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_15(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 56
    x1 = (xindex // 56)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (56*r2) + (392*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mv/cmv5qd5wryeaf4ya2a7xxhhjpcmkipfjs36eeqxlp2qxnb7qhw6l.py
# Topologically Sorted Source Nodes: [x_se_56, x_se_57, x_se_58], Original ATen: [aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_se_56 => mean_15
#   x_se_57 => convolution_79
#   x_se_58 => relu_60
# Graph fragment:
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_59, [2, 3], True), kwargs = {})
#   %convolution_79 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_15, %arg40_1, %arg41_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_60 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_79,), kwargs = {})
triton_poi_fused_convolution_mean_relu_16 = async_compile.triton('triton_poi_fused_convolution_mean_relu_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_16(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wb/cwbly5uhv337v4lagecwsryylbbszm2tpnqanjbsraddkttifhf2.py
# Topologically Sorted Source Nodes: [x_se_56, x_se_57, x_se_58, x_se_59, sigmoid_14, x_180], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_14 => sigmoid_14
#   x_180 => mul_167
#   x_se_56 => mean_15
#   x_se_57 => convolution_79
#   x_se_58 => relu_60
#   x_se_59 => convolution_80
# Graph fragment:
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_59, [2, 3], True), kwargs = {})
#   %convolution_79 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_15, %arg40_1, %arg41_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_60 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_79,), kwargs = {})
#   %convolution_80 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_60, %arg42_1, %arg43_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_14 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_80,), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_59, %sigmoid_14), kwargs = {})
triton_poi_fused_convolution_mean_mul_relu_sigmoid_17 = async_compile.triton('triton_poi_fused_convolution_mean_mul_relu_sigmoid_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_relu_sigmoid_17(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 56
    x2 = (xindex // 43904)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (56*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5o/c5oiiyuhe7r55fin6hch4pu7b76o3y4pkgdq3rd6km4in4ihwkss.py
# Topologically Sorted Source Nodes: [x_182, x_184, x_185, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_182 => add_117, mul_169, mul_170, sub_51
#   x_184 => add_119, mul_172, mul_173, sub_52
#   x_185 => add_120
#   x_186 => relu_61
# Graph fragment:
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_81, %unsqueeze_409), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_411), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_169, %unsqueeze_413), kwargs = {})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_170, %unsqueeze_415), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_82, %unsqueeze_417), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_419), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %unsqueeze_421), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %unsqueeze_423), kwargs = {})
#   %add_120 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_117, %add_119), kwargs = {})
#   %relu_61 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_120,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 56
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kg/ckgshdt27kfgyniyulbmg62itrajxn22incjbbenomh5ysf645x6.py
# Topologically Sorted Source Nodes: [x_188, x_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_188 => add_122, mul_175, mul_176, sub_53
#   x_189 => relu_62
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_83, %unsqueeze_425), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %unsqueeze_429), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %unsqueeze_431), kwargs = {})
#   %relu_62 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_122,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 152
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jr/cjrkqw5p3g2y3eaw35q6f3q3dybt3wzwdbfobien66oistglpnlw.py
# Topologically Sorted Source Nodes: [x_188, x_189, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_188 => add_122, mul_175, mul_176, sub_53
#   x_189 => relu_62
#   x_190 => convolution_84
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_83, %unsqueeze_425), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %unsqueeze_429), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %unsqueeze_431), kwargs = {})
#   %relu_62 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_122,), kwargs = {})
#   %convolution_84 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_62, %arg59_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 19), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1216
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


# kernel path: /tmp/torchinductor_sahanp/f7/cf7hlvqjag4whtpfvcecmwaaveuv7dr5fehoeuqyfsc27gwksql5.py
# Topologically Sorted Source Nodes: [x_191, x_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_191 => add_124, mul_178, mul_179, sub_54
#   x_192 => relu_63
# Graph fragment:
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_84, %unsqueeze_433), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_435), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_178, %unsqueeze_437), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_179, %unsqueeze_439), kwargs = {})
#   %relu_63 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_124,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 152
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7x/c7x54zhg2pe2n7vhh6bcyq3p2xcmjjuswiw7al64kjundoiq5ato.py
# Topologically Sorted Source Nodes: [x_se_60], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_60 => mean_16
# Graph fragment:
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_63, [2, 3], True), kwargs = {})
triton_red_fused_mean_22 = async_compile.triton('triton_red_fused_mean_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_22(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2432
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 152
    x1 = (xindex // 152)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (152*r2) + (14896*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ws/cwsytm2ux2euxcwjlapgle6hlmetn73npklj6sxh4ifdfwhx63ag.py
# Topologically Sorted Source Nodes: [x_se_60], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_60 => mean_16
# Graph fragment:
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_63, [2, 3], True), kwargs = {})
triton_per_fused_mean_23 = async_compile.triton('triton_per_fused_mean_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_23(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1216
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 152
    x1 = (xindex // 152)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (152*r2) + (304*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2u/c2u7iqfcdgm26xbcnjgidgjhmjmbaqxrmgtucaacj7smwban7uio.py
# Topologically Sorted Source Nodes: [x_se_60, x_se_61, x_se_62], Original ATen: [aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_se_60 => mean_16
#   x_se_61 => convolution_85
#   x_se_62 => relu_64
# Graph fragment:
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_63, [2, 3], True), kwargs = {})
#   %convolution_85 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_16, %arg64_1, %arg65_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_64 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_85,), kwargs = {})
triton_poi_fused_convolution_mean_relu_24 = async_compile.triton('triton_poi_fused_convolution_mean_relu_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_24(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 14
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gt/cgtmwbv3t2lmx2hopwf4qjqldnjtd65hqbdfwryzt5tazfdw3avi.py
# Topologically Sorted Source Nodes: [x_se_60, x_se_61, x_se_62, x_se_63, sigmoid_15, x_193], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_15 => sigmoid_15
#   x_193 => mul_180
#   x_se_60 => mean_16
#   x_se_61 => convolution_85
#   x_se_62 => relu_64
#   x_se_63 => convolution_86
# Graph fragment:
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_63, [2, 3], True), kwargs = {})
#   %convolution_85 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_16, %arg64_1, %arg65_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_64 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_85,), kwargs = {})
#   %convolution_86 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_64, %arg66_1, %arg67_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_86,), kwargs = {})
#   %mul_180 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_63, %sigmoid_15), kwargs = {})
triton_poi_fused_convolution_mean_mul_relu_sigmoid_25 = async_compile.triton('triton_poi_fused_convolution_mean_mul_relu_sigmoid_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_relu_sigmoid_25(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 152
    x2 = (xindex // 29792)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (152*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ph/cphjkkchhadbeuqarhjfx77lwefn653x5cxz257xc3shtrh5l4n7.py
# Topologically Sorted Source Nodes: [x_195, x_197, x_198, x_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_195 => add_126, mul_182, mul_183, sub_55
#   x_197 => add_128, mul_185, mul_186, sub_56
#   x_198 => add_129
#   x_199 => relu_65
# Graph fragment:
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_441), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_443), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_182, %unsqueeze_445), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_183, %unsqueeze_447), kwargs = {})
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_88, %unsqueeze_449), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %unsqueeze_451), kwargs = {})
#   %mul_186 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_185, %unsqueeze_453), kwargs = {})
#   %add_128 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_186, %unsqueeze_455), kwargs = {})
#   %add_129 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_126, %add_128), kwargs = {})
#   %relu_65 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_129,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 152
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kc/ckcdgt2rigdw7grnu7l64b6ibdasrtk2bj4pfe6v5odlwrlmxefa.py
# Topologically Sorted Source Nodes: [x_se_64, x_se_65, x_se_66], Original ATen: [aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_se_64 => mean_17
#   x_se_65 => convolution_91
#   x_se_66 => relu_68
# Graph fragment:
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_67, [2, 3], True), kwargs = {})
#   %convolution_91 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_17, %arg88_1, %arg89_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_68 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_91,), kwargs = {})
triton_poi_fused_convolution_mean_relu_27 = async_compile.triton('triton_poi_fused_convolution_mean_relu_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_27(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 38
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uh/cuhzrkzzxy7xzfag6dfrcfu7cs64wbzsth4casnidwbtdyz7tgmy.py
# Topologically Sorted Source Nodes: [x_208, x_209, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_208 => add_135, mul_195, mul_196, sub_59
#   x_209 => add_136
#   x_210 => relu_69
# Graph fragment:
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_93, %unsqueeze_473), kwargs = {})
#   %mul_195 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_475), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_195, %unsqueeze_477), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_196, %unsqueeze_479), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_135, %relu_65), kwargs = {})
#   %relu_69 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_136,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 152
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dn/cdn3mnsucehb5bu6fxcupmjlojkfrtg5vctxdhbkofk4lwn3rdn4.py
# Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_234 => add_152, mul_218, mul_219, sub_66
#   x_235 => relu_78
# Graph fragment:
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_104, %unsqueeze_529), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_219 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_218, %unsqueeze_533), kwargs = {})
#   %add_152 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_219, %unsqueeze_535), kwargs = {})
#   %relu_78 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_152,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 577024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 368
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mk/cmkax55wgrggsmbbod5ltalvfn56t4vhpc7h5nxaijkbsvgm7csy.py
# Topologically Sorted Source Nodes: [x_234, x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_234 => add_152, mul_218, mul_219, sub_66
#   x_235 => relu_78
#   x_236 => convolution_105
# Graph fragment:
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_104, %unsqueeze_529), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_219 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_218, %unsqueeze_533), kwargs = {})
#   %add_152 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_219, %unsqueeze_535), kwargs = {})
#   %relu_78 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_152,), kwargs = {})
#   %convolution_105 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_78, %arg140_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 46), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2944
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


# kernel path: /tmp/torchinductor_sahanp/ti/ctiup2oy4byorr4zuckdlhxik2qym5lwk6t6v4dn7lv442njy77o.py
# Topologically Sorted Source Nodes: [x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_237 => add_154, mul_221, mul_222, sub_67
#   x_238 => relu_79
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_537), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_221, %unsqueeze_541), kwargs = {})
#   %add_154 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_222, %unsqueeze_543), kwargs = {})
#   %relu_79 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_154,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 368
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tp/ctpezyj7zvxb3eu3ezvbukjcmncc33k4i5zgdd47onhd7rxjbznh.py
# Topologically Sorted Source Nodes: [x_se_76], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_76 => mean_20
# Graph fragment:
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_79, [2, 3], True), kwargs = {})
triton_per_fused_mean_32 = async_compile.triton('triton_per_fused_mean_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_32(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2944
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 368
    x1 = (xindex // 368)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (368*r2) + (18032*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vf/cvfok23ii7mr7kkyau5ho6mw6p55mkxhkre2tkfmbg7wtgko3exi.py
# Topologically Sorted Source Nodes: [x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, x_239], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_19 => sigmoid_19
#   x_239 => mul_223
#   x_se_76 => mean_20
#   x_se_77 => convolution_106
#   x_se_78 => relu_80
#   x_se_79 => convolution_107
# Graph fragment:
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_79, [2, 3], True), kwargs = {})
#   %convolution_106 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_20, %arg145_1, %arg146_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_80 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_106,), kwargs = {})
#   %convolution_107 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_80, %arg147_1, %arg148_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_19 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_107,), kwargs = {})
#   %mul_223 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_79, %sigmoid_19), kwargs = {})
triton_poi_fused_convolution_mean_mul_relu_sigmoid_33 = async_compile.triton('triton_poi_fused_convolution_mean_mul_relu_sigmoid_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_relu_sigmoid_33(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 368
    x2 = (xindex // 18032)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (368*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ie/cie6g3q7bpz5uqr4tk67xt4ujsed2inkxpnoz73d4awkgjj2psms.py
# Topologically Sorted Source Nodes: [x_241, x_243, x_244, x_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_241 => add_156, mul_225, mul_226, sub_68
#   x_243 => add_158, mul_228, mul_229, sub_69
#   x_244 => add_159
#   x_245 => relu_81
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_108, %unsqueeze_545), kwargs = {})
#   %mul_225 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_225, %unsqueeze_549), kwargs = {})
#   %add_156 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_226, %unsqueeze_551), kwargs = {})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_109, %unsqueeze_553), kwargs = {})
#   %mul_228 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_228, %unsqueeze_557), kwargs = {})
#   %add_158 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_229, %unsqueeze_559), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_156, %add_158), kwargs = {})
#   %relu_81 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_159,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 368
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xv/cxvja5b2sjj3kh2duupdfd2kaf45y5pqym3lgdosj2eck6vf3rpd.py
# Topologically Sorted Source Nodes: [x_se_80, x_se_81, x_se_82], Original ATen: [aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_se_80 => mean_21
#   x_se_81 => convolution_112
#   x_se_82 => relu_84
# Graph fragment:
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_83, [2, 3], True), kwargs = {})
#   %convolution_112 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_21, %arg169_1, %arg170_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_84 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_112,), kwargs = {})
triton_poi_fused_convolution_mean_relu_35 = async_compile.triton('triton_poi_fused_convolution_mean_relu_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_35(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 92
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ke/ckerzl6o2zs7ykxfspbrpeobgp2cfbuncocjr2iv4wf7w6xdll5y.py
# Topologically Sorted Source Nodes: [x_254, x_255, x_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_254 => add_165, mul_238, mul_239, sub_72
#   x_255 => add_166
#   x_256 => relu_85
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_114, %unsqueeze_577), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_238, %unsqueeze_581), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_239, %unsqueeze_583), kwargs = {})
#   %add_166 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_165, %relu_81), kwargs = {})
#   %relu_85 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_166,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 368
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ii/ciikxvf6jp2tushhkf6rnbmuwab7fyfnsobpwh37apj5ru7c6oaf.py
# Topologically Sorted Source Nodes: [x_309, x_310, x_311, x_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   x_309 => add_200, mul_288, mul_289, sub_87
#   x_310 => add_201
#   x_311 => relu_105
#   x_312 => mean_27
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_139, %unsqueeze_697), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_288, %unsqueeze_701), kwargs = {})
#   %add_200 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_289, %unsqueeze_703), kwargs = {})
#   %add_201 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_200, %relu_101), kwargs = {})
#   %relu_105 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_201,), kwargs = {})
#   %mean_27 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_105, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_37 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2944
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 368
    x1 = (xindex // 368)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (368*r2) + (18032*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0 + (368*r2) + (18032*x1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1, 1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = 49.0
    tmp25 = tmp23 / tmp24
    tl.store(out_ptr1 + (x3), tmp25, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg7_1, (24, ), (1, ))
    assert_size_stride(arg8_1, (24, ), (1, ))
    assert_size_stride(arg9_1, (24, ), (1, ))
    assert_size_stride(arg10_1, (24, ), (1, ))
    assert_size_stride(arg11_1, (24, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg12_1, (24, ), (1, ))
    assert_size_stride(arg13_1, (24, ), (1, ))
    assert_size_stride(arg14_1, (24, ), (1, ))
    assert_size_stride(arg15_1, (24, ), (1, ))
    assert_size_stride(arg16_1, (8, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg17_1, (8, ), (1, ))
    assert_size_stride(arg18_1, (24, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg19_1, (24, ), (1, ))
    assert_size_stride(arg20_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg21_1, (24, ), (1, ))
    assert_size_stride(arg22_1, (24, ), (1, ))
    assert_size_stride(arg23_1, (24, ), (1, ))
    assert_size_stride(arg24_1, (24, ), (1, ))
    assert_size_stride(arg25_1, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg26_1, (24, ), (1, ))
    assert_size_stride(arg27_1, (24, ), (1, ))
    assert_size_stride(arg28_1, (24, ), (1, ))
    assert_size_stride(arg29_1, (24, ), (1, ))
    assert_size_stride(arg30_1, (56, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg31_1, (56, ), (1, ))
    assert_size_stride(arg32_1, (56, ), (1, ))
    assert_size_stride(arg33_1, (56, ), (1, ))
    assert_size_stride(arg34_1, (56, ), (1, ))
    assert_size_stride(arg35_1, (56, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg36_1, (56, ), (1, ))
    assert_size_stride(arg37_1, (56, ), (1, ))
    assert_size_stride(arg38_1, (56, ), (1, ))
    assert_size_stride(arg39_1, (56, ), (1, ))
    assert_size_stride(arg40_1, (6, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg41_1, (6, ), (1, ))
    assert_size_stride(arg42_1, (56, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg43_1, (56, ), (1, ))
    assert_size_stride(arg44_1, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg45_1, (56, ), (1, ))
    assert_size_stride(arg46_1, (56, ), (1, ))
    assert_size_stride(arg47_1, (56, ), (1, ))
    assert_size_stride(arg48_1, (56, ), (1, ))
    assert_size_stride(arg49_1, (56, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg50_1, (56, ), (1, ))
    assert_size_stride(arg51_1, (56, ), (1, ))
    assert_size_stride(arg52_1, (56, ), (1, ))
    assert_size_stride(arg53_1, (56, ), (1, ))
    assert_size_stride(arg54_1, (152, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg55_1, (152, ), (1, ))
    assert_size_stride(arg56_1, (152, ), (1, ))
    assert_size_stride(arg57_1, (152, ), (1, ))
    assert_size_stride(arg58_1, (152, ), (1, ))
    assert_size_stride(arg59_1, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg60_1, (152, ), (1, ))
    assert_size_stride(arg61_1, (152, ), (1, ))
    assert_size_stride(arg62_1, (152, ), (1, ))
    assert_size_stride(arg63_1, (152, ), (1, ))
    assert_size_stride(arg64_1, (14, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg65_1, (14, ), (1, ))
    assert_size_stride(arg66_1, (152, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(arg67_1, (152, ), (1, ))
    assert_size_stride(arg68_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg69_1, (152, ), (1, ))
    assert_size_stride(arg70_1, (152, ), (1, ))
    assert_size_stride(arg71_1, (152, ), (1, ))
    assert_size_stride(arg72_1, (152, ), (1, ))
    assert_size_stride(arg73_1, (152, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg74_1, (152, ), (1, ))
    assert_size_stride(arg75_1, (152, ), (1, ))
    assert_size_stride(arg76_1, (152, ), (1, ))
    assert_size_stride(arg77_1, (152, ), (1, ))
    assert_size_stride(arg78_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg79_1, (152, ), (1, ))
    assert_size_stride(arg80_1, (152, ), (1, ))
    assert_size_stride(arg81_1, (152, ), (1, ))
    assert_size_stride(arg82_1, (152, ), (1, ))
    assert_size_stride(arg83_1, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg84_1, (152, ), (1, ))
    assert_size_stride(arg85_1, (152, ), (1, ))
    assert_size_stride(arg86_1, (152, ), (1, ))
    assert_size_stride(arg87_1, (152, ), (1, ))
    assert_size_stride(arg88_1, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg89_1, (38, ), (1, ))
    assert_size_stride(arg90_1, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg91_1, (152, ), (1, ))
    assert_size_stride(arg92_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg93_1, (152, ), (1, ))
    assert_size_stride(arg94_1, (152, ), (1, ))
    assert_size_stride(arg95_1, (152, ), (1, ))
    assert_size_stride(arg96_1, (152, ), (1, ))
    assert_size_stride(arg97_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg98_1, (152, ), (1, ))
    assert_size_stride(arg99_1, (152, ), (1, ))
    assert_size_stride(arg100_1, (152, ), (1, ))
    assert_size_stride(arg101_1, (152, ), (1, ))
    assert_size_stride(arg102_1, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg103_1, (152, ), (1, ))
    assert_size_stride(arg104_1, (152, ), (1, ))
    assert_size_stride(arg105_1, (152, ), (1, ))
    assert_size_stride(arg106_1, (152, ), (1, ))
    assert_size_stride(arg107_1, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg108_1, (38, ), (1, ))
    assert_size_stride(arg109_1, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg110_1, (152, ), (1, ))
    assert_size_stride(arg111_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg112_1, (152, ), (1, ))
    assert_size_stride(arg113_1, (152, ), (1, ))
    assert_size_stride(arg114_1, (152, ), (1, ))
    assert_size_stride(arg115_1, (152, ), (1, ))
    assert_size_stride(arg116_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg117_1, (152, ), (1, ))
    assert_size_stride(arg118_1, (152, ), (1, ))
    assert_size_stride(arg119_1, (152, ), (1, ))
    assert_size_stride(arg120_1, (152, ), (1, ))
    assert_size_stride(arg121_1, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg122_1, (152, ), (1, ))
    assert_size_stride(arg123_1, (152, ), (1, ))
    assert_size_stride(arg124_1, (152, ), (1, ))
    assert_size_stride(arg125_1, (152, ), (1, ))
    assert_size_stride(arg126_1, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg127_1, (38, ), (1, ))
    assert_size_stride(arg128_1, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg129_1, (152, ), (1, ))
    assert_size_stride(arg130_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg131_1, (152, ), (1, ))
    assert_size_stride(arg132_1, (152, ), (1, ))
    assert_size_stride(arg133_1, (152, ), (1, ))
    assert_size_stride(arg134_1, (152, ), (1, ))
    assert_size_stride(arg135_1, (368, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg136_1, (368, ), (1, ))
    assert_size_stride(arg137_1, (368, ), (1, ))
    assert_size_stride(arg138_1, (368, ), (1, ))
    assert_size_stride(arg139_1, (368, ), (1, ))
    assert_size_stride(arg140_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg141_1, (368, ), (1, ))
    assert_size_stride(arg142_1, (368, ), (1, ))
    assert_size_stride(arg143_1, (368, ), (1, ))
    assert_size_stride(arg144_1, (368, ), (1, ))
    assert_size_stride(arg145_1, (38, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg146_1, (38, ), (1, ))
    assert_size_stride(arg147_1, (368, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg148_1, (368, ), (1, ))
    assert_size_stride(arg149_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg150_1, (368, ), (1, ))
    assert_size_stride(arg151_1, (368, ), (1, ))
    assert_size_stride(arg152_1, (368, ), (1, ))
    assert_size_stride(arg153_1, (368, ), (1, ))
    assert_size_stride(arg154_1, (368, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg155_1, (368, ), (1, ))
    assert_size_stride(arg156_1, (368, ), (1, ))
    assert_size_stride(arg157_1, (368, ), (1, ))
    assert_size_stride(arg158_1, (368, ), (1, ))
    assert_size_stride(arg159_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg160_1, (368, ), (1, ))
    assert_size_stride(arg161_1, (368, ), (1, ))
    assert_size_stride(arg162_1, (368, ), (1, ))
    assert_size_stride(arg163_1, (368, ), (1, ))
    assert_size_stride(arg164_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg165_1, (368, ), (1, ))
    assert_size_stride(arg166_1, (368, ), (1, ))
    assert_size_stride(arg167_1, (368, ), (1, ))
    assert_size_stride(arg168_1, (368, ), (1, ))
    assert_size_stride(arg169_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg170_1, (92, ), (1, ))
    assert_size_stride(arg171_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg172_1, (368, ), (1, ))
    assert_size_stride(arg173_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg174_1, (368, ), (1, ))
    assert_size_stride(arg175_1, (368, ), (1, ))
    assert_size_stride(arg176_1, (368, ), (1, ))
    assert_size_stride(arg177_1, (368, ), (1, ))
    assert_size_stride(arg178_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg179_1, (368, ), (1, ))
    assert_size_stride(arg180_1, (368, ), (1, ))
    assert_size_stride(arg181_1, (368, ), (1, ))
    assert_size_stride(arg182_1, (368, ), (1, ))
    assert_size_stride(arg183_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg184_1, (368, ), (1, ))
    assert_size_stride(arg185_1, (368, ), (1, ))
    assert_size_stride(arg186_1, (368, ), (1, ))
    assert_size_stride(arg187_1, (368, ), (1, ))
    assert_size_stride(arg188_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg189_1, (92, ), (1, ))
    assert_size_stride(arg190_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg191_1, (368, ), (1, ))
    assert_size_stride(arg192_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg193_1, (368, ), (1, ))
    assert_size_stride(arg194_1, (368, ), (1, ))
    assert_size_stride(arg195_1, (368, ), (1, ))
    assert_size_stride(arg196_1, (368, ), (1, ))
    assert_size_stride(arg197_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg198_1, (368, ), (1, ))
    assert_size_stride(arg199_1, (368, ), (1, ))
    assert_size_stride(arg200_1, (368, ), (1, ))
    assert_size_stride(arg201_1, (368, ), (1, ))
    assert_size_stride(arg202_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg203_1, (368, ), (1, ))
    assert_size_stride(arg204_1, (368, ), (1, ))
    assert_size_stride(arg205_1, (368, ), (1, ))
    assert_size_stride(arg206_1, (368, ), (1, ))
    assert_size_stride(arg207_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg208_1, (92, ), (1, ))
    assert_size_stride(arg209_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg210_1, (368, ), (1, ))
    assert_size_stride(arg211_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg212_1, (368, ), (1, ))
    assert_size_stride(arg213_1, (368, ), (1, ))
    assert_size_stride(arg214_1, (368, ), (1, ))
    assert_size_stride(arg215_1, (368, ), (1, ))
    assert_size_stride(arg216_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg217_1, (368, ), (1, ))
    assert_size_stride(arg218_1, (368, ), (1, ))
    assert_size_stride(arg219_1, (368, ), (1, ))
    assert_size_stride(arg220_1, (368, ), (1, ))
    assert_size_stride(arg221_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg222_1, (368, ), (1, ))
    assert_size_stride(arg223_1, (368, ), (1, ))
    assert_size_stride(arg224_1, (368, ), (1, ))
    assert_size_stride(arg225_1, (368, ), (1, ))
    assert_size_stride(arg226_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg227_1, (92, ), (1, ))
    assert_size_stride(arg228_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg229_1, (368, ), (1, ))
    assert_size_stride(arg230_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg231_1, (368, ), (1, ))
    assert_size_stride(arg232_1, (368, ), (1, ))
    assert_size_stride(arg233_1, (368, ), (1, ))
    assert_size_stride(arg234_1, (368, ), (1, ))
    assert_size_stride(arg235_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg236_1, (368, ), (1, ))
    assert_size_stride(arg237_1, (368, ), (1, ))
    assert_size_stride(arg238_1, (368, ), (1, ))
    assert_size_stride(arg239_1, (368, ), (1, ))
    assert_size_stride(arg240_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg241_1, (368, ), (1, ))
    assert_size_stride(arg242_1, (368, ), (1, ))
    assert_size_stride(arg243_1, (368, ), (1, ))
    assert_size_stride(arg244_1, (368, ), (1, ))
    assert_size_stride(arg245_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg246_1, (92, ), (1, ))
    assert_size_stride(arg247_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg248_1, (368, ), (1, ))
    assert_size_stride(arg249_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg250_1, (368, ), (1, ))
    assert_size_stride(arg251_1, (368, ), (1, ))
    assert_size_stride(arg252_1, (368, ), (1, ))
    assert_size_stride(arg253_1, (368, ), (1, ))
    assert_size_stride(arg254_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg255_1, (368, ), (1, ))
    assert_size_stride(arg256_1, (368, ), (1, ))
    assert_size_stride(arg257_1, (368, ), (1, ))
    assert_size_stride(arg258_1, (368, ), (1, ))
    assert_size_stride(arg259_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg260_1, (368, ), (1, ))
    assert_size_stride(arg261_1, (368, ), (1, ))
    assert_size_stride(arg262_1, (368, ), (1, ))
    assert_size_stride(arg263_1, (368, ), (1, ))
    assert_size_stride(arg264_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg265_1, (92, ), (1, ))
    assert_size_stride(arg266_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg267_1, (368, ), (1, ))
    assert_size_stride(arg268_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg269_1, (368, ), (1, ))
    assert_size_stride(arg270_1, (368, ), (1, ))
    assert_size_stride(arg271_1, (368, ), (1, ))
    assert_size_stride(arg272_1, (368, ), (1, ))
    assert_size_stride(arg273_1, (1000, 368), (368, 1))
    assert_size_stride(arg274_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        # Topologically Sorted Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 24, 112, 112), (301056, 1, 2688, 24))
        del arg6_1
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_162, x_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf5, arg7_1, arg8_1, arg9_1, arg10_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf6 = empty_strided_cuda((24, 8, 3, 3), (72, 1, 24, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x_162, x_163, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(arg11_1, buf6, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [x_162, x_163, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf7 = extern_kernels.convolution(buf5, buf6, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf7, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del buf5
        del buf6
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_165, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf8, arg12_1, arg13_1, arg14_1, arg15_1, 602112, grid=grid(602112), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        buf9 = empty_strided_cuda((8, 24, 1, 1, 25), (600, 1, 4800, 4800, 24), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_52], Original ATen: [aten.mean]
        triton_red_fused_mean_6.run(buf8, buf9, 4800, 126, grid=grid(4800), stream=stream0)
        buf11 = empty_strided_cuda((8, 24, 1, 1), (24, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_52], Original ATen: [aten.mean]
        triton_per_fused_mean_7.run(buf9, buf11, 192, 25, grid=grid(192), stream=stream0)
        del buf9
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53], Original ATen: [aten.mean, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg16_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53, x_se_54], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_8.run(buf13, arg17_1, 64, grid=grid(64), stream=stream0)
        del arg17_1
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf14 = extern_kernels.convolution(buf13, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 24, 1, 1), (24, 1, 1, 1))
        del arg18_1
        del buf13
        buf15 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53, x_se_54, x_se_55, sigmoid_13, x_167], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_9.run(buf15, buf14, arg19_1, 602112, grid=grid(602112), stream=stream0)
        del arg19_1
        del buf14
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53, x_se_54, x_se_55, sigmoid_13, x_167, x_168], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf16 = extern_kernels.convolution(buf15, arg20_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg20_1
        # Topologically Sorted Source Nodes: [x_170], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf3, arg25_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg25_1
        del buf3
        buf18 = buf16; del buf16  # reuse
        buf19 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_169, x_171, x_172, x_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf18, arg21_1, arg22_1, arg23_1, arg24_1, buf17, arg26_1, arg27_1, arg28_1, arg29_1, buf19, 602112, grid=grid(602112), stream=stream0)
        del arg21_1
        del arg22_1
        del arg23_1
        del arg24_1
        del arg26_1
        del arg27_1
        del arg28_1
        del arg29_1
        del buf17
        del buf18
        # Topologically Sorted Source Nodes: [x_173, x_174], Original ATen: [aten.relu, aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 56, 56, 56), (175616, 1, 3136, 56))
        del arg30_1
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_175, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf21, arg31_1, arg32_1, arg33_1, arg34_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg31_1
        del arg32_1
        del arg33_1
        del arg34_1
        buf22 = empty_strided_cuda((56, 8, 3, 3), (72, 1, 24, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x_175, x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(arg35_1, buf22, 448, 9, grid=grid(448, 9), stream=stream0)
        del arg35_1
        # Topologically Sorted Source Nodes: [x_175, x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf23 = extern_kernels.convolution(buf21, buf22, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=7, bias=None)
        assert_size_stride(buf23, (8, 56, 28, 28), (43904, 1, 1568, 56))
        del buf21
        del buf22
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_178, x_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf24, arg36_1, arg37_1, arg38_1, arg39_1, 351232, grid=grid(351232), stream=stream0)
        del arg36_1
        del arg37_1
        del arg38_1
        del arg39_1
        buf25 = empty_strided_cuda((8, 56, 1, 1, 7), (392, 1, 3136, 3136, 56), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_56], Original ATen: [aten.mean]
        triton_red_fused_mean_14.run(buf24, buf25, 3136, 112, grid=grid(3136), stream=stream0)
        buf27 = empty_strided_cuda((8, 56, 1, 1), (56, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_56], Original ATen: [aten.mean]
        triton_per_fused_mean_15.run(buf25, buf27, 448, 7, grid=grid(448), stream=stream0)
        del buf25
        # Topologically Sorted Source Nodes: [x_se_56, x_se_57], Original ATen: [aten.mean, aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg40_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 6, 1, 1), (6, 1, 1, 1))
        del arg40_1
        del buf27
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_se_56, x_se_57, x_se_58], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_16.run(buf29, arg41_1, 48, grid=grid(48), stream=stream0)
        del arg41_1
        # Topologically Sorted Source Nodes: [x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf30 = extern_kernels.convolution(buf29, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 56, 1, 1), (56, 1, 1, 1))
        del arg42_1
        del buf29
        buf31 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_se_56, x_se_57, x_se_58, x_se_59, sigmoid_14, x_180], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_17.run(buf31, buf30, arg43_1, 351232, grid=grid(351232), stream=stream0)
        del arg43_1
        del buf30
        # Topologically Sorted Source Nodes: [x_se_56, x_se_57, x_se_58, x_se_59, sigmoid_14, x_180, x_181], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf32 = extern_kernels.convolution(buf31, arg44_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 56, 28, 28), (43904, 1, 1568, 56))
        del arg44_1
        # Topologically Sorted Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf19, arg49_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 56, 28, 28), (43904, 1, 1568, 56))
        del arg49_1
        del buf19
        buf34 = buf32; del buf32  # reuse
        buf35 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_182, x_184, x_185, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf34, arg45_1, arg46_1, arg47_1, arg48_1, buf33, arg50_1, arg51_1, arg52_1, arg53_1, buf35, 351232, grid=grid(351232), stream=stream0)
        del arg45_1
        del arg46_1
        del arg47_1
        del arg48_1
        del arg50_1
        del arg51_1
        del arg52_1
        del arg53_1
        del buf33
        del buf34
        # Topologically Sorted Source Nodes: [x_186, x_187], Original ATen: [aten.relu, aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 152, 28, 28), (119168, 1, 4256, 152))
        del arg54_1
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_188, x_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf37, arg55_1, arg56_1, arg57_1, arg58_1, 953344, grid=grid(953344), stream=stream0)
        del arg55_1
        del arg56_1
        del arg57_1
        del arg58_1
        buf38 = empty_strided_cuda((152, 8, 3, 3), (72, 1, 24, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x_188, x_189, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(arg59_1, buf38, 1216, 9, grid=grid(1216, 9), stream=stream0)
        del arg59_1
        # Topologically Sorted Source Nodes: [x_188, x_189, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf39 = extern_kernels.convolution(buf37, buf38, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
        assert_size_stride(buf39, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del buf37
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_191, x_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf40, arg60_1, arg61_1, arg62_1, arg63_1, 238336, grid=grid(238336), stream=stream0)
        del arg60_1
        del arg61_1
        del arg62_1
        del arg63_1
        buf41 = empty_strided_cuda((8, 152, 1, 1, 2), (304, 1, 2432, 2432, 152), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_60], Original ATen: [aten.mean]
        triton_red_fused_mean_22.run(buf40, buf41, 2432, 98, grid=grid(2432), stream=stream0)
        buf43 = empty_strided_cuda((8, 152, 1, 1), (152, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_60], Original ATen: [aten.mean]
        triton_per_fused_mean_23.run(buf41, buf43, 1216, 2, grid=grid(1216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_60, x_se_61], Original ATen: [aten.mean, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg64_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 14, 1, 1), (14, 1, 1, 1))
        del arg64_1
        del buf43
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [x_se_60, x_se_61, x_se_62], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_24.run(buf45, arg65_1, 112, grid=grid(112), stream=stream0)
        del arg65_1
        # Topologically Sorted Source Nodes: [x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf46 = extern_kernels.convolution(buf45, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 152, 1, 1), (152, 1, 1, 1))
        del arg66_1
        del buf45
        buf47 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_se_60, x_se_61, x_se_62, x_se_63, sigmoid_15, x_193], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_25.run(buf47, buf46, arg67_1, 238336, grid=grid(238336), stream=stream0)
        del arg67_1
        # Topologically Sorted Source Nodes: [x_se_60, x_se_61, x_se_62, x_se_63, sigmoid_15, x_193, x_194], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf48 = extern_kernels.convolution(buf47, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del arg68_1
        # Topologically Sorted Source Nodes: [x_196], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf35, arg73_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del arg73_1
        del buf35
        buf50 = buf48; del buf48  # reuse
        buf51 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_195, x_197, x_198, x_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26.run(buf50, arg69_1, arg70_1, arg71_1, arg72_1, buf49, arg74_1, arg75_1, arg76_1, arg77_1, buf51, 238336, grid=grid(238336), stream=stream0)
        del arg69_1
        del arg70_1
        del arg71_1
        del arg72_1
        del arg74_1
        del arg75_1
        del arg76_1
        del arg77_1
        del buf49
        del buf50
        # Topologically Sorted Source Nodes: [x_199, x_200], Original ATen: [aten.relu, aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del arg78_1
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_201, x_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf53, arg79_1, arg80_1, arg81_1, arg82_1, 238336, grid=grid(238336), stream=stream0)
        del arg79_1
        del arg80_1
        del arg81_1
        del arg82_1
        buf54 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_201, x_202, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(arg83_1, buf54, 1216, 9, grid=grid(1216, 9), stream=stream0)
        del arg83_1
        # Topologically Sorted Source Nodes: [x_201, x_202, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf55 = extern_kernels.convolution(buf53, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
        assert_size_stride(buf55, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del buf53
        buf56 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_204, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf56, arg84_1, arg85_1, arg86_1, arg87_1, 238336, grid=grid(238336), stream=stream0)
        del arg84_1
        del arg85_1
        del arg86_1
        del arg87_1
        buf57 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_se_64], Original ATen: [aten.mean]
        triton_red_fused_mean_22.run(buf56, buf57, 2432, 98, grid=grid(2432), stream=stream0)
        buf59 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_se_64], Original ATen: [aten.mean]
        triton_per_fused_mean_23.run(buf57, buf59, 1216, 2, grid=grid(1216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_64, x_se_65], Original ATen: [aten.mean, aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg88_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 38, 1, 1), (38, 1, 1, 1))
        del arg88_1
        del buf59
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_se_64, x_se_65, x_se_66], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_27.run(buf61, arg89_1, 304, grid=grid(304), stream=stream0)
        del arg89_1
        # Topologically Sorted Source Nodes: [x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf62 = extern_kernels.convolution(buf61, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 152, 1, 1), (152, 1, 1, 1))
        del arg90_1
        del buf61
        buf63 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_se_64, x_se_65, x_se_66, x_se_67, sigmoid_16, x_206], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_25.run(buf63, buf62, arg91_1, 238336, grid=grid(238336), stream=stream0)
        del arg91_1
        # Topologically Sorted Source Nodes: [x_se_64, x_se_65, x_se_66, x_se_67, sigmoid_16, x_206, x_207], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf64 = extern_kernels.convolution(buf63, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del arg92_1
        del buf63
        buf65 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_208, x_209, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf65, buf64, arg93_1, arg94_1, arg95_1, arg96_1, 238336, grid=grid(238336), stream=stream0)
        del arg93_1
        del arg94_1
        del arg95_1
        del arg96_1
        del buf64
        # Topologically Sorted Source Nodes: [x_211], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del arg97_1
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_212, x_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf67, arg98_1, arg99_1, arg100_1, arg101_1, 238336, grid=grid(238336), stream=stream0)
        del arg100_1
        del arg101_1
        del arg98_1
        del arg99_1
        buf68 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_212, x_213, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(arg102_1, buf68, 1216, 9, grid=grid(1216, 9), stream=stream0)
        del arg102_1
        # Topologically Sorted Source Nodes: [x_212, x_213, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf69 = extern_kernels.convolution(buf67, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
        assert_size_stride(buf69, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del buf67
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_215, x_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf70, arg103_1, arg104_1, arg105_1, arg106_1, 238336, grid=grid(238336), stream=stream0)
        del arg103_1
        del arg104_1
        del arg105_1
        del arg106_1
        buf71 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_se_68], Original ATen: [aten.mean]
        triton_red_fused_mean_22.run(buf70, buf71, 2432, 98, grid=grid(2432), stream=stream0)
        buf73 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_se_68], Original ATen: [aten.mean]
        triton_per_fused_mean_23.run(buf71, buf73, 1216, 2, grid=grid(1216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_68, x_se_69], Original ATen: [aten.mean, aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg107_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 38, 1, 1), (38, 1, 1, 1))
        del arg107_1
        del buf73
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_se_68, x_se_69, x_se_70], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_27.run(buf75, arg108_1, 304, grid=grid(304), stream=stream0)
        del arg108_1
        # Topologically Sorted Source Nodes: [x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf76 = extern_kernels.convolution(buf75, arg109_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 152, 1, 1), (152, 1, 1, 1))
        del arg109_1
        del buf75
        buf77 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_se_68, x_se_69, x_se_70, x_se_71, sigmoid_17, x_217], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_25.run(buf77, buf76, arg110_1, 238336, grid=grid(238336), stream=stream0)
        del arg110_1
        # Topologically Sorted Source Nodes: [x_se_68, x_se_69, x_se_70, x_se_71, sigmoid_17, x_217, x_218], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf78 = extern_kernels.convolution(buf77, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del arg111_1
        del buf77
        buf79 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_219, x_220, x_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf79, buf78, arg112_1, arg113_1, arg114_1, arg115_1, 238336, grid=grid(238336), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        del buf78
        # Topologically Sorted Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del arg116_1
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_223, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf81, arg117_1, arg118_1, arg119_1, arg120_1, 238336, grid=grid(238336), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        buf82 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_223, x_224, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(arg121_1, buf82, 1216, 9, grid=grid(1216, 9), stream=stream0)
        del arg121_1
        # Topologically Sorted Source Nodes: [x_223, x_224, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf83 = extern_kernels.convolution(buf81, buf82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
        assert_size_stride(buf83, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del buf81
        del buf82
        buf84 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_226, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf84, arg122_1, arg123_1, arg124_1, arg125_1, 238336, grid=grid(238336), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        buf85 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_se_72], Original ATen: [aten.mean]
        triton_red_fused_mean_22.run(buf84, buf85, 2432, 98, grid=grid(2432), stream=stream0)
        buf87 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_se_72], Original ATen: [aten.mean]
        triton_per_fused_mean_23.run(buf85, buf87, 1216, 2, grid=grid(1216), stream=stream0)
        del buf85
        # Topologically Sorted Source Nodes: [x_se_72, x_se_73], Original ATen: [aten.mean, aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 38, 1, 1), (38, 1, 1, 1))
        del arg126_1
        del buf87
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_se_72, x_se_73, x_se_74], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_27.run(buf89, arg127_1, 304, grid=grid(304), stream=stream0)
        del arg127_1
        # Topologically Sorted Source Nodes: [x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf90 = extern_kernels.convolution(buf89, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 152, 1, 1), (152, 1, 1, 1))
        del arg128_1
        del buf89
        buf91 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_se_72, x_se_73, x_se_74, x_se_75, sigmoid_18, x_228], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_25.run(buf91, buf90, arg129_1, 238336, grid=grid(238336), stream=stream0)
        del arg129_1
        del buf90
        # Topologically Sorted Source Nodes: [x_se_72, x_se_73, x_se_74, x_se_75, sigmoid_18, x_228, x_229], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf92 = extern_kernels.convolution(buf91, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del arg130_1
        del buf91
        buf93 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [x_230, x_231, x_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf93, buf92, arg131_1, arg132_1, arg133_1, arg134_1, 238336, grid=grid(238336), stream=stream0)
        del arg131_1
        del arg132_1
        del arg133_1
        del arg134_1
        del buf92
        # Topologically Sorted Source Nodes: [x_233], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 368, 14, 14), (72128, 1, 5152, 368))
        del arg135_1
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf95, arg136_1, arg137_1, arg138_1, arg139_1, 577024, grid=grid(577024), stream=stream0)
        del arg136_1
        del arg137_1
        del arg138_1
        del arg139_1
        buf96 = empty_strided_cuda((368, 8, 3, 3), (72, 1, 24, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x_234, x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(arg140_1, buf96, 2944, 9, grid=grid(2944, 9), stream=stream0)
        del arg140_1
        # Topologically Sorted Source Nodes: [x_234, x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf97 = extern_kernels.convolution(buf95, buf96, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf97, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del buf95
        buf98 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf98, arg141_1, arg142_1, arg143_1, arg144_1, 144256, grid=grid(144256), stream=stream0)
        del arg141_1
        del arg142_1
        del arg143_1
        del arg144_1
        buf100 = empty_strided_cuda((8, 368, 1, 1), (368, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_76], Original ATen: [aten.mean]
        triton_per_fused_mean_32.run(buf98, buf100, 2944, 49, grid=grid(2944), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_76, x_se_77], Original ATen: [aten.mean, aten.convolution]
        buf101 = extern_kernels.convolution(buf100, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 38, 1, 1), (38, 1, 1, 1))
        del arg145_1
        del buf100
        buf102 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_se_76, x_se_77, x_se_78], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_27.run(buf102, arg146_1, 304, grid=grid(304), stream=stream0)
        del arg146_1
        # Topologically Sorted Source Nodes: [x_se_76, x_se_77, x_se_78, x_se_79], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf103 = extern_kernels.convolution(buf102, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg147_1
        del buf102
        buf104 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, x_239], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_33.run(buf104, buf103, arg148_1, 144256, grid=grid(144256), stream=stream0)
        del arg148_1
        # Topologically Sorted Source Nodes: [x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, x_239, x_240], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf105 = extern_kernels.convolution(buf104, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg149_1
        # Topologically Sorted Source Nodes: [x_242], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf93, arg154_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg154_1
        del buf93
        buf107 = buf105; del buf105  # reuse
        buf108 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_243, x_244, x_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf107, arg150_1, arg151_1, arg152_1, arg153_1, buf106, arg155_1, arg156_1, arg157_1, arg158_1, buf108, 144256, grid=grid(144256), stream=stream0)
        del arg150_1
        del arg151_1
        del arg152_1
        del arg153_1
        del arg155_1
        del arg156_1
        del arg157_1
        del arg158_1
        del buf106
        del buf107
        # Topologically Sorted Source Nodes: [x_245, x_246], Original ATen: [aten.relu, aten.convolution]
        buf109 = extern_kernels.convolution(buf108, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg159_1
        buf110 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_247, x_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf110, arg160_1, arg161_1, arg162_1, arg163_1, 144256, grid=grid(144256), stream=stream0)
        del arg160_1
        del arg161_1
        del arg162_1
        del arg163_1
        buf111 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_247, x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(arg164_1, buf111, 2944, 9, grid=grid(2944, 9), stream=stream0)
        del arg164_1
        # Topologically Sorted Source Nodes: [x_247, x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf112 = extern_kernels.convolution(buf110, buf111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf112, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del buf110
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_250, x_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf113, arg165_1, arg166_1, arg167_1, arg168_1, 144256, grid=grid(144256), stream=stream0)
        del arg165_1
        del arg166_1
        del arg167_1
        del arg168_1
        buf115 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [x_se_80], Original ATen: [aten.mean]
        triton_per_fused_mean_32.run(buf113, buf115, 2944, 49, grid=grid(2944), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_80, x_se_81], Original ATen: [aten.mean, aten.convolution]
        buf116 = extern_kernels.convolution(buf115, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 92, 1, 1), (92, 1, 1, 1))
        del arg169_1
        del buf115
        buf117 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_se_80, x_se_81, x_se_82], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_35.run(buf117, arg170_1, 736, grid=grid(736), stream=stream0)
        del arg170_1
        # Topologically Sorted Source Nodes: [x_se_80, x_se_81, x_se_82, x_se_83], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf118 = extern_kernels.convolution(buf117, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg171_1
        del buf117
        buf119 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, x_252], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_33.run(buf119, buf118, arg172_1, 144256, grid=grid(144256), stream=stream0)
        del arg172_1
        # Topologically Sorted Source Nodes: [x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, x_252, x_253], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf120 = extern_kernels.convolution(buf119, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg173_1
        del buf119
        buf121 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_254, x_255, x_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf121, buf120, arg174_1, arg175_1, arg176_1, arg177_1, 144256, grid=grid(144256), stream=stream0)
        del arg174_1
        del arg175_1
        del arg176_1
        del arg177_1
        del buf120
        # Topologically Sorted Source Nodes: [x_257], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg178_1
        buf123 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_258, x_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf123, arg179_1, arg180_1, arg181_1, arg182_1, 144256, grid=grid(144256), stream=stream0)
        del arg179_1
        del arg180_1
        del arg181_1
        del arg182_1
        buf124 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [x_258, x_259, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(arg183_1, buf124, 2944, 9, grid=grid(2944, 9), stream=stream0)
        del arg183_1
        # Topologically Sorted Source Nodes: [x_258, x_259, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf125 = extern_kernels.convolution(buf123, buf124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf125, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del buf123
        buf126 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [x_261, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf126, arg184_1, arg185_1, arg186_1, arg187_1, 144256, grid=grid(144256), stream=stream0)
        del arg184_1
        del arg185_1
        del arg186_1
        del arg187_1
        buf128 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_se_84], Original ATen: [aten.mean]
        triton_per_fused_mean_32.run(buf126, buf128, 2944, 49, grid=grid(2944), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_84, x_se_85], Original ATen: [aten.mean, aten.convolution]
        buf129 = extern_kernels.convolution(buf128, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 92, 1, 1), (92, 1, 1, 1))
        del arg188_1
        del buf128
        buf130 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [x_se_84, x_se_85, x_se_86], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_35.run(buf130, arg189_1, 736, grid=grid(736), stream=stream0)
        del arg189_1
        # Topologically Sorted Source Nodes: [x_se_84, x_se_85, x_se_86, x_se_87], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf131 = extern_kernels.convolution(buf130, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg190_1
        del buf130
        buf132 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, x_263], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_33.run(buf132, buf131, arg191_1, 144256, grid=grid(144256), stream=stream0)
        del arg191_1
        # Topologically Sorted Source Nodes: [x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, x_263, x_264], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf133 = extern_kernels.convolution(buf132, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg192_1
        del buf132
        buf134 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [x_265, x_266, x_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf134, buf133, arg193_1, arg194_1, arg195_1, arg196_1, 144256, grid=grid(144256), stream=stream0)
        del arg193_1
        del arg194_1
        del arg195_1
        del arg196_1
        del buf133
        # Topologically Sorted Source Nodes: [x_268], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg197_1
        buf136 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [x_269, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf136, arg198_1, arg199_1, arg200_1, arg201_1, 144256, grid=grid(144256), stream=stream0)
        del arg198_1
        del arg199_1
        del arg200_1
        del arg201_1
        buf137 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_269, x_270, x_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(arg202_1, buf137, 2944, 9, grid=grid(2944, 9), stream=stream0)
        del arg202_1
        # Topologically Sorted Source Nodes: [x_269, x_270, x_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf138 = extern_kernels.convolution(buf136, buf137, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf138, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del buf136
        buf139 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf139, arg203_1, arg204_1, arg205_1, arg206_1, 144256, grid=grid(144256), stream=stream0)
        del arg203_1
        del arg204_1
        del arg205_1
        del arg206_1
        buf141 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_se_88], Original ATen: [aten.mean]
        triton_per_fused_mean_32.run(buf139, buf141, 2944, 49, grid=grid(2944), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_88, x_se_89], Original ATen: [aten.mean, aten.convolution]
        buf142 = extern_kernels.convolution(buf141, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 92, 1, 1), (92, 1, 1, 1))
        del arg207_1
        del buf141
        buf143 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [x_se_88, x_se_89, x_se_90], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_35.run(buf143, arg208_1, 736, grid=grid(736), stream=stream0)
        del arg208_1
        # Topologically Sorted Source Nodes: [x_se_88, x_se_89, x_se_90, x_se_91], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf144 = extern_kernels.convolution(buf143, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg209_1
        del buf143
        buf145 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, x_274], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_33.run(buf145, buf144, arg210_1, 144256, grid=grid(144256), stream=stream0)
        del arg210_1
        # Topologically Sorted Source Nodes: [x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, x_274, x_275], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf146 = extern_kernels.convolution(buf145, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg211_1
        del buf145
        buf147 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [x_276, x_277, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf147, buf146, arg212_1, arg213_1, arg214_1, arg215_1, 144256, grid=grid(144256), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        del buf146
        # Topologically Sorted Source Nodes: [x_279], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg216_1
        buf149 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_280, x_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf149, arg217_1, arg218_1, arg219_1, arg220_1, 144256, grid=grid(144256), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        buf150 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [x_280, x_281, x_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(arg221_1, buf150, 2944, 9, grid=grid(2944, 9), stream=stream0)
        del arg221_1
        # Topologically Sorted Source Nodes: [x_280, x_281, x_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf151 = extern_kernels.convolution(buf149, buf150, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf151, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del buf149
        buf152 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [x_283, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf152, arg222_1, arg223_1, arg224_1, arg225_1, 144256, grid=grid(144256), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        buf154 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_se_92], Original ATen: [aten.mean]
        triton_per_fused_mean_32.run(buf152, buf154, 2944, 49, grid=grid(2944), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_92, x_se_93], Original ATen: [aten.mean, aten.convolution]
        buf155 = extern_kernels.convolution(buf154, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (8, 92, 1, 1), (92, 1, 1, 1))
        del arg226_1
        del buf154
        buf156 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_se_92, x_se_93, x_se_94], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_35.run(buf156, arg227_1, 736, grid=grid(736), stream=stream0)
        del arg227_1
        # Topologically Sorted Source Nodes: [x_se_92, x_se_93, x_se_94, x_se_95], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf157 = extern_kernels.convolution(buf156, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg228_1
        del buf156
        buf158 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, x_285], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_33.run(buf158, buf157, arg229_1, 144256, grid=grid(144256), stream=stream0)
        del arg229_1
        # Topologically Sorted Source Nodes: [x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, x_285, x_286], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf159 = extern_kernels.convolution(buf158, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg230_1
        del buf158
        buf160 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_287, x_288, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf160, buf159, arg231_1, arg232_1, arg233_1, arg234_1, 144256, grid=grid(144256), stream=stream0)
        del arg231_1
        del arg232_1
        del arg233_1
        del arg234_1
        del buf159
        # Topologically Sorted Source Nodes: [x_290], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg235_1
        buf162 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_291, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf162, arg236_1, arg237_1, arg238_1, arg239_1, 144256, grid=grid(144256), stream=stream0)
        del arg236_1
        del arg237_1
        del arg238_1
        del arg239_1
        buf163 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [x_291, x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(arg240_1, buf163, 2944, 9, grid=grid(2944, 9), stream=stream0)
        del arg240_1
        # Topologically Sorted Source Nodes: [x_291, x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf164 = extern_kernels.convolution(buf162, buf163, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf164, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del buf162
        buf165 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [x_294, x_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf165, arg241_1, arg242_1, arg243_1, arg244_1, 144256, grid=grid(144256), stream=stream0)
        del arg241_1
        del arg242_1
        del arg243_1
        del arg244_1
        buf167 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_se_96], Original ATen: [aten.mean]
        triton_per_fused_mean_32.run(buf165, buf167, 2944, 49, grid=grid(2944), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_96, x_se_97], Original ATen: [aten.mean, aten.convolution]
        buf168 = extern_kernels.convolution(buf167, arg245_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 92, 1, 1), (92, 1, 1, 1))
        del arg245_1
        del buf167
        buf169 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_se_96, x_se_97, x_se_98], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_35.run(buf169, arg246_1, 736, grid=grid(736), stream=stream0)
        del arg246_1
        # Topologically Sorted Source Nodes: [x_se_96, x_se_97, x_se_98, x_se_99], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf170 = extern_kernels.convolution(buf169, arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg247_1
        del buf169
        buf171 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_se_96, x_se_97, x_se_98, x_se_99, sigmoid_24, x_296], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_33.run(buf171, buf170, arg248_1, 144256, grid=grid(144256), stream=stream0)
        del arg248_1
        # Topologically Sorted Source Nodes: [x_se_96, x_se_97, x_se_98, x_se_99, sigmoid_24, x_296, x_297], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf172 = extern_kernels.convolution(buf171, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg249_1
        del buf171
        buf173 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_298, x_299, x_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf173, buf172, arg250_1, arg251_1, arg252_1, arg253_1, 144256, grid=grid(144256), stream=stream0)
        del arg250_1
        del arg251_1
        del arg252_1
        del arg253_1
        del buf172
        # Topologically Sorted Source Nodes: [x_301], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, arg254_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg254_1
        buf175 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [x_302, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf175, arg255_1, arg256_1, arg257_1, arg258_1, 144256, grid=grid(144256), stream=stream0)
        del arg255_1
        del arg256_1
        del arg257_1
        del arg258_1
        buf176 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_302, x_303, x_304], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(arg259_1, buf176, 2944, 9, grid=grid(2944, 9), stream=stream0)
        del arg259_1
        # Topologically Sorted Source Nodes: [x_302, x_303, x_304], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf177 = extern_kernels.convolution(buf175, buf176, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf177, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del buf175
        del buf176
        buf178 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_305, x_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf178, arg260_1, arg261_1, arg262_1, arg263_1, 144256, grid=grid(144256), stream=stream0)
        del arg260_1
        del arg261_1
        del arg262_1
        del arg263_1
        buf180 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [x_se_100], Original ATen: [aten.mean]
        triton_per_fused_mean_32.run(buf178, buf180, 2944, 49, grid=grid(2944), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_100, x_se_101], Original ATen: [aten.mean, aten.convolution]
        buf181 = extern_kernels.convolution(buf180, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 92, 1, 1), (92, 1, 1, 1))
        del arg264_1
        del buf180
        buf182 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [x_se_100, x_se_101, x_se_102], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_35.run(buf182, arg265_1, 736, grid=grid(736), stream=stream0)
        del arg265_1
        # Topologically Sorted Source Nodes: [x_se_100, x_se_101, x_se_102, x_se_103], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf183 = extern_kernels.convolution(buf182, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 368, 1, 1), (368, 1, 1, 1))
        del arg266_1
        del buf182
        buf184 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_se_100, x_se_101, x_se_102, x_se_103, sigmoid_25, x_307], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_33.run(buf184, buf183, arg267_1, 144256, grid=grid(144256), stream=stream0)
        del arg267_1
        # Topologically Sorted Source Nodes: [x_se_100, x_se_101, x_se_102, x_se_103, sigmoid_25, x_307, x_308], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf185 = extern_kernels.convolution(buf184, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 368, 7, 7), (18032, 1, 2576, 368))
        del arg268_1
        del buf184
        buf187 = reinterpret_tensor(buf183, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [x_309, x_310, x_311, x_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_37.run(buf185, arg269_1, arg270_1, arg271_1, arg272_1, buf173, buf187, 2944, 49, grid=grid(2944), stream=stream0)
        del arg269_1
        del arg270_1
        del arg271_1
        del arg272_1
        del buf173
        del buf185
        buf188 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_315], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg274_1, reinterpret_tensor(buf187, (8, 368), (368, 1), 0), reinterpret_tensor(arg273_1, (368, 1000), (1, 368), 0), alpha=1, beta=1, out=buf188)
        del arg273_1
        del arg274_1
        del buf187
    return (buf188, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((24, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((8, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((24, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((56, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((56, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((6, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((56, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((56, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((152, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((14, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((152, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((152, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((368, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((38, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((368, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((368, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1000, 368), (368, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('regnety_002', benchmark_compiled_module)
