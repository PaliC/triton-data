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


# kernel path: /tmp/torchinductor_sahanp/eh/ceht27vbsmlt6ox5m7xhqumvk5yvhnv6w4p3xfjvfagojzfql7rr.py
# Topologically Sorted Source Nodes: [x_171], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_171 => convolution_57
# Graph fragment:
#   %convolution_57 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (65536*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (196608*y1)), tmp0, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fn/cfnqc5t7ruox32h42p5hkjg336a6auwfmlumtz3g3aru5iaouonc.py
# Topologically Sorted Source Nodes: [x_171], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_171 => convolution_57
# Graph fragment:
#   %convolution_57 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/s6/cs6ptkkc47kcn7653ugdpjqxukhsejmviojy2auetqadio5sc4ob.py
# Topologically Sorted Source Nodes: [x_172, x_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_172 => add_133, mul_172, mul_173, sub_57
#   x_173 => relu_53
# Graph fragment:
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_57, %unsqueeze_457), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_459), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %unsqueeze_461), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %unsqueeze_463), kwargs = {})
#   %relu_53 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_133,), kwargs = {})
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
    xnumel = 4194304
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


# kernel path: /tmp/torchinductor_sahanp/sp/cspi44bguxmbkrxygvgy3tqjdmttfre5osfopwrftcicb222qs2a.py
# Topologically Sorted Source Nodes: [x_174], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_174 => convolution_58
# Graph fragment:
#   %convolution_58 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_53, %arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
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


# kernel path: /tmp/torchinductor_sahanp/ne/cne34eagguzzwchxds26etggteuntk3bojgl67fxbj6mp5cyh7en.py
# Topologically Sorted Source Nodes: [x_175, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_175 => add_135, mul_175, mul_176, sub_58
#   x_176 => relu_54
# Graph fragment:
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_58, %unsqueeze_465), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_467), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %unsqueeze_469), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %unsqueeze_471), kwargs = {})
#   %relu_54 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_135,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /tmp/torchinductor_sahanp/xi/cxixowz3kddujpeqd7xrmhjkde5uswwhvi4rjf5wiv4zqmt4d47g.py
# Topologically Sorted Source Nodes: [x_175, x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_175 => add_135, mul_175, mul_176, sub_58
#   x_176 => relu_54
#   x_177 => convolution_59
# Graph fragment:
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_58, %unsqueeze_465), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_467), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %unsqueeze_469), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %unsqueeze_471), kwargs = {})
#   %relu_54 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_135,), kwargs = {})
#   %convolution_59 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_54, %arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dw/cdwyhhz3c46jbp6o6swhvghtte6jgxo6vc7znb7tee5nrrngwbyn.py
# Topologically Sorted Source Nodes: [x_178, x_180, x_181, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_19 => relu_55
#   x_178 => add_137, mul_178, mul_179, sub_59
#   x_180 => add_139, mul_181, mul_182, sub_60
#   x_181 => add_140
# Graph fragment:
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_59, %unsqueeze_473), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_475), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_178, %unsqueeze_477), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_179, %unsqueeze_479), kwargs = {})
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_60, %unsqueeze_481), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_483), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_181, %unsqueeze_485), kwargs = {})
#   %add_139 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_182, %unsqueeze_487), kwargs = {})
#   %add_140 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_137, %add_139), kwargs = {})
#   %relu_55 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_140,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /tmp/torchinductor_sahanp/fk/cfkcgmhgdybc2g7sl374cbiokgwb5dbxtfcp6mqyit6zbqhgdj5m.py
# Topologically Sorted Source Nodes: [input_19, x_182], Original ATen: [aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_19 => relu_55
#   x_182 => convolution_61
# Graph fragment:
#   %relu_55 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_140,), kwargs = {})
#   %convolution_61 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_55, %arg21_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_relu_7 = async_compile.triton('triton_poi_fused_convolution_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vs/cvsu3turneuaycertqmgspxssdrg47laitl7uumbotvqpr56niaa.py
# Topologically Sorted Source Nodes: [x_183, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_183 => add_142, mul_184, mul_185, sub_61
#   x_184 => relu_56
# Graph fragment:
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_61, %unsqueeze_489), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %unsqueeze_493), kwargs = {})
#   %add_142 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_495), kwargs = {})
#   %relu_56 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_142,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 192
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


# kernel path: /tmp/torchinductor_sahanp/eg/cegih62xmlrrtuyiu5mqrffvlc7u7lndxx33ibsdaf3vs7y6icc6.py
# Topologically Sorted Source Nodes: [x_183, x_184, x_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_183 => add_142, mul_184, mul_185, sub_61
#   x_184 => relu_56
#   x_185 => convolution_62
# Graph fragment:
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_61, %unsqueeze_489), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %unsqueeze_493), kwargs = {})
#   %add_142 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_495), kwargs = {})
#   %relu_56 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_142,), kwargs = {})
#   %convolution_62 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_56, %arg26_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
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


# kernel path: /tmp/torchinductor_sahanp/pe/cpeibvs3g4bobk3vjxmqjirait7iop7sa2nll7zozccl2qlvkd2f.py
# Topologically Sorted Source Nodes: [x_186, x_188, x_189, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_20 => relu_57
#   x_186 => add_144, mul_187, mul_188, sub_62
#   x_188 => add_146, mul_190, mul_191, sub_63
#   x_189 => add_147
# Graph fragment:
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_62, %unsqueeze_497), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %unsqueeze_499), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_187, %unsqueeze_501), kwargs = {})
#   %add_144 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_188, %unsqueeze_503), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_63, %unsqueeze_505), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_190, %unsqueeze_509), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %unsqueeze_511), kwargs = {})
#   %add_147 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_144, %add_146), kwargs = {})
#   %relu_57 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_147,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 192
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


# kernel path: /tmp/torchinductor_sahanp/mz/cmzsl5godwtnsjqzhrgchphkz6m42sfg5addhy5zmtqhxgxfelf6.py
# Topologically Sorted Source Nodes: [x_194, x_195, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_21 => relu_59
#   x_194 => add_151, mul_196, mul_197, sub_65
#   x_195 => add_152
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_65, %unsqueeze_521), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_196, %unsqueeze_525), kwargs = {})
#   %add_151 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_197, %unsqueeze_527), kwargs = {})
#   %add_152 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_151, %relu_57), kwargs = {})
#   %relu_59 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_152,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jq/cjqduglbev4pru7ekmmdl3uvlouhjyuipvr2dx657ryotbkso5f7.py
# Topologically Sorted Source Nodes: [x_197, x_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_197 => add_154, mul_199, mul_200, sub_66
#   x_198 => relu_60
# Graph fragment:
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_66, %unsqueeze_529), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %unsqueeze_533), kwargs = {})
#   %add_154 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %unsqueeze_535), kwargs = {})
#   %relu_60 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_154,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 160
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


# kernel path: /tmp/torchinductor_sahanp/co/cco5aco437q6l7jrcapestm3brpekddyd3w4frun4lzqyxpunl36.py
# Topologically Sorted Source Nodes: [x_197, x_198, x_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_197 => add_154, mul_199, mul_200, sub_66
#   x_198 => relu_60
#   x_199 => convolution_67
# Graph fragment:
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_66, %unsqueeze_529), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %unsqueeze_533), kwargs = {})
#   %add_154 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %unsqueeze_535), kwargs = {})
#   %relu_60 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_154,), kwargs = {})
#   %convolution_67 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_60, %arg51_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25600
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (1440*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2u/c2up5q6hkwxdkzpqpfjyxmnr3biouoesxicdxcqbox3lfubit6tg.py
# Topologically Sorted Source Nodes: [x_200, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_200 => add_156, mul_202, mul_203, sub_67
#   x_201 => relu_61
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_67, %unsqueeze_537), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %unsqueeze_541), kwargs = {})
#   %add_156 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %unsqueeze_543), kwargs = {})
#   %relu_61 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_156,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 160
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


# kernel path: /tmp/torchinductor_sahanp/jm/cjm2se4mryk346ivyghydrwml3pa322juzc2i75kiveju3yauhiu.py
# Topologically Sorted Source Nodes: [x_203, x_205, x_206, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_22 => relu_62
#   x_203 => add_158, mul_205, mul_206, sub_68
#   x_205 => add_160, mul_208, mul_209, sub_69
#   x_206 => add_161
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_545), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_205, %unsqueeze_549), kwargs = {})
#   %add_158 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_206, %unsqueeze_551), kwargs = {})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_69, %unsqueeze_553), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_208, %unsqueeze_557), kwargs = {})
#   %add_160 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_209, %unsqueeze_559), kwargs = {})
#   %add_161 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_158, %add_160), kwargs = {})
#   %relu_62 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_161,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 640
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


# kernel path: /tmp/torchinductor_sahanp/tj/ctj4osej2kjtp6mcl7ovkcf2oioqihwv5u6hgxxftfjg3duw7j5s.py
# Topologically Sorted Source Nodes: [x_214, x_215, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_23 => relu_65
#   x_214 => add_167, mul_217, mul_218, sub_72
#   x_215 => add_168
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_72, %unsqueeze_577), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_217, %unsqueeze_581), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_218, %unsqueeze_583), kwargs = {})
#   %add_168 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_167, %relu_62), kwargs = {})
#   %relu_65 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_168,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 640
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zm/czmrcutkxgu3udvxh52jgtjxnbykjfq23p7rzdp45dvmbs3ssajc.py
# Topologically Sorted Source Nodes: [x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_253 => add_198, mul_256, mul_257, sub_85
#   x_254 => relu_78
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_85, %unsqueeze_681), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %unsqueeze_685), kwargs = {})
#   %add_198 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %unsqueeze_687), kwargs = {})
#   %relu_78 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_198,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3932160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1920
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


# kernel path: /tmp/torchinductor_sahanp/pd/cpddvczm7p44fn3n25vdahin6fteeynmwhhv4fxsjwjqpwxphela.py
# Topologically Sorted Source Nodes: [x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_256 => add_200, mul_259, mul_260, sub_86
#   x_257 => relu_79
# Graph fragment:
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_86, %unsqueeze_689), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_86, %unsqueeze_691), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_259, %unsqueeze_693), kwargs = {})
#   %add_200 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_260, %unsqueeze_695), kwargs = {})
#   %relu_79 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_200,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1920
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


# kernel path: /tmp/torchinductor_sahanp/k6/ck6lscgoygnrjimxxtv7plpnwhs53zocwollij3pbnx7jbpt43yy.py
# Topologically Sorted Source Nodes: [x_259, x_261, x_262, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_28 => relu_80
#   x_259 => add_202, mul_262, mul_263, sub_87
#   x_261 => add_204, mul_265, mul_266, sub_88
#   x_262 => add_205
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_697), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %unsqueeze_701), kwargs = {})
#   %add_202 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %unsqueeze_703), kwargs = {})
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_88, %unsqueeze_705), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_707), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_265, %unsqueeze_709), kwargs = {})
#   %add_204 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_266, %unsqueeze_711), kwargs = {})
#   %add_205 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_202, %add_204), kwargs = {})
#   %relu_80 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_205,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 640
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


# kernel path: /tmp/torchinductor_sahanp/ab/cab6ve3kvydkbc36iych62ngu4zb2ymeixuolcrajcsozyawjtz5.py
# Topologically Sorted Source Nodes: [x_270, x_271, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_29 => relu_83
#   x_270 => add_211, mul_274, mul_275, sub_91
#   x_271 => add_212
# Graph fragment:
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_91, %unsqueeze_729), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %unsqueeze_731), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_274, %unsqueeze_733), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_275, %unsqueeze_735), kwargs = {})
#   %add_212 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_211, %relu_80), kwargs = {})
#   %relu_83 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_212,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 640
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tf/ctflegsotpqacuqsnezxwreokwmg6uwbnz5iz5yl7u56uegoo3zc.py
# Topologically Sorted Source Nodes: [x_297, x_298, input_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_32 => relu_92
#   x_297 => add_232, mul_301, mul_302, sub_100
#   x_298 => add_233
# Graph fragment:
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_801), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_803), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_301, %unsqueeze_805), kwargs = {})
#   %add_232 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_302, %unsqueeze_807), kwargs = {})
#   %add_233 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_232, %relu_89), kwargs = {})
#   %relu_92 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_233,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 640
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ec/cecpwwfxd4qrujt6mfy2usik4zsg2nchnfehxsolbm3pinvgwzna.py
# Topologically Sorted Source Nodes: [x_336, x_337, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   x_336 => add_263, mul_340, mul_341, sub_113
#   x_337 => relu_105
#   x_338 => mean_1
# Graph fragment:
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_113, %unsqueeze_905), kwargs = {})
#   %mul_340 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %unsqueeze_907), kwargs = {})
#   %mul_341 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_340, %unsqueeze_909), kwargs = {})
#   %add_263 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_341, %unsqueeze_911), kwargs = {})
#   %relu_105 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_263,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_105, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_22 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_mean_relu_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_mean_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 2560
    x1 = (xindex // 2560)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2560*r2) + (163840*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.sum(tmp18, 1)[:, None]
    tmp21 = 64.0
    tmp22 = tmp20 / tmp21
    tl.store(out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (192, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg22_1, (192, ), (1, ))
    assert_size_stride(arg23_1, (192, ), (1, ))
    assert_size_stride(arg24_1, (192, ), (1, ))
    assert_size_stride(arg25_1, (192, ), (1, ))
    assert_size_stride(arg26_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg27_1, (192, ), (1, ))
    assert_size_stride(arg28_1, (192, ), (1, ))
    assert_size_stride(arg29_1, (192, ), (1, ))
    assert_size_stride(arg30_1, (192, ), (1, ))
    assert_size_stride(arg31_1, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg32_1, (192, ), (1, ))
    assert_size_stride(arg33_1, (192, ), (1, ))
    assert_size_stride(arg34_1, (192, ), (1, ))
    assert_size_stride(arg35_1, (192, ), (1, ))
    assert_size_stride(arg36_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg37_1, (192, ), (1, ))
    assert_size_stride(arg38_1, (192, ), (1, ))
    assert_size_stride(arg39_1, (192, ), (1, ))
    assert_size_stride(arg40_1, (192, ), (1, ))
    assert_size_stride(arg41_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg42_1, (192, ), (1, ))
    assert_size_stride(arg43_1, (192, ), (1, ))
    assert_size_stride(arg44_1, (192, ), (1, ))
    assert_size_stride(arg45_1, (192, ), (1, ))
    assert_size_stride(arg46_1, (160, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg47_1, (160, ), (1, ))
    assert_size_stride(arg48_1, (160, ), (1, ))
    assert_size_stride(arg49_1, (160, ), (1, ))
    assert_size_stride(arg50_1, (160, ), (1, ))
    assert_size_stride(arg51_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg52_1, (160, ), (1, ))
    assert_size_stride(arg53_1, (160, ), (1, ))
    assert_size_stride(arg54_1, (160, ), (1, ))
    assert_size_stride(arg55_1, (160, ), (1, ))
    assert_size_stride(arg56_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg57_1, (640, ), (1, ))
    assert_size_stride(arg58_1, (640, ), (1, ))
    assert_size_stride(arg59_1, (640, ), (1, ))
    assert_size_stride(arg60_1, (640, ), (1, ))
    assert_size_stride(arg61_1, (640, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg62_1, (640, ), (1, ))
    assert_size_stride(arg63_1, (640, ), (1, ))
    assert_size_stride(arg64_1, (640, ), (1, ))
    assert_size_stride(arg65_1, (640, ), (1, ))
    assert_size_stride(arg66_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg67_1, (160, ), (1, ))
    assert_size_stride(arg68_1, (160, ), (1, ))
    assert_size_stride(arg69_1, (160, ), (1, ))
    assert_size_stride(arg70_1, (160, ), (1, ))
    assert_size_stride(arg71_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg72_1, (160, ), (1, ))
    assert_size_stride(arg73_1, (160, ), (1, ))
    assert_size_stride(arg74_1, (160, ), (1, ))
    assert_size_stride(arg75_1, (160, ), (1, ))
    assert_size_stride(arg76_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg77_1, (640, ), (1, ))
    assert_size_stride(arg78_1, (640, ), (1, ))
    assert_size_stride(arg79_1, (640, ), (1, ))
    assert_size_stride(arg80_1, (640, ), (1, ))
    assert_size_stride(arg81_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg82_1, (160, ), (1, ))
    assert_size_stride(arg83_1, (160, ), (1, ))
    assert_size_stride(arg84_1, (160, ), (1, ))
    assert_size_stride(arg85_1, (160, ), (1, ))
    assert_size_stride(arg86_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg87_1, (160, ), (1, ))
    assert_size_stride(arg88_1, (160, ), (1, ))
    assert_size_stride(arg89_1, (160, ), (1, ))
    assert_size_stride(arg90_1, (160, ), (1, ))
    assert_size_stride(arg91_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg92_1, (640, ), (1, ))
    assert_size_stride(arg93_1, (640, ), (1, ))
    assert_size_stride(arg94_1, (640, ), (1, ))
    assert_size_stride(arg95_1, (640, ), (1, ))
    assert_size_stride(arg96_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg97_1, (160, ), (1, ))
    assert_size_stride(arg98_1, (160, ), (1, ))
    assert_size_stride(arg99_1, (160, ), (1, ))
    assert_size_stride(arg100_1, (160, ), (1, ))
    assert_size_stride(arg101_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg102_1, (160, ), (1, ))
    assert_size_stride(arg103_1, (160, ), (1, ))
    assert_size_stride(arg104_1, (160, ), (1, ))
    assert_size_stride(arg105_1, (160, ), (1, ))
    assert_size_stride(arg106_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg107_1, (640, ), (1, ))
    assert_size_stride(arg108_1, (640, ), (1, ))
    assert_size_stride(arg109_1, (640, ), (1, ))
    assert_size_stride(arg110_1, (640, ), (1, ))
    assert_size_stride(arg111_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg112_1, (160, ), (1, ))
    assert_size_stride(arg113_1, (160, ), (1, ))
    assert_size_stride(arg114_1, (160, ), (1, ))
    assert_size_stride(arg115_1, (160, ), (1, ))
    assert_size_stride(arg116_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg117_1, (160, ), (1, ))
    assert_size_stride(arg118_1, (160, ), (1, ))
    assert_size_stride(arg119_1, (160, ), (1, ))
    assert_size_stride(arg120_1, (160, ), (1, ))
    assert_size_stride(arg121_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg122_1, (640, ), (1, ))
    assert_size_stride(arg123_1, (640, ), (1, ))
    assert_size_stride(arg124_1, (640, ), (1, ))
    assert_size_stride(arg125_1, (640, ), (1, ))
    assert_size_stride(arg126_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg127_1, (160, ), (1, ))
    assert_size_stride(arg128_1, (160, ), (1, ))
    assert_size_stride(arg129_1, (160, ), (1, ))
    assert_size_stride(arg130_1, (160, ), (1, ))
    assert_size_stride(arg131_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg132_1, (160, ), (1, ))
    assert_size_stride(arg133_1, (160, ), (1, ))
    assert_size_stride(arg134_1, (160, ), (1, ))
    assert_size_stride(arg135_1, (160, ), (1, ))
    assert_size_stride(arg136_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg137_1, (640, ), (1, ))
    assert_size_stride(arg138_1, (640, ), (1, ))
    assert_size_stride(arg139_1, (640, ), (1, ))
    assert_size_stride(arg140_1, (640, ), (1, ))
    assert_size_stride(arg141_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg142_1, (1920, ), (1, ))
    assert_size_stride(arg143_1, (1920, ), (1, ))
    assert_size_stride(arg144_1, (1920, ), (1, ))
    assert_size_stride(arg145_1, (1920, ), (1, ))
    assert_size_stride(arg146_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg147_1, (1920, ), (1, ))
    assert_size_stride(arg148_1, (1920, ), (1, ))
    assert_size_stride(arg149_1, (1920, ), (1, ))
    assert_size_stride(arg150_1, (1920, ), (1, ))
    assert_size_stride(arg151_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg152_1, (640, ), (1, ))
    assert_size_stride(arg153_1, (640, ), (1, ))
    assert_size_stride(arg154_1, (640, ), (1, ))
    assert_size_stride(arg155_1, (640, ), (1, ))
    assert_size_stride(arg156_1, (640, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg157_1, (640, ), (1, ))
    assert_size_stride(arg158_1, (640, ), (1, ))
    assert_size_stride(arg159_1, (640, ), (1, ))
    assert_size_stride(arg160_1, (640, ), (1, ))
    assert_size_stride(arg161_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg162_1, (1920, ), (1, ))
    assert_size_stride(arg163_1, (1920, ), (1, ))
    assert_size_stride(arg164_1, (1920, ), (1, ))
    assert_size_stride(arg165_1, (1920, ), (1, ))
    assert_size_stride(arg166_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg167_1, (1920, ), (1, ))
    assert_size_stride(arg168_1, (1920, ), (1, ))
    assert_size_stride(arg169_1, (1920, ), (1, ))
    assert_size_stride(arg170_1, (1920, ), (1, ))
    assert_size_stride(arg171_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg172_1, (640, ), (1, ))
    assert_size_stride(arg173_1, (640, ), (1, ))
    assert_size_stride(arg174_1, (640, ), (1, ))
    assert_size_stride(arg175_1, (640, ), (1, ))
    assert_size_stride(arg176_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg177_1, (1920, ), (1, ))
    assert_size_stride(arg178_1, (1920, ), (1, ))
    assert_size_stride(arg179_1, (1920, ), (1, ))
    assert_size_stride(arg180_1, (1920, ), (1, ))
    assert_size_stride(arg181_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg182_1, (1920, ), (1, ))
    assert_size_stride(arg183_1, (1920, ), (1, ))
    assert_size_stride(arg184_1, (1920, ), (1, ))
    assert_size_stride(arg185_1, (1920, ), (1, ))
    assert_size_stride(arg186_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg187_1, (640, ), (1, ))
    assert_size_stride(arg188_1, (640, ), (1, ))
    assert_size_stride(arg189_1, (640, ), (1, ))
    assert_size_stride(arg190_1, (640, ), (1, ))
    assert_size_stride(arg191_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg192_1, (1920, ), (1, ))
    assert_size_stride(arg193_1, (1920, ), (1, ))
    assert_size_stride(arg194_1, (1920, ), (1, ))
    assert_size_stride(arg195_1, (1920, ), (1, ))
    assert_size_stride(arg196_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg197_1, (1920, ), (1, ))
    assert_size_stride(arg198_1, (1920, ), (1, ))
    assert_size_stride(arg199_1, (1920, ), (1, ))
    assert_size_stride(arg200_1, (1920, ), (1, ))
    assert_size_stride(arg201_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg202_1, (640, ), (1, ))
    assert_size_stride(arg203_1, (640, ), (1, ))
    assert_size_stride(arg204_1, (640, ), (1, ))
    assert_size_stride(arg205_1, (640, ), (1, ))
    assert_size_stride(arg206_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg207_1, (1920, ), (1, ))
    assert_size_stride(arg208_1, (1920, ), (1, ))
    assert_size_stride(arg209_1, (1920, ), (1, ))
    assert_size_stride(arg210_1, (1920, ), (1, ))
    assert_size_stride(arg211_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg212_1, (1920, ), (1, ))
    assert_size_stride(arg213_1, (1920, ), (1, ))
    assert_size_stride(arg214_1, (1920, ), (1, ))
    assert_size_stride(arg215_1, (1920, ), (1, ))
    assert_size_stride(arg216_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg217_1, (640, ), (1, ))
    assert_size_stride(arg218_1, (640, ), (1, ))
    assert_size_stride(arg219_1, (640, ), (1, ))
    assert_size_stride(arg220_1, (640, ), (1, ))
    assert_size_stride(arg221_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg222_1, (1920, ), (1, ))
    assert_size_stride(arg223_1, (1920, ), (1, ))
    assert_size_stride(arg224_1, (1920, ), (1, ))
    assert_size_stride(arg225_1, (1920, ), (1, ))
    assert_size_stride(arg226_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg227_1, (1920, ), (1, ))
    assert_size_stride(arg228_1, (1920, ), (1, ))
    assert_size_stride(arg229_1, (1920, ), (1, ))
    assert_size_stride(arg230_1, (1920, ), (1, ))
    assert_size_stride(arg231_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg232_1, (640, ), (1, ))
    assert_size_stride(arg233_1, (640, ), (1, ))
    assert_size_stride(arg234_1, (640, ), (1, ))
    assert_size_stride(arg235_1, (640, ), (1, ))
    assert_size_stride(arg236_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg237_1, (1920, ), (1, ))
    assert_size_stride(arg238_1, (1920, ), (1, ))
    assert_size_stride(arg239_1, (1920, ), (1, ))
    assert_size_stride(arg240_1, (1920, ), (1, ))
    assert_size_stride(arg241_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg242_1, (1920, ), (1, ))
    assert_size_stride(arg243_1, (1920, ), (1, ))
    assert_size_stride(arg244_1, (1920, ), (1, ))
    assert_size_stride(arg245_1, (1920, ), (1, ))
    assert_size_stride(arg246_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg247_1, (640, ), (1, ))
    assert_size_stride(arg248_1, (640, ), (1, ))
    assert_size_stride(arg249_1, (640, ), (1, ))
    assert_size_stride(arg250_1, (640, ), (1, ))
    assert_size_stride(arg251_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg252_1, (1920, ), (1, ))
    assert_size_stride(arg253_1, (1920, ), (1, ))
    assert_size_stride(arg254_1, (1920, ), (1, ))
    assert_size_stride(arg255_1, (1920, ), (1, ))
    assert_size_stride(arg256_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg257_1, (1920, ), (1, ))
    assert_size_stride(arg258_1, (1920, ), (1, ))
    assert_size_stride(arg259_1, (1920, ), (1, ))
    assert_size_stride(arg260_1, (1920, ), (1, ))
    assert_size_stride(arg261_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg262_1, (640, ), (1, ))
    assert_size_stride(arg263_1, (640, ), (1, ))
    assert_size_stride(arg264_1, (640, ), (1, ))
    assert_size_stride(arg265_1, (640, ), (1, ))
    assert_size_stride(arg266_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg267_1, (1920, ), (1, ))
    assert_size_stride(arg268_1, (1920, ), (1, ))
    assert_size_stride(arg269_1, (1920, ), (1, ))
    assert_size_stride(arg270_1, (1920, ), (1, ))
    assert_size_stride(arg271_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg272_1, (1920, ), (1, ))
    assert_size_stride(arg273_1, (1920, ), (1, ))
    assert_size_stride(arg274_1, (1920, ), (1, ))
    assert_size_stride(arg275_1, (1920, ), (1, ))
    assert_size_stride(arg276_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg277_1, (640, ), (1, ))
    assert_size_stride(arg278_1, (640, ), (1, ))
    assert_size_stride(arg279_1, (640, ), (1, ))
    assert_size_stride(arg280_1, (640, ), (1, ))
    assert_size_stride(arg281_1, (2560, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg282_1, (2560, ), (1, ))
    assert_size_stride(arg283_1, (2560, ), (1, ))
    assert_size_stride(arg284_1, (2560, ), (1, ))
    assert_size_stride(arg285_1, (2560, ), (1, ))
    assert_size_stride(arg286_1, (1000, 2560), (2560, 1))
    assert_size_stride(arg287_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 256, 256), (196608, 1, 768, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_171], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_171], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_171], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 128, 128), (524288, 1, 4096, 32))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_172, x_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((128, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_174], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_3.run(arg6_1, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 128, 64, 64), (524288, 1, 8192, 128))
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_175, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf7 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_175, x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg11_1, buf7, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [x_175, x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 128, 64, 64), (524288, 1, 8192, 128))
        del buf6
        del buf7
        # Topologically Sorted Source Nodes: [x_179], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf3, arg16_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 128, 64, 64), (524288, 1, 8192, 128))
        del arg16_1
        buf10 = buf8; del buf8  # reuse
        buf11 = reinterpret_tensor(buf3, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_178, x_180, x_181, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf10, arg12_1, arg13_1, arg14_1, arg15_1, buf9, arg17_1, arg18_1, arg19_1, arg20_1, buf11, 4194304, grid=grid(4194304), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del buf10
        del buf9
        buf12 = empty_strided_cuda((192, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_19, x_182], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_7.run(arg21_1, buf12, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg21_1
        # Topologically Sorted Source Nodes: [input_19, x_182], Original ATen: [aten.relu, aten.convolution]
        buf13 = extern_kernels.convolution(buf11, buf12, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 192, 32, 32), (196608, 1, 6144, 192))
        del buf12
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_183, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf14, arg22_1, arg23_1, arg24_1, arg25_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        buf15 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_183, x_184, x_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg26_1, buf15, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg26_1
        # Topologically Sorted Source Nodes: [x_183, x_184, x_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 192, 32, 32), (196608, 1, 6144, 192))
        # Topologically Sorted Source Nodes: [x_187], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf11, arg31_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 192, 32, 32), (196608, 1, 6144, 192))
        del arg31_1
        del buf11
        buf18 = buf16; del buf16  # reuse
        buf19 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_186, x_188, x_189, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf18, arg27_1, arg28_1, arg29_1, arg30_1, buf17, arg32_1, arg33_1, arg34_1, arg35_1, buf19, 1572864, grid=grid(1572864), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        del buf17
        del buf18
        buf20 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [input_20, x_190], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg36_1, buf20, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg36_1
        # Topologically Sorted Source Nodes: [input_20, x_190], Original ATen: [aten.relu, aten.convolution]
        buf21 = extern_kernels.convolution(buf19, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 192, 32, 32), (196608, 1, 6144, 192))
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_191, x_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf22, arg37_1, arg38_1, arg39_1, arg40_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        buf23 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_191, x_192, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg41_1, buf23, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg41_1
        # Topologically Sorted Source Nodes: [x_191, x_192, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf24 = extern_kernels.convolution(buf22, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 192, 32, 32), (196608, 1, 6144, 192))
        del buf22
        del buf23
        buf25 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_194, x_195, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf25, buf24, arg42_1, arg43_1, arg44_1, arg45_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        del buf24
        # Topologically Sorted Source Nodes: [x_196], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 160, 32, 32), (163840, 1, 5120, 160))
        del arg46_1
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_197, x_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf27, arg47_1, arg48_1, arg49_1, arg50_1, 1310720, grid=grid(1310720), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        buf28 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_197, x_198, x_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg51_1, buf28, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg51_1
        # Topologically Sorted Source Nodes: [x_197, x_198, x_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf29 = extern_kernels.convolution(buf27, buf28, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 160, 16, 16), (40960, 1, 2560, 160))
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_200, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf30, arg52_1, arg53_1, arg54_1, arg55_1, 327680, grid=grid(327680), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        # Topologically Sorted Source Nodes: [x_200, x_201, x_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 640, 16, 16), (163840, 1, 10240, 640))
        del arg56_1
        del buf30
        # Topologically Sorted Source Nodes: [x_204], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf25, arg61_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 640, 16, 16), (163840, 1, 10240, 640))
        del arg61_1
        del buf25
        buf33 = buf31; del buf31  # reuse
        buf34 = reinterpret_tensor(buf27, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [x_203, x_205, x_206, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf33, arg57_1, arg58_1, arg59_1, arg60_1, buf32, arg62_1, arg63_1, arg64_1, arg65_1, buf34, 1310720, grid=grid(1310720), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        del buf32
        del buf33
        # Topologically Sorted Source Nodes: [input_22, x_207], Original ATen: [aten.relu, aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 160, 16, 16), (40960, 1, 2560, 160))
        del arg66_1
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [x_208, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf36, arg67_1, arg68_1, arg69_1, arg70_1, 327680, grid=grid(327680), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        buf37 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_208, x_209, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg71_1, buf37, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg71_1
        # Topologically Sorted Source Nodes: [x_208, x_209, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf38 = extern_kernels.convolution(buf36, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 160, 16, 16), (40960, 1, 2560, 160))
        del buf36
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_211, x_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf39, arg72_1, arg73_1, arg74_1, arg75_1, 327680, grid=grid(327680), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        # Topologically Sorted Source Nodes: [x_211, x_212, x_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 640, 16, 16), (163840, 1, 10240, 640))
        del arg76_1
        del buf39
        buf41 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_214, x_215, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf41, buf40, arg77_1, arg78_1, arg79_1, arg80_1, 1310720, grid=grid(1310720), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        del buf40
        # Topologically Sorted Source Nodes: [x_216], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 160, 16, 16), (40960, 1, 2560, 160))
        del arg81_1
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_217, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf43, arg82_1, arg83_1, arg84_1, arg85_1, 327680, grid=grid(327680), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        buf44 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_217, x_218, x_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg86_1, buf44, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg86_1
        # Topologically Sorted Source Nodes: [x_217, x_218, x_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf45 = extern_kernels.convolution(buf43, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 160, 16, 16), (40960, 1, 2560, 160))
        del buf43
        buf46 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [x_220, x_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf46, arg87_1, arg88_1, arg89_1, arg90_1, 327680, grid=grid(327680), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        # Topologically Sorted Source Nodes: [x_220, x_221, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 640, 16, 16), (163840, 1, 10240, 640))
        del arg91_1
        del buf46
        buf48 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_223, x_224, input_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf48, buf47, arg92_1, arg93_1, arg94_1, arg95_1, 1310720, grid=grid(1310720), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf47
        # Topologically Sorted Source Nodes: [x_225], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 160, 16, 16), (40960, 1, 2560, 160))
        del arg96_1
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [x_226, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf50, arg97_1, arg98_1, arg99_1, arg100_1, 327680, grid=grid(327680), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        buf51 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [x_226, x_227, x_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg101_1, buf51, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg101_1
        # Topologically Sorted Source Nodes: [x_226, x_227, x_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf52 = extern_kernels.convolution(buf50, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 160, 16, 16), (40960, 1, 2560, 160))
        del buf50
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_229, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf53, arg102_1, arg103_1, arg104_1, arg105_1, 327680, grid=grid(327680), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        # Topologically Sorted Source Nodes: [x_229, x_230, x_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 640, 16, 16), (163840, 1, 10240, 640))
        del arg106_1
        del buf53
        buf55 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_232, x_233, input_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf55, buf54, arg107_1, arg108_1, arg109_1, arg110_1, 1310720, grid=grid(1310720), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        del buf54
        # Topologically Sorted Source Nodes: [x_234], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 160, 16, 16), (40960, 1, 2560, 160))
        del arg111_1
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf57, arg112_1, arg113_1, arg114_1, arg115_1, 327680, grid=grid(327680), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        buf58 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_235, x_236, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg116_1, buf58, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg116_1
        # Topologically Sorted Source Nodes: [x_235, x_236, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf59 = extern_kernels.convolution(buf57, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 160, 16, 16), (40960, 1, 2560, 160))
        del buf57
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_238, x_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf60, arg117_1, arg118_1, arg119_1, arg120_1, 327680, grid=grid(327680), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        # Topologically Sorted Source Nodes: [x_238, x_239, x_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf61 = extern_kernels.convolution(buf60, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (8, 640, 16, 16), (163840, 1, 10240, 640))
        del arg121_1
        del buf60
        buf62 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_242, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf62, buf61, arg122_1, arg123_1, arg124_1, arg125_1, 1310720, grid=grid(1310720), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        del buf61
        # Topologically Sorted Source Nodes: [x_243], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 160, 16, 16), (40960, 1, 2560, 160))
        del arg126_1
        buf64 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [x_244, x_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf64, arg127_1, arg128_1, arg129_1, arg130_1, 327680, grid=grid(327680), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        buf65 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_244, x_245, x_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg131_1, buf65, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg131_1
        # Topologically Sorted Source Nodes: [x_244, x_245, x_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf66 = extern_kernels.convolution(buf64, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 160, 16, 16), (40960, 1, 2560, 160))
        del buf64
        del buf65
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_247, x_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf67, arg132_1, arg133_1, arg134_1, arg135_1, 327680, grid=grid(327680), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        # Topologically Sorted Source Nodes: [x_247, x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 640, 16, 16), (163840, 1, 10240, 640))
        del arg136_1
        buf69 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_250, x_251, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf69, buf68, arg137_1, arg138_1, arg139_1, arg140_1, 1310720, grid=grid(1310720), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        del buf68
        # Topologically Sorted Source Nodes: [x_252], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 1920, 16, 16), (491520, 1, 30720, 1920))
        del arg141_1
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf71, arg142_1, arg143_1, arg144_1, arg145_1, 3932160, grid=grid(3932160), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        # Topologically Sorted Source Nodes: [x_253, x_254, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg146_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf72, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg146_1
        del buf71
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf73, arg147_1, arg148_1, arg149_1, arg150_1, 983040, grid=grid(983040), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        # Topologically Sorted Source Nodes: [x_256, x_257, x_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 640, 8, 8), (40960, 1, 5120, 640))
        del arg151_1
        del buf73
        # Topologically Sorted Source Nodes: [x_260], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf69, arg156_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 640, 8, 8), (40960, 1, 5120, 640))
        del arg156_1
        del buf69
        buf76 = buf74; del buf74  # reuse
        buf77 = reinterpret_tensor(buf67, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_259, x_261, x_262, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf76, arg152_1, arg153_1, arg154_1, arg155_1, buf75, arg157_1, arg158_1, arg159_1, arg160_1, buf77, 327680, grid=grid(327680), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        del buf75
        del buf76
        # Topologically Sorted Source Nodes: [input_28, x_263], Original ATen: [aten.relu, aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg161_1
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_264, x_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf79, arg162_1, arg163_1, arg164_1, arg165_1, 983040, grid=grid(983040), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        # Topologically Sorted Source Nodes: [x_264, x_265, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg166_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf80, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg166_1
        del buf79
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_267, x_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf81, arg167_1, arg168_1, arg169_1, arg170_1, 983040, grid=grid(983040), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        # Topologically Sorted Source Nodes: [x_267, x_268, x_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 640, 8, 8), (40960, 1, 5120, 640))
        del arg171_1
        del buf81
        buf83 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [x_270, x_271, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf83, buf82, arg172_1, arg173_1, arg174_1, arg175_1, 327680, grid=grid(327680), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        del buf82
        # Topologically Sorted Source Nodes: [x_272], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg176_1
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_273, x_274], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf85, arg177_1, arg178_1, arg179_1, arg180_1, 983040, grid=grid(983040), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        # Topologically Sorted Source Nodes: [x_273, x_274, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg181_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf86, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg181_1
        del buf85
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_276, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf87, arg182_1, arg183_1, arg184_1, arg185_1, 983040, grid=grid(983040), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        # Topologically Sorted Source Nodes: [x_276, x_277, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 640, 8, 8), (40960, 1, 5120, 640))
        del arg186_1
        del buf87
        buf89 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_279, x_280, input_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf89, buf88, arg187_1, arg188_1, arg189_1, arg190_1, 327680, grid=grid(327680), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        del buf88
        # Topologically Sorted Source Nodes: [x_281], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg191_1
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_282, x_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf91, arg192_1, arg193_1, arg194_1, arg195_1, 983040, grid=grid(983040), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        # Topologically Sorted Source Nodes: [x_282, x_283, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg196_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf92, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg196_1
        del buf91
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_285, x_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf93, arg197_1, arg198_1, arg199_1, arg200_1, 983040, grid=grid(983040), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        # Topologically Sorted Source Nodes: [x_285, x_286, x_287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 640, 8, 8), (40960, 1, 5120, 640))
        del arg201_1
        del buf93
        buf95 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [x_288, x_289, input_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf95, buf94, arg202_1, arg203_1, arg204_1, arg205_1, 327680, grid=grid(327680), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        del buf94
        # Topologically Sorted Source Nodes: [x_290], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg206_1
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_291, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf97, arg207_1, arg208_1, arg209_1, arg210_1, 983040, grid=grid(983040), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        # Topologically Sorted Source Nodes: [x_291, x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf98 = extern_kernels.convolution(buf97, arg211_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf98, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg211_1
        del buf97
        buf99 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [x_294, x_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf99, arg212_1, arg213_1, arg214_1, arg215_1, 983040, grid=grid(983040), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        # Topologically Sorted Source Nodes: [x_294, x_295, x_296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 640, 8, 8), (40960, 1, 5120, 640))
        del arg216_1
        del buf99
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_297, x_298, input_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21.run(buf101, arg217_1, arg218_1, arg219_1, arg220_1, buf95, 327680, grid=grid(327680), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        del buf95
        # Topologically Sorted Source Nodes: [x_299], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg221_1
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_300, x_301], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf103, arg222_1, arg223_1, arg224_1, arg225_1, 983040, grid=grid(983040), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        # Topologically Sorted Source Nodes: [x_300, x_301, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg226_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf104, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg226_1
        del buf103
        buf105 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_303, x_304], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf105, arg227_1, arg228_1, arg229_1, arg230_1, 983040, grid=grid(983040), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        # Topologically Sorted Source Nodes: [x_303, x_304, x_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg231_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 640, 8, 8), (40960, 1, 5120, 640))
        del arg231_1
        del buf105
        buf107 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_306, x_307, input_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf107, buf106, arg232_1, arg233_1, arg234_1, arg235_1, 327680, grid=grid(327680), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        del buf106
        # Topologically Sorted Source Nodes: [x_308], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg236_1
        buf109 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_309, x_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf109, arg237_1, arg238_1, arg239_1, arg240_1, 983040, grid=grid(983040), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        # Topologically Sorted Source Nodes: [x_309, x_310, x_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg241_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf110, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg241_1
        del buf109
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_312, x_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf111, arg242_1, arg243_1, arg244_1, arg245_1, 983040, grid=grid(983040), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        # Topologically Sorted Source Nodes: [x_312, x_313, x_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 640, 8, 8), (40960, 1, 5120, 640))
        del arg246_1
        del buf111
        buf113 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [x_315, x_316, input_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf113, buf112, arg247_1, arg248_1, arg249_1, arg250_1, 327680, grid=grid(327680), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        del buf112
        # Topologically Sorted Source Nodes: [x_317], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg251_1
        buf115 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [x_318, x_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf115, arg252_1, arg253_1, arg254_1, arg255_1, 983040, grid=grid(983040), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        # Topologically Sorted Source Nodes: [x_318, x_319, x_320], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf116 = extern_kernels.convolution(buf115, arg256_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf116, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg256_1
        del buf115
        buf117 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_321, x_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf117, arg257_1, arg258_1, arg259_1, arg260_1, 983040, grid=grid(983040), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        # Topologically Sorted Source Nodes: [x_321, x_322, x_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg261_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 640, 8, 8), (40960, 1, 5120, 640))
        del arg261_1
        del buf117
        buf119 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [x_324, x_325, input_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf119, buf118, arg262_1, arg263_1, arg264_1, arg265_1, 327680, grid=grid(327680), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        del buf118
        # Topologically Sorted Source Nodes: [x_326], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg266_1
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [x_327, x_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf121, arg267_1, arg268_1, arg269_1, arg270_1, 983040, grid=grid(983040), stream=stream0)
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        # Topologically Sorted Source Nodes: [x_327, x_328, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf122 = extern_kernels.convolution(buf121, arg271_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
        assert_size_stride(buf122, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
        del arg271_1
        del buf121
        buf123 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_330, x_331], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf123, arg272_1, arg273_1, arg274_1, arg275_1, 983040, grid=grid(983040), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        # Topologically Sorted Source Nodes: [x_330, x_331, x_332], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 640, 8, 8), (40960, 1, 5120, 640))
        del arg276_1
        del buf123
        buf125 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [x_333, x_334, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf125, buf124, arg277_1, arg278_1, arg279_1, arg280_1, 327680, grid=grid(327680), stream=stream0)
        del arg277_1
        del arg278_1
        del arg279_1
        del arg280_1
        del buf124
        # Topologically Sorted Source Nodes: [x_333, x_334, input_36, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.convolution]
        buf126 = extern_kernels.convolution(buf125, arg281_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 2560, 8, 8), (163840, 1, 20480, 2560))
        del arg281_1
        del buf125
        buf128 = empty_strided_cuda((8, 2560, 1, 1), (2560, 1, 20480, 20480), torch.float32)
        # Topologically Sorted Source Nodes: [x_336, x_337, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_22.run(buf126, arg282_1, arg283_1, arg284_1, arg285_1, buf128, 20480, 64, grid=grid(20480), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        del buf126
        buf129 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_341], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg287_1, reinterpret_tensor(buf128, (8, 2560), (2560, 1), 0), reinterpret_tensor(arg286_1, (2560, 1000), (1, 2560), 0), alpha=1, beta=1, out=buf129)
        del arg286_1
        del arg287_1
        del buf128
    return (buf129, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((192, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((160, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((640, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((640, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((2560, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1000, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gernet_l', benchmark_compiled_module)
