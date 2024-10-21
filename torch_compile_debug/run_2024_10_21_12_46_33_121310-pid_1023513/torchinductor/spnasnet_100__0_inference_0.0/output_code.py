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
# Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_188 => convolution_64
# Graph fragment:
#   %convolution_64 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
# Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_188 => convolution_64
# Graph fragment:
#   %convolution_64 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
# Topologically Sorted Source Nodes: [x_189, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_189 => add_143, mul_193, mul_194, sub_64
#   x_190 => relu_43
# Graph fragment:
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_64, %unsqueeze_513), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_515), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_193, %unsqueeze_517), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_194, %unsqueeze_519), kwargs = {})
#   %relu_43 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_143,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/u3/cu3xxa5afujbnaiyq35kigtl3nrohqj654bzcy3eqmqkshn5viid.py
# Topologically Sorted Source Nodes: [x_195], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_195 => add_147, mul_199, mul_200, sub_66
# Graph fragment:
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_66, %unsqueeze_529), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %unsqueeze_533), kwargs = {})
#   %add_147 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %unsqueeze_535), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 16
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/up/cupv6d6ruakeugiqdv3oewcq7aogdnkczopywsywl2dnce5wqdbl.py
# Topologically Sorted Source Nodes: [x_197, x_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_197 => add_149, mul_202, mul_203, sub_67
#   x_198 => relu_45
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_67, %unsqueeze_537), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %unsqueeze_541), kwargs = {})
#   %add_149 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %unsqueeze_543), kwargs = {})
#   %relu_45 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_149,), kwargs = {})
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
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 48
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


# kernel path: /tmp/torchinductor_sahanp/sz/cszb6csgnj6hugqiib3qbudacgyhkbd2ztyvlya73ma2hkt2jz7p.py
# Topologically Sorted Source Nodes: [x_200, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_200 => add_151, mul_205, mul_206, sub_68
#   x_201 => relu_46
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_545), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_205, %unsqueeze_549), kwargs = {})
#   %add_151 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_206, %unsqueeze_551), kwargs = {})
#   %relu_46 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_151,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 48
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


# kernel path: /tmp/torchinductor_sahanp/pe/cpeeb57i23ji6kcxdbrcfll4ge3fxlyru4oze74dlfhunlx4m5cs.py
# Topologically Sorted Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_203 => add_153, mul_208, mul_209, sub_69
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_69, %unsqueeze_553), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_208, %unsqueeze_557), kwargs = {})
#   %add_153 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_209, %unsqueeze_559), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/u7/cu7b2dxujktiu7yypbzbz2inmpbns4nwqkth6gbk6j5h3kcvrxhz.py
# Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_205 => add_155, mul_211, mul_212, sub_70
#   x_206 => relu_47
# Graph fragment:
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_70, %unsqueeze_561), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_563), kwargs = {})
#   %mul_212 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_211, %unsqueeze_565), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_212, %unsqueeze_567), kwargs = {})
#   %relu_47 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_155,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 72
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


# kernel path: /tmp/torchinductor_sahanp/jc/cjclfqm4qeqbngjhojlq7sx4edkljmviuq33xlxcbzgygdzenwx2.py
# Topologically Sorted Source Nodes: [x_211, x_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_211 => add_159, mul_217, mul_218, sub_72
#   x_212 => add_160
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_72, %unsqueeze_577), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_217, %unsqueeze_581), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_218, %unsqueeze_583), kwargs = {})
#   %add_160 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_159, %add_153), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 24
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cj/ccjqvh3aqcjstyrpkfoc22vfu2q3qmyvatzy75hvkyhluocdghcg.py
# Topologically Sorted Source Nodes: [x_223, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_223 => add_169, mul_229, mul_230, sub_76
#   x_224 => relu_51
# Graph fragment:
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_76, %unsqueeze_609), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %unsqueeze_611), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_229, %unsqueeze_613), kwargs = {})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_230, %unsqueeze_615), kwargs = {})
#   %relu_51 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_169,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 144
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


# kernel path: /tmp/torchinductor_sahanp/bd/cbd4eokrlyjjcmgho2o5hyonhe7d2mmutixpyyq4k5hf5ew2iowf.py
# Topologically Sorted Source Nodes: [x_226, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_226 => add_171, mul_232, mul_233, sub_77
#   x_227 => relu_52
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_617), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_232, %unsqueeze_621), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_233, %unsqueeze_623), kwargs = {})
#   %relu_52 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_171,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
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


# kernel path: /tmp/torchinductor_sahanp/ow/cow6wradpnk6vliyzgwkbt2o7j2dg2s5ohlmthlf4t2sucippxpk.py
# Topologically Sorted Source Nodes: [x_229], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_229 => add_173, mul_235, mul_236, sub_78
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_78, %unsqueeze_625), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_235, %unsqueeze_629), kwargs = {})
#   %add_173 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_236, %unsqueeze_631), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 40
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/q7/cq7yugcf32qcgm45ytghrubt7fewzk6qwifyr4x3ezfknaykscsv.py
# Topologically Sorted Source Nodes: [x_231, x_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_231 => add_175, mul_238, mul_239, sub_79
#   x_232 => relu_53
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_79, %unsqueeze_633), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_635), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_238, %unsqueeze_637), kwargs = {})
#   %add_175 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_239, %unsqueeze_639), kwargs = {})
#   %relu_53 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_175,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
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


# kernel path: /tmp/torchinductor_sahanp/6q/c6qvuk2tsgekx5zmwiq25k37lpacwiws43txo3ut5r7drasxchyw.py
# Topologically Sorted Source Nodes: [x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_237 => add_179, mul_244, mul_245, sub_81
#   x_238 => add_180
# Graph fragment:
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_81, %unsqueeze_649), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_651), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_244, %unsqueeze_653), kwargs = {})
#   %add_179 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_245, %unsqueeze_655), kwargs = {})
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_179, %add_173), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 40
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/z4/cz42czb2n4plee2hnuckjgb3mfathx5u4cahjb6c7ra5nijxqlcg.py
# Topologically Sorted Source Nodes: [x_258, x_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_258 => add_196, mul_265, mul_266, sub_88
#   x_259 => relu_59
# Graph fragment:
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_88, %unsqueeze_705), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_707), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_265, %unsqueeze_709), kwargs = {})
#   %add_196 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_266, %unsqueeze_711), kwargs = {})
#   %relu_59 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_196,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
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


# kernel path: /tmp/torchinductor_sahanp/ga/cgadmnfmk7vmaqfieymebakiu2ckpfuqetvfsnfkmietsnwcmzce.py
# Topologically Sorted Source Nodes: [x_261, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_261 => add_198, mul_268, mul_269, sub_89
#   x_262 => relu_60
# Graph fragment:
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_89, %unsqueeze_713), kwargs = {})
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %unsqueeze_715), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_268, %unsqueeze_717), kwargs = {})
#   %add_198 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_269, %unsqueeze_719), kwargs = {})
#   %relu_60 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_198,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
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


# kernel path: /tmp/torchinductor_sahanp/js/cjsyglm3bedac3etpo4atgq2iqa4jplvs4rt3vrsj7qcfsmoa4v5.py
# Topologically Sorted Source Nodes: [x_264], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_264 => add_200, mul_271, mul_272, sub_90
# Graph fragment:
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_90, %unsqueeze_721), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_723), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %unsqueeze_725), kwargs = {})
#   %add_200 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_272, %unsqueeze_727), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bb/cbbpg2o4ryoohiyo4a4oakwjigd3cadtb6rp3uzgpv3ceeh74ygu.py
# Topologically Sorted Source Nodes: [x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_272 => add_206, mul_280, mul_281, sub_93
#   x_273 => add_207
# Graph fragment:
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_93, %unsqueeze_745), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_747), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_280, %unsqueeze_749), kwargs = {})
#   %add_206 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_281, %unsqueeze_751), kwargs = {})
#   %add_207 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_206, %add_200), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/m7/cm7bgtfjxiq7oknykdmpvqxwfiuwf5iaefo5zreqljevwnxjrflw.py
# Topologically Sorted Source Nodes: [x_293, x_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_293 => add_223, mul_301, mul_302, sub_100
#   x_294 => relu_67
# Graph fragment:
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_801), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_803), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_301, %unsqueeze_805), kwargs = {})
#   %add_223 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_302, %unsqueeze_807), kwargs = {})
#   %relu_67 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_223,), kwargs = {})
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
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
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


# kernel path: /tmp/torchinductor_sahanp/zz/czzmegjqtz4tsjfv4u52hyccosidrk6lk2jrvtdnsd27eqkcnazw.py
# Topologically Sorted Source Nodes: [x_299], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_299 => add_227, mul_307, mul_308, sub_102
# Graph fragment:
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_102, %unsqueeze_817), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %unsqueeze_819), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_307, %unsqueeze_821), kwargs = {})
#   %add_227 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_308, %unsqueeze_823), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kb/ckb2nhjlqt5xhk7xbuy7iek6s23h7eix66qseb2fhvn3z3dprz7p.py
# Topologically Sorted Source Nodes: [x_301, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_301 => add_229, mul_310, mul_311, sub_103
#   x_302 => relu_69
# Graph fragment:
#   %sub_103 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_103, %unsqueeze_825), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_103, %unsqueeze_827), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, %unsqueeze_829), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_311, %unsqueeze_831), kwargs = {})
#   %relu_69 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_229,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 288
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


# kernel path: /tmp/torchinductor_sahanp/wv/cwvt3dacxm4kdzm53tr6icpx3unyagsg3wg3sbz7fbua56u7bgtn.py
# Topologically Sorted Source Nodes: [x_307, x_308], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_307 => add_233, mul_316, mul_317, sub_105
#   x_308 => add_234
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_841), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_316, %unsqueeze_845), kwargs = {})
#   %add_233 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_317, %unsqueeze_847), kwargs = {})
#   %add_234 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_233, %add_227), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/v7/cv7segpnobkvwboo23cqcb3oshpjhlfw5svf3naa7nvp5rq3ow6i.py
# Topologically Sorted Source Nodes: [x_328, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_328 => add_250, mul_337, mul_338, sub_112
#   x_329 => relu_75
# Graph fragment:
#   %sub_112 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_112, %unsqueeze_897), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_112, %unsqueeze_899), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_337, %unsqueeze_901), kwargs = {})
#   %add_250 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_338, %unsqueeze_903), kwargs = {})
#   %relu_75 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_250,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 576
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


# kernel path: /tmp/torchinductor_sahanp/bf/cbfeig3becll7cax3gcb2v63mrabqjnjegqtyzftvde3wxilcmvk.py
# Topologically Sorted Source Nodes: [x_331, x_332], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_331 => add_252, mul_340, mul_341, sub_113
#   x_332 => relu_76
# Graph fragment:
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_113, %unsqueeze_905), kwargs = {})
#   %mul_340 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %unsqueeze_907), kwargs = {})
#   %mul_341 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_340, %unsqueeze_909), kwargs = {})
#   %add_252 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_341, %unsqueeze_911), kwargs = {})
#   %relu_76 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_252,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 576
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


# kernel path: /tmp/torchinductor_sahanp/yo/cyocq3bseqczqw2xb3luza7dcrdbwygzqelbiymz5f5pqxhnky3m.py
# Topologically Sorted Source Nodes: [x_334], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_334 => add_254, mul_343, mul_344, sub_114
# Graph fragment:
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_114, %unsqueeze_913), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_915), kwargs = {})
#   %mul_344 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_343, %unsqueeze_917), kwargs = {})
#   %add_254 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_344, %unsqueeze_919), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/af/cafczx76rq2m4t4322yxgywbgdwezuklvtzk2o7eadzt7su3mp7q.py
# Topologically Sorted Source Nodes: [x_336, x_337], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_336 => add_256, mul_346, mul_347, sub_115
#   x_337 => relu_77
# Graph fragment:
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_921), kwargs = {})
#   %mul_346 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %unsqueeze_923), kwargs = {})
#   %mul_347 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_346, %unsqueeze_925), kwargs = {})
#   %add_256 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_347, %unsqueeze_927), kwargs = {})
#   %relu_77 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_256,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
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


# kernel path: /tmp/torchinductor_sahanp/ut/cutwezpl7rinlavdwrm7lrzb33v6srr6nawp672r5hwnu4o454mf.py
# Topologically Sorted Source Nodes: [x_342, x_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_342 => add_260, mul_352, mul_353, sub_117
#   x_343 => add_261
# Graph fragment:
#   %sub_117 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_117, %unsqueeze_937), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_117, %unsqueeze_939), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, %unsqueeze_941), kwargs = {})
#   %add_260 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_353, %unsqueeze_943), kwargs = {})
#   %add_261 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_260, %add_254), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4k/c4kvnw35hez7pnsquhj6aig67lium4ngdl2kfqhmnslu6rm266t5.py
# Topologically Sorted Source Nodes: [x_369], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_369 => add_281, mul_379, mul_380, sub_126
# Graph fragment:
#   %sub_126 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_126, %unsqueeze_1009), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_126, %unsqueeze_1011), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_379, %unsqueeze_1013), kwargs = {})
#   %add_281 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_380, %unsqueeze_1015), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xp/cxpnghfr2ppnyvxhlsqy2s2qrubpiy5upmwgtvmzq3obdemztnfb.py
# Topologically Sorted Source Nodes: [x_371, x_372, x_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   x_371 => add_283, mul_382, mul_383, sub_127
#   x_372 => relu_85
#   x_373 => mean_1
# Graph fragment:
#   %sub_127 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_127, %unsqueeze_1017), kwargs = {})
#   %mul_382 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_127, %unsqueeze_1019), kwargs = {})
#   %mul_383 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_382, %unsqueeze_1021), kwargs = {})
#   %add_283 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_383, %unsqueeze_1023), kwargs = {})
#   %relu_85 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_283,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_85, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (62720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 49.0
    tmp23 = tmp21 / tmp22
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (32, ), (1, ))
    assert_size_stride(arg9_1, (32, ), (1, ))
    assert_size_stride(arg10_1, (32, ), (1, ))
    assert_size_stride(arg11_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg12_1, (16, ), (1, ))
    assert_size_stride(arg13_1, (16, ), (1, ))
    assert_size_stride(arg14_1, (16, ), (1, ))
    assert_size_stride(arg15_1, (16, ), (1, ))
    assert_size_stride(arg16_1, (48, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg17_1, (48, ), (1, ))
    assert_size_stride(arg18_1, (48, ), (1, ))
    assert_size_stride(arg19_1, (48, ), (1, ))
    assert_size_stride(arg20_1, (48, ), (1, ))
    assert_size_stride(arg21_1, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg22_1, (48, ), (1, ))
    assert_size_stride(arg23_1, (48, ), (1, ))
    assert_size_stride(arg24_1, (48, ), (1, ))
    assert_size_stride(arg25_1, (48, ), (1, ))
    assert_size_stride(arg26_1, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg27_1, (24, ), (1, ))
    assert_size_stride(arg28_1, (24, ), (1, ))
    assert_size_stride(arg29_1, (24, ), (1, ))
    assert_size_stride(arg30_1, (24, ), (1, ))
    assert_size_stride(arg31_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg32_1, (72, ), (1, ))
    assert_size_stride(arg33_1, (72, ), (1, ))
    assert_size_stride(arg34_1, (72, ), (1, ))
    assert_size_stride(arg35_1, (72, ), (1, ))
    assert_size_stride(arg36_1, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg37_1, (72, ), (1, ))
    assert_size_stride(arg38_1, (72, ), (1, ))
    assert_size_stride(arg39_1, (72, ), (1, ))
    assert_size_stride(arg40_1, (72, ), (1, ))
    assert_size_stride(arg41_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg42_1, (24, ), (1, ))
    assert_size_stride(arg43_1, (24, ), (1, ))
    assert_size_stride(arg44_1, (24, ), (1, ))
    assert_size_stride(arg45_1, (24, ), (1, ))
    assert_size_stride(arg46_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg47_1, (72, ), (1, ))
    assert_size_stride(arg48_1, (72, ), (1, ))
    assert_size_stride(arg49_1, (72, ), (1, ))
    assert_size_stride(arg50_1, (72, ), (1, ))
    assert_size_stride(arg51_1, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg52_1, (72, ), (1, ))
    assert_size_stride(arg53_1, (72, ), (1, ))
    assert_size_stride(arg54_1, (72, ), (1, ))
    assert_size_stride(arg55_1, (72, ), (1, ))
    assert_size_stride(arg56_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg57_1, (24, ), (1, ))
    assert_size_stride(arg58_1, (24, ), (1, ))
    assert_size_stride(arg59_1, (24, ), (1, ))
    assert_size_stride(arg60_1, (24, ), (1, ))
    assert_size_stride(arg61_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg62_1, (144, ), (1, ))
    assert_size_stride(arg63_1, (144, ), (1, ))
    assert_size_stride(arg64_1, (144, ), (1, ))
    assert_size_stride(arg65_1, (144, ), (1, ))
    assert_size_stride(arg66_1, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg67_1, (144, ), (1, ))
    assert_size_stride(arg68_1, (144, ), (1, ))
    assert_size_stride(arg69_1, (144, ), (1, ))
    assert_size_stride(arg70_1, (144, ), (1, ))
    assert_size_stride(arg71_1, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg72_1, (40, ), (1, ))
    assert_size_stride(arg73_1, (40, ), (1, ))
    assert_size_stride(arg74_1, (40, ), (1, ))
    assert_size_stride(arg75_1, (40, ), (1, ))
    assert_size_stride(arg76_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg77_1, (120, ), (1, ))
    assert_size_stride(arg78_1, (120, ), (1, ))
    assert_size_stride(arg79_1, (120, ), (1, ))
    assert_size_stride(arg80_1, (120, ), (1, ))
    assert_size_stride(arg81_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg82_1, (120, ), (1, ))
    assert_size_stride(arg83_1, (120, ), (1, ))
    assert_size_stride(arg84_1, (120, ), (1, ))
    assert_size_stride(arg85_1, (120, ), (1, ))
    assert_size_stride(arg86_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg87_1, (40, ), (1, ))
    assert_size_stride(arg88_1, (40, ), (1, ))
    assert_size_stride(arg89_1, (40, ), (1, ))
    assert_size_stride(arg90_1, (40, ), (1, ))
    assert_size_stride(arg91_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg92_1, (120, ), (1, ))
    assert_size_stride(arg93_1, (120, ), (1, ))
    assert_size_stride(arg94_1, (120, ), (1, ))
    assert_size_stride(arg95_1, (120, ), (1, ))
    assert_size_stride(arg96_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg97_1, (120, ), (1, ))
    assert_size_stride(arg98_1, (120, ), (1, ))
    assert_size_stride(arg99_1, (120, ), (1, ))
    assert_size_stride(arg100_1, (120, ), (1, ))
    assert_size_stride(arg101_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg102_1, (40, ), (1, ))
    assert_size_stride(arg103_1, (40, ), (1, ))
    assert_size_stride(arg104_1, (40, ), (1, ))
    assert_size_stride(arg105_1, (40, ), (1, ))
    assert_size_stride(arg106_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg107_1, (120, ), (1, ))
    assert_size_stride(arg108_1, (120, ), (1, ))
    assert_size_stride(arg109_1, (120, ), (1, ))
    assert_size_stride(arg110_1, (120, ), (1, ))
    assert_size_stride(arg111_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg112_1, (120, ), (1, ))
    assert_size_stride(arg113_1, (120, ), (1, ))
    assert_size_stride(arg114_1, (120, ), (1, ))
    assert_size_stride(arg115_1, (120, ), (1, ))
    assert_size_stride(arg116_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg117_1, (40, ), (1, ))
    assert_size_stride(arg118_1, (40, ), (1, ))
    assert_size_stride(arg119_1, (40, ), (1, ))
    assert_size_stride(arg120_1, (40, ), (1, ))
    assert_size_stride(arg121_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg122_1, (240, ), (1, ))
    assert_size_stride(arg123_1, (240, ), (1, ))
    assert_size_stride(arg124_1, (240, ), (1, ))
    assert_size_stride(arg125_1, (240, ), (1, ))
    assert_size_stride(arg126_1, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg127_1, (240, ), (1, ))
    assert_size_stride(arg128_1, (240, ), (1, ))
    assert_size_stride(arg129_1, (240, ), (1, ))
    assert_size_stride(arg130_1, (240, ), (1, ))
    assert_size_stride(arg131_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg132_1, (80, ), (1, ))
    assert_size_stride(arg133_1, (80, ), (1, ))
    assert_size_stride(arg134_1, (80, ), (1, ))
    assert_size_stride(arg135_1, (80, ), (1, ))
    assert_size_stride(arg136_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg137_1, (240, ), (1, ))
    assert_size_stride(arg138_1, (240, ), (1, ))
    assert_size_stride(arg139_1, (240, ), (1, ))
    assert_size_stride(arg140_1, (240, ), (1, ))
    assert_size_stride(arg141_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg142_1, (240, ), (1, ))
    assert_size_stride(arg143_1, (240, ), (1, ))
    assert_size_stride(arg144_1, (240, ), (1, ))
    assert_size_stride(arg145_1, (240, ), (1, ))
    assert_size_stride(arg146_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg147_1, (80, ), (1, ))
    assert_size_stride(arg148_1, (80, ), (1, ))
    assert_size_stride(arg149_1, (80, ), (1, ))
    assert_size_stride(arg150_1, (80, ), (1, ))
    assert_size_stride(arg151_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg152_1, (240, ), (1, ))
    assert_size_stride(arg153_1, (240, ), (1, ))
    assert_size_stride(arg154_1, (240, ), (1, ))
    assert_size_stride(arg155_1, (240, ), (1, ))
    assert_size_stride(arg156_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg157_1, (240, ), (1, ))
    assert_size_stride(arg158_1, (240, ), (1, ))
    assert_size_stride(arg159_1, (240, ), (1, ))
    assert_size_stride(arg160_1, (240, ), (1, ))
    assert_size_stride(arg161_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg162_1, (80, ), (1, ))
    assert_size_stride(arg163_1, (80, ), (1, ))
    assert_size_stride(arg164_1, (80, ), (1, ))
    assert_size_stride(arg165_1, (80, ), (1, ))
    assert_size_stride(arg166_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg167_1, (240, ), (1, ))
    assert_size_stride(arg168_1, (240, ), (1, ))
    assert_size_stride(arg169_1, (240, ), (1, ))
    assert_size_stride(arg170_1, (240, ), (1, ))
    assert_size_stride(arg171_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg172_1, (240, ), (1, ))
    assert_size_stride(arg173_1, (240, ), (1, ))
    assert_size_stride(arg174_1, (240, ), (1, ))
    assert_size_stride(arg175_1, (240, ), (1, ))
    assert_size_stride(arg176_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg177_1, (80, ), (1, ))
    assert_size_stride(arg178_1, (80, ), (1, ))
    assert_size_stride(arg179_1, (80, ), (1, ))
    assert_size_stride(arg180_1, (80, ), (1, ))
    assert_size_stride(arg181_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg182_1, (480, ), (1, ))
    assert_size_stride(arg183_1, (480, ), (1, ))
    assert_size_stride(arg184_1, (480, ), (1, ))
    assert_size_stride(arg185_1, (480, ), (1, ))
    assert_size_stride(arg186_1, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg187_1, (480, ), (1, ))
    assert_size_stride(arg188_1, (480, ), (1, ))
    assert_size_stride(arg189_1, (480, ), (1, ))
    assert_size_stride(arg190_1, (480, ), (1, ))
    assert_size_stride(arg191_1, (96, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg192_1, (96, ), (1, ))
    assert_size_stride(arg193_1, (96, ), (1, ))
    assert_size_stride(arg194_1, (96, ), (1, ))
    assert_size_stride(arg195_1, (96, ), (1, ))
    assert_size_stride(arg196_1, (288, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg197_1, (288, ), (1, ))
    assert_size_stride(arg198_1, (288, ), (1, ))
    assert_size_stride(arg199_1, (288, ), (1, ))
    assert_size_stride(arg200_1, (288, ), (1, ))
    assert_size_stride(arg201_1, (288, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg202_1, (288, ), (1, ))
    assert_size_stride(arg203_1, (288, ), (1, ))
    assert_size_stride(arg204_1, (288, ), (1, ))
    assert_size_stride(arg205_1, (288, ), (1, ))
    assert_size_stride(arg206_1, (96, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg207_1, (96, ), (1, ))
    assert_size_stride(arg208_1, (96, ), (1, ))
    assert_size_stride(arg209_1, (96, ), (1, ))
    assert_size_stride(arg210_1, (96, ), (1, ))
    assert_size_stride(arg211_1, (288, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg212_1, (288, ), (1, ))
    assert_size_stride(arg213_1, (288, ), (1, ))
    assert_size_stride(arg214_1, (288, ), (1, ))
    assert_size_stride(arg215_1, (288, ), (1, ))
    assert_size_stride(arg216_1, (288, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg217_1, (288, ), (1, ))
    assert_size_stride(arg218_1, (288, ), (1, ))
    assert_size_stride(arg219_1, (288, ), (1, ))
    assert_size_stride(arg220_1, (288, ), (1, ))
    assert_size_stride(arg221_1, (96, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg222_1, (96, ), (1, ))
    assert_size_stride(arg223_1, (96, ), (1, ))
    assert_size_stride(arg224_1, (96, ), (1, ))
    assert_size_stride(arg225_1, (96, ), (1, ))
    assert_size_stride(arg226_1, (288, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg227_1, (288, ), (1, ))
    assert_size_stride(arg228_1, (288, ), (1, ))
    assert_size_stride(arg229_1, (288, ), (1, ))
    assert_size_stride(arg230_1, (288, ), (1, ))
    assert_size_stride(arg231_1, (288, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg232_1, (288, ), (1, ))
    assert_size_stride(arg233_1, (288, ), (1, ))
    assert_size_stride(arg234_1, (288, ), (1, ))
    assert_size_stride(arg235_1, (288, ), (1, ))
    assert_size_stride(arg236_1, (96, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg237_1, (96, ), (1, ))
    assert_size_stride(arg238_1, (96, ), (1, ))
    assert_size_stride(arg239_1, (96, ), (1, ))
    assert_size_stride(arg240_1, (96, ), (1, ))
    assert_size_stride(arg241_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg242_1, (576, ), (1, ))
    assert_size_stride(arg243_1, (576, ), (1, ))
    assert_size_stride(arg244_1, (576, ), (1, ))
    assert_size_stride(arg245_1, (576, ), (1, ))
    assert_size_stride(arg246_1, (576, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg247_1, (576, ), (1, ))
    assert_size_stride(arg248_1, (576, ), (1, ))
    assert_size_stride(arg249_1, (576, ), (1, ))
    assert_size_stride(arg250_1, (576, ), (1, ))
    assert_size_stride(arg251_1, (192, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg252_1, (192, ), (1, ))
    assert_size_stride(arg253_1, (192, ), (1, ))
    assert_size_stride(arg254_1, (192, ), (1, ))
    assert_size_stride(arg255_1, (192, ), (1, ))
    assert_size_stride(arg256_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg257_1, (1152, ), (1, ))
    assert_size_stride(arg258_1, (1152, ), (1, ))
    assert_size_stride(arg259_1, (1152, ), (1, ))
    assert_size_stride(arg260_1, (1152, ), (1, ))
    assert_size_stride(arg261_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg262_1, (1152, ), (1, ))
    assert_size_stride(arg263_1, (1152, ), (1, ))
    assert_size_stride(arg264_1, (1152, ), (1, ))
    assert_size_stride(arg265_1, (1152, ), (1, ))
    assert_size_stride(arg266_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg267_1, (192, ), (1, ))
    assert_size_stride(arg268_1, (192, ), (1, ))
    assert_size_stride(arg269_1, (192, ), (1, ))
    assert_size_stride(arg270_1, (192, ), (1, ))
    assert_size_stride(arg271_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg272_1, (1152, ), (1, ))
    assert_size_stride(arg273_1, (1152, ), (1, ))
    assert_size_stride(arg274_1, (1152, ), (1, ))
    assert_size_stride(arg275_1, (1152, ), (1, ))
    assert_size_stride(arg276_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg277_1, (1152, ), (1, ))
    assert_size_stride(arg278_1, (1152, ), (1, ))
    assert_size_stride(arg279_1, (1152, ), (1, ))
    assert_size_stride(arg280_1, (1152, ), (1, ))
    assert_size_stride(arg281_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg282_1, (192, ), (1, ))
    assert_size_stride(arg283_1, (192, ), (1, ))
    assert_size_stride(arg284_1, (192, ), (1, ))
    assert_size_stride(arg285_1, (192, ), (1, ))
    assert_size_stride(arg286_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg287_1, (1152, ), (1, ))
    assert_size_stride(arg288_1, (1152, ), (1, ))
    assert_size_stride(arg289_1, (1152, ), (1, ))
    assert_size_stride(arg290_1, (1152, ), (1, ))
    assert_size_stride(arg291_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg292_1, (1152, ), (1, ))
    assert_size_stride(arg293_1, (1152, ), (1, ))
    assert_size_stride(arg294_1, (1152, ), (1, ))
    assert_size_stride(arg295_1, (1152, ), (1, ))
    assert_size_stride(arg296_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg297_1, (192, ), (1, ))
    assert_size_stride(arg298_1, (192, ), (1, ))
    assert_size_stride(arg299_1, (192, ), (1, ))
    assert_size_stride(arg300_1, (192, ), (1, ))
    assert_size_stride(arg301_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg302_1, (1152, ), (1, ))
    assert_size_stride(arg303_1, (1152, ), (1, ))
    assert_size_stride(arg304_1, (1152, ), (1, ))
    assert_size_stride(arg305_1, (1152, ), (1, ))
    assert_size_stride(arg306_1, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg307_1, (1152, ), (1, ))
    assert_size_stride(arg308_1, (1152, ), (1, ))
    assert_size_stride(arg309_1, (1152, ), (1, ))
    assert_size_stride(arg310_1, (1152, ), (1, ))
    assert_size_stride(arg311_1, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg312_1, (320, ), (1, ))
    assert_size_stride(arg313_1, (320, ), (1, ))
    assert_size_stride(arg314_1, (320, ), (1, ))
    assert_size_stride(arg315_1, (320, ), (1, ))
    assert_size_stride(arg316_1, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg317_1, (1280, ), (1, ))
    assert_size_stride(arg318_1, (1280, ), (1, ))
    assert_size_stride(arg319_1, (1280, ), (1, ))
    assert_size_stride(arg320_1, (1280, ), (1, ))
    assert_size_stride(arg321_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg322_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_189, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        # Topologically Sorted Source Nodes: [x_189, x_190, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf4, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del arg6_1
        del buf3
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_192, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf5, arg7_1, arg8_1, arg9_1, arg10_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [x_192, x_193, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del arg11_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_195], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf7, arg12_1, arg13_1, arg14_1, arg15_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        # Topologically Sorted Source Nodes: [x_195, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 48, 112, 112), (602112, 1, 5376, 48))
        del arg16_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_197, x_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf9, arg17_1, arg18_1, arg19_1, arg20_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        # Topologically Sorted Source Nodes: [x_197, x_198, x_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg21_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf10, (8, 48, 56, 56), (150528, 1, 2688, 48))
        del arg21_1
        del buf9
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_200, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf11, arg22_1, arg23_1, arg24_1, arg25_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        # Topologically Sorted Source Nodes: [x_200, x_201, x_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg26_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf13, arg27_1, arg28_1, arg29_1, arg30_1, 602112, grid=grid(602112), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        # Topologically Sorted Source Nodes: [x_204], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 72, 56, 56), (225792, 1, 4032, 72))
        del arg31_1
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf15, arg32_1, arg33_1, arg34_1, arg35_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        # Topologically Sorted Source Nodes: [x_205, x_206, x_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg36_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf16, (8, 72, 56, 56), (225792, 1, 4032, 72))
        del arg36_1
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_208, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf17, arg37_1, arg38_1, arg39_1, arg40_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        # Topologically Sorted Source Nodes: [x_208, x_209, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg41_1
        del buf17
        buf19 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_211, x_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf19, buf18, arg42_1, arg43_1, arg44_1, arg45_1, 602112, grid=grid(602112), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        del buf18
        # Topologically Sorted Source Nodes: [x_213], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 72, 56, 56), (225792, 1, 4032, 72))
        del arg46_1
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_214, x_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf21, arg47_1, arg48_1, arg49_1, arg50_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        # Topologically Sorted Source Nodes: [x_214, x_215, x_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg51_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf22, (8, 72, 56, 56), (225792, 1, 4032, 72))
        del arg51_1
        del buf21
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_217, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf23, arg52_1, arg53_1, arg54_1, arg55_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        # Topologically Sorted Source Nodes: [x_217, x_218, x_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg56_1
        del buf23
        buf25 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_220, x_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf25, buf24, arg57_1, arg58_1, arg59_1, arg60_1, 602112, grid=grid(602112), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf24
        # Topologically Sorted Source Nodes: [x_220, x_221, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 144, 56, 56), (451584, 1, 8064, 144))
        del arg61_1
        del buf25
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_223, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf27, arg62_1, arg63_1, arg64_1, arg65_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        # Topologically Sorted Source Nodes: [x_223, x_224, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg66_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf28, (8, 144, 28, 28), (112896, 1, 4032, 144))
        del arg66_1
        del buf27
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_226, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf29, arg67_1, arg68_1, arg69_1, arg70_1, 903168, grid=grid(903168), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        # Topologically Sorted Source Nodes: [x_226, x_227, x_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 40, 28, 28), (31360, 1, 1120, 40))
        del arg71_1
        del buf29
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_229], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf31, arg72_1, arg73_1, arg74_1, arg75_1, 250880, grid=grid(250880), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        # Topologically Sorted Source Nodes: [x_230], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 120, 28, 28), (94080, 1, 3360, 120))
        del arg76_1
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_231, x_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf33, arg77_1, arg78_1, arg79_1, arg80_1, 752640, grid=grid(752640), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        # Topologically Sorted Source Nodes: [x_231, x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg81_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf34, (8, 120, 28, 28), (94080, 1, 3360, 120))
        del arg81_1
        del buf33
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf35, arg82_1, arg83_1, arg84_1, arg85_1, 752640, grid=grid(752640), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        # Topologically Sorted Source Nodes: [x_234, x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 40, 28, 28), (31360, 1, 1120, 40))
        del arg86_1
        del buf35
        buf37 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf37, buf36, arg87_1, arg88_1, arg89_1, arg90_1, 250880, grid=grid(250880), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        del buf36
        # Topologically Sorted Source Nodes: [x_239], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 120, 28, 28), (94080, 1, 3360, 120))
        del arg91_1
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_240, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf39, arg92_1, arg93_1, arg94_1, arg95_1, 752640, grid=grid(752640), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        # Topologically Sorted Source Nodes: [x_240, x_241, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg96_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf40, (8, 120, 28, 28), (94080, 1, 3360, 120))
        del arg96_1
        del buf39
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_243, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf41, arg97_1, arg98_1, arg99_1, arg100_1, 752640, grid=grid(752640), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        # Topologically Sorted Source Nodes: [x_243, x_244, x_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 40, 28, 28), (31360, 1, 1120, 40))
        del arg101_1
        del buf41
        buf43 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_246, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf43, buf42, arg102_1, arg103_1, arg104_1, arg105_1, 250880, grid=grid(250880), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        del buf42
        # Topologically Sorted Source Nodes: [x_248], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 120, 28, 28), (94080, 1, 3360, 120))
        del arg106_1
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [x_249, x_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf45, arg107_1, arg108_1, arg109_1, arg110_1, 752640, grid=grid(752640), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        # Topologically Sorted Source Nodes: [x_249, x_250, x_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf46 = extern_kernels.convolution(buf45, arg111_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf46, (8, 120, 28, 28), (94080, 1, 3360, 120))
        del arg111_1
        del buf45
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_252, x_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf47, arg112_1, arg113_1, arg114_1, arg115_1, 752640, grid=grid(752640), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        # Topologically Sorted Source Nodes: [x_252, x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 40, 28, 28), (31360, 1, 1120, 40))
        del arg116_1
        del buf47
        buf49 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_255, x_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf49, buf48, arg117_1, arg118_1, arg119_1, arg120_1, 250880, grid=grid(250880), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        del buf48
        # Topologically Sorted Source Nodes: [x_255, x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 240, 28, 28), (188160, 1, 6720, 240))
        del arg121_1
        del buf49
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_258, x_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf51, arg122_1, arg123_1, arg124_1, arg125_1, 1505280, grid=grid(1505280), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        # Topologically Sorted Source Nodes: [x_258, x_259, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg126_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf52, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg126_1
        del buf51
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_261, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf53, arg127_1, arg128_1, arg129_1, arg130_1, 376320, grid=grid(376320), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        # Topologically Sorted Source Nodes: [x_261, x_262, x_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg131_1
        del buf53
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_264], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf55, arg132_1, arg133_1, arg134_1, arg135_1, 125440, grid=grid(125440), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        # Topologically Sorted Source Nodes: [x_265], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg136_1
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_266, x_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf57, arg137_1, arg138_1, arg139_1, arg140_1, 376320, grid=grid(376320), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        # Topologically Sorted Source Nodes: [x_266, x_267, x_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf58, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg141_1
        del buf57
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_269, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf59, arg142_1, arg143_1, arg144_1, arg145_1, 376320, grid=grid(376320), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        # Topologically Sorted Source Nodes: [x_269, x_270, x_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg146_1
        del buf59
        buf61 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf61, buf60, arg147_1, arg148_1, arg149_1, arg150_1, 125440, grid=grid(125440), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        del buf60
        # Topologically Sorted Source Nodes: [x_274], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg151_1
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_275, x_276], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf63, arg152_1, arg153_1, arg154_1, arg155_1, 376320, grid=grid(376320), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        # Topologically Sorted Source Nodes: [x_275, x_276, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg156_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf64, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg156_1
        del buf63
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_278, x_279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf65, arg157_1, arg158_1, arg159_1, arg160_1, 376320, grid=grid(376320), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        # Topologically Sorted Source Nodes: [x_278, x_279, x_280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg161_1
        del buf65
        buf67 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_281, x_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf67, buf66, arg162_1, arg163_1, arg164_1, arg165_1, 125440, grid=grid(125440), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        del buf66
        # Topologically Sorted Source Nodes: [x_283], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg166_1
        buf69 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_284, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf69, arg167_1, arg168_1, arg169_1, arg170_1, 376320, grid=grid(376320), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        # Topologically Sorted Source Nodes: [x_284, x_285, x_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg171_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf70, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg171_1
        del buf69
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_287, x_288], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf71, arg172_1, arg173_1, arg174_1, arg175_1, 376320, grid=grid(376320), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        # Topologically Sorted Source Nodes: [x_287, x_288, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg176_1
        del buf71
        buf73 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_290, x_291], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf73, buf72, arg177_1, arg178_1, arg179_1, arg180_1, 125440, grid=grid(125440), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        del buf72
        # Topologically Sorted Source Nodes: [x_290, x_291, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 480, 14, 14), (94080, 1, 6720, 480))
        del arg181_1
        del buf73
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_293, x_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf75, arg182_1, arg183_1, arg184_1, arg185_1, 752640, grid=grid(752640), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        # Topologically Sorted Source Nodes: [x_293, x_294, x_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg186_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf76, (8, 480, 14, 14), (94080, 1, 6720, 480))
        del arg186_1
        del buf75
        buf77 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_296, x_297], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf77, arg187_1, arg188_1, arg189_1, arg190_1, 752640, grid=grid(752640), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        # Topologically Sorted Source Nodes: [x_296, x_297, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 96, 14, 14), (18816, 1, 1344, 96))
        del arg191_1
        del buf77
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_299], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf79, arg192_1, arg193_1, arg194_1, arg195_1, 150528, grid=grid(150528), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        # Topologically Sorted Source Nodes: [x_300], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 288, 14, 14), (56448, 1, 4032, 288))
        del arg196_1
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_301, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf81, arg197_1, arg198_1, arg199_1, arg200_1, 451584, grid=grid(451584), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        # Topologically Sorted Source Nodes: [x_301, x_302, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg201_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=288, bias=None)
        assert_size_stride(buf82, (8, 288, 14, 14), (56448, 1, 4032, 288))
        del arg201_1
        del buf81
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [x_304, x_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf83, arg202_1, arg203_1, arg204_1, arg205_1, 451584, grid=grid(451584), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        # Topologically Sorted Source Nodes: [x_304, x_305, x_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf84 = extern_kernels.convolution(buf83, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 96, 14, 14), (18816, 1, 1344, 96))
        del arg206_1
        del buf83
        buf85 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [x_307, x_308], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf85, buf84, arg207_1, arg208_1, arg209_1, arg210_1, 150528, grid=grid(150528), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        del buf84
        # Topologically Sorted Source Nodes: [x_309], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 288, 14, 14), (56448, 1, 4032, 288))
        del arg211_1
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_310, x_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf87, arg212_1, arg213_1, arg214_1, arg215_1, 451584, grid=grid(451584), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        # Topologically Sorted Source Nodes: [x_310, x_311, x_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg216_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=288, bias=None)
        assert_size_stride(buf88, (8, 288, 14, 14), (56448, 1, 4032, 288))
        del arg216_1
        del buf87
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_313, x_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf89, arg217_1, arg218_1, arg219_1, arg220_1, 451584, grid=grid(451584), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        # Topologically Sorted Source Nodes: [x_313, x_314, x_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 96, 14, 14), (18816, 1, 1344, 96))
        del arg221_1
        del buf89
        buf91 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_316, x_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf91, buf90, arg222_1, arg223_1, arg224_1, arg225_1, 150528, grid=grid(150528), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        del buf90
        # Topologically Sorted Source Nodes: [x_318], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 288, 14, 14), (56448, 1, 4032, 288))
        del arg226_1
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_319, x_320], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf93, arg227_1, arg228_1, arg229_1, arg230_1, 451584, grid=grid(451584), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        # Topologically Sorted Source Nodes: [x_319, x_320, x_321], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg231_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=288, bias=None)
        assert_size_stride(buf94, (8, 288, 14, 14), (56448, 1, 4032, 288))
        del arg231_1
        del buf93
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_322, x_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf95, arg232_1, arg233_1, arg234_1, arg235_1, 451584, grid=grid(451584), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        # Topologically Sorted Source Nodes: [x_322, x_323, x_324], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 96, 14, 14), (18816, 1, 1344, 96))
        del arg236_1
        del buf95
        buf97 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_325, x_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf97, buf96, arg237_1, arg238_1, arg239_1, arg240_1, 150528, grid=grid(150528), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        del buf96
        # Topologically Sorted Source Nodes: [x_325, x_326, x_327], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf98 = extern_kernels.convolution(buf97, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 576, 14, 14), (112896, 1, 8064, 576))
        del arg241_1
        del buf97
        buf99 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [x_328, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf99, arg242_1, arg243_1, arg244_1, arg245_1, 903168, grid=grid(903168), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        # Topologically Sorted Source Nodes: [x_328, x_329, x_330], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg246_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf100, (8, 576, 7, 7), (28224, 1, 4032, 576))
        del arg246_1
        del buf99
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_331, x_332], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf101, arg247_1, arg248_1, arg249_1, arg250_1, 225792, grid=grid(225792), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        # Topologically Sorted Source Nodes: [x_331, x_332, x_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 192, 7, 7), (9408, 1, 1344, 192))
        del arg251_1
        del buf101
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_334], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf103, arg252_1, arg253_1, arg254_1, arg255_1, 75264, grid=grid(75264), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        # Topologically Sorted Source Nodes: [x_335], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg256_1
        buf105 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_336, x_337], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf105, arg257_1, arg258_1, arg259_1, arg260_1, 451584, grid=grid(451584), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        # Topologically Sorted Source Nodes: [x_336, x_337, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg261_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf106, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg261_1
        del buf105
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_339, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf107, arg262_1, arg263_1, arg264_1, arg265_1, 451584, grid=grid(451584), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        # Topologically Sorted Source Nodes: [x_339, x_340, x_341], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 192, 7, 7), (9408, 1, 1344, 192))
        del arg266_1
        del buf107
        buf109 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [x_342, x_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf109, buf108, arg267_1, arg268_1, arg269_1, arg270_1, 75264, grid=grid(75264), stream=stream0)
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        del buf108
        # Topologically Sorted Source Nodes: [x_344], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg271_1
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_345, x_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf111, arg272_1, arg273_1, arg274_1, arg275_1, 451584, grid=grid(451584), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        # Topologically Sorted Source Nodes: [x_345, x_346, x_347], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg276_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf112, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg276_1
        del buf111
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_348, x_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf113, arg277_1, arg278_1, arg279_1, arg280_1, 451584, grid=grid(451584), stream=stream0)
        del arg277_1
        del arg278_1
        del arg279_1
        del arg280_1
        # Topologically Sorted Source Nodes: [x_348, x_349, x_350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf114 = extern_kernels.convolution(buf113, arg281_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 192, 7, 7), (9408, 1, 1344, 192))
        del arg281_1
        del buf113
        buf115 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_351, x_352], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf115, buf114, arg282_1, arg283_1, arg284_1, arg285_1, 75264, grid=grid(75264), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        del buf114
        # Topologically Sorted Source Nodes: [x_353], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg286_1
        buf117 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_354, x_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf117, arg287_1, arg288_1, arg289_1, arg290_1, 451584, grid=grid(451584), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        # Topologically Sorted Source Nodes: [x_354, x_355, x_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg291_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf118, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg291_1
        del buf117
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_357, x_358], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf119, arg292_1, arg293_1, arg294_1, arg295_1, 451584, grid=grid(451584), stream=stream0)
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        # Topologically Sorted Source Nodes: [x_357, x_358, x_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 192, 7, 7), (9408, 1, 1344, 192))
        del arg296_1
        del buf119
        buf121 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_360, x_361], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf121, buf120, arg297_1, arg298_1, arg299_1, arg300_1, 75264, grid=grid(75264), stream=stream0)
        del arg297_1
        del arg298_1
        del arg299_1
        del arg300_1
        del buf120
        # Topologically Sorted Source Nodes: [x_360, x_361, x_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf122 = extern_kernels.convolution(buf121, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg301_1
        del buf121
        buf123 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_363, x_364], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf123, arg302_1, arg303_1, arg304_1, arg305_1, 451584, grid=grid(451584), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        # Topologically Sorted Source Nodes: [x_363, x_364, x_365], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg306_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf124, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg306_1
        del buf123
        buf125 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_366, x_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf125, arg307_1, arg308_1, arg309_1, arg310_1, 451584, grid=grid(451584), stream=stream0)
        del arg307_1
        del arg308_1
        del arg309_1
        del arg310_1
        # Topologically Sorted Source Nodes: [x_366, x_367, x_368], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf126 = extern_kernels.convolution(buf125, arg311_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 320, 7, 7), (15680, 1, 2240, 320))
        del arg311_1
        del buf125
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_369], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf127, arg312_1, arg313_1, arg314_1, arg315_1, 125440, grid=grid(125440), stream=stream0)
        del arg312_1
        del arg313_1
        del arg314_1
        del arg315_1
        # Topologically Sorted Source Nodes: [x_369, x_370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
        del arg316_1
        del buf127
        buf130 = empty_strided_cuda((8, 1280, 1, 1), (1280, 1, 10240, 10240), torch.float32)
        # Topologically Sorted Source Nodes: [x_371, x_372, x_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28.run(buf128, arg317_1, arg318_1, arg319_1, arg320_1, buf130, 10240, 49, grid=grid(10240), stream=stream0)
        del arg317_1
        del arg318_1
        del arg319_1
        del arg320_1
        del buf128
        buf131 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_375], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg322_1, reinterpret_tensor(buf130, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg321_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf131)
        del arg321_1
        del arg322_1
        del buf130
    return (buf131, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((48, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((96, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((288, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((288, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((96, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((288, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((288, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((96, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((288, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((288, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((96, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((576, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((192, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('spnasnet_100', benchmark_compiled_module)
