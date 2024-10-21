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
# Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_124 => convolution_41
# Graph fragment:
#   %convolution_41 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
# Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_124 => convolution_41
# Graph fragment:
#   %convolution_41 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
# Topologically Sorted Source Nodes: [input_125, input_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_125 => add_83, mul_124, mul_125, sub_41
#   input_126 => relu_41
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_329), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %unsqueeze_333), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %unsqueeze_335), kwargs = {})
#   %relu_41 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_83,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/ti/cti33qhrb63jnqzqnrtsenfo7w2kdkdukoiwyw55mdhzdhlpfwal.py
# Topologically Sorted Source Nodes: [input_125, input_126, input_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_125 => add_83, mul_124, mul_125, sub_41
#   input_126 => relu_41
#   input_127 => convolution_42
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_329), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %unsqueeze_333), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %unsqueeze_335), kwargs = {})
#   %relu_41 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_83,), kwargs = {})
#   %convolution_42 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_41, %arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
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


# kernel path: /tmp/torchinductor_sahanp/rw/crwgtvcg7yovghatbwpk3asjp32xbuvhy3kgxny7iikvzx2qthpq.py
# Topologically Sorted Source Nodes: [input_128, input_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_128 => add_85, mul_127, mul_128, sub_42
#   input_129 => relu_42
# Graph fragment:
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, %unsqueeze_341), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %unsqueeze_343), kwargs = {})
#   %relu_42 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_85,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
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


# kernel path: /tmp/torchinductor_sahanp/c7/cc7bqtflts3bm3trozpfg6rsb6nyjvxcekctp2e3mz5je6w4eafy.py
# Topologically Sorted Source Nodes: [input_131, input_132, input_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_131 => add_87, mul_130, mul_131, sub_43
#   input_132 => relu_43
#   input_133 => convolution_44
# Graph fragment:
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %unsqueeze_349), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %unsqueeze_351), kwargs = {})
#   %relu_43 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_87,), kwargs = {})
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_43, %arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6c/c6cndq7nymafdqe77g2cki4oy6wrvx4w6y6zpdenhlbjswu6yo6l.py
# Topologically Sorted Source Nodes: [input_134, input_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_134 => add_89, mul_133, mul_134, sub_44
#   input_135 => relu_44
# Graph fragment:
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_353), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %unsqueeze_355), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %unsqueeze_357), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, %unsqueeze_359), kwargs = {})
#   %relu_44 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_89,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
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


# kernel path: /tmp/torchinductor_sahanp/cq/ccqkvt2cahyp4jt2vqfnnvn2w525cfpy6xclfp6oxdjztsqn2ayk.py
# Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_6 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_42, %relu_44, %relu_46], 1), kwargs = {})
triton_poi_fused_cat_7 = async_compile.triton('triton_poi_fused_cat_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((64*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((32*x1) + ((-64) + x0)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + ((32*x1) + ((-96) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + ((-96) + x0), tmp11, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 - tmp15
    tmp17 = tl.load(in_ptr4 + ((-96) + x0), tmp11, eviction_policy='evict_last', other=0.0)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = tl.full([1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp16 * tmp24
    tmp26 = tl.load(in_ptr5 + ((-96) + x0), tmp11, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.load(in_ptr6 + ((-96) + x0), tmp11, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp11, tmp31, tmp32)
    tmp34 = tl.where(tmp9, tmp10, tmp33)
    tmp35 = tl.where(tmp4, tmp5, tmp34)
    tl.store(out_ptr0 + (x2), tmp35, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zl/czlh2gq2cmbb7djfft4e3excavtqyoilzdsg2frpp3zwi7p6hafr.py
# Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_145 => convolution_48
# Graph fragment:
#   %convolution_48 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_47, %arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_8 = async_compile.triton('triton_poi_fused_convolution_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/av/cavp46upjgc2d57cre5arje2jtenisjpwf4ax4fbiwokus7ffatg.py
# Topologically Sorted Source Nodes: [cat_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_7 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_48, %relu_50, %relu_52, %relu_47], 1), kwargs = {})
triton_poi_fused_cat_9 = async_compile.triton('triton_poi_fused_cat_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 192
    x1 = (xindex // 192)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((64*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((32*x1) + ((-64) + x0)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + ((32*x1) + ((-96) + x0)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-96) + x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 - tmp16
    tmp18 = tl.load(in_ptr4 + ((-96) + x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tl.full([1], 1, tl.int32)
    tmp23 = tmp22 / tmp21
    tmp24 = 1.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp17 * tmp25
    tmp27 = tl.load(in_ptr5 + ((-96) + x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.load(in_ptr6 + ((-96) + x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full([1], 0, tl.int32)
    tmp32 = triton_helpers.maximum(tmp31, tmp30)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp14, tmp32, tmp33)
    tmp35 = tmp0 >= tmp12
    tmp36 = tl.full([1], 192, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tl.load(in_ptr7 + ((64*x1) + ((-128) + x0)), tmp35, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.where(tmp14, tmp34, tmp38)
    tmp40 = tl.where(tmp9, tmp10, tmp39)
    tmp41 = tl.where(tmp4, tmp5, tmp40)
    tl.store(out_ptr0 + (x2), tmp41, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gm/cgmqf37yx7j6nj3w2wy77am64xrmjf4prreutxjhe3z7kei7ulep.py
# Topologically Sorted Source Nodes: [input_161, input_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_161 => add_107, mul_160, mul_161, sub_53
#   input_162 => relu_53
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_425), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_160, %unsqueeze_429), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_161, %unsqueeze_431), kwargs = {})
#   %relu_53 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_107,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_sahanp/qx/cqx7mtsjoew554venb47z27d6gsvxc5t6hqf5y7yeddxnpxsfwvx.py
# Topologically Sorted Source Nodes: [input_161, input_162, input_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_161 => add_107, mul_160, mul_161, sub_53
#   input_162 => relu_53
#   input_163 => convolution_54
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_425), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_160, %unsqueeze_429), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_161, %unsqueeze_431), kwargs = {})
#   %relu_53 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_107,), kwargs = {})
#   %convolution_54 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_53, %arg66_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5b/c5buvl2lai5xls7mskmp633bxafwkyzsnbygbvuhfea4yxb7iqdv.py
# Topologically Sorted Source Nodes: [input_164, input_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_164 => add_109, mul_163, mul_164, sub_54
#   input_165 => relu_54
# Graph fragment:
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_54, %unsqueeze_433), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_435), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %unsqueeze_437), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %unsqueeze_439), kwargs = {})
#   %relu_54 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_109,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/r3/cr3ptk2zmb7u6u2akkmk3wc3nteb7is7vyhtp3nhiinyc5kx6zfm.py
# Topologically Sorted Source Nodes: [input_167, input_168, input_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_167 => add_111, mul_166, mul_167, sub_55
#   input_168 => relu_55
#   input_169 => convolution_56
# Graph fragment:
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_55, %unsqueeze_441), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_443), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_166, %unsqueeze_445), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_167, %unsqueeze_447), kwargs = {})
#   %relu_55 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_111,), kwargs = {})
#   %convolution_56 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_55, %arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10368
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (144*x2) + (1296*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/s5/cs5nhoqslndlaq4qhhfrjivajik6zdfrhd7s7hd7bgfibisxwffp.py
# Topologically Sorted Source Nodes: [input_170, input_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_170 => add_113, mul_169, mul_170, sub_56
#   input_171 => relu_56
# Graph fragment:
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_56, %unsqueeze_449), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %unsqueeze_451), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_169, %unsqueeze_453), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_170, %unsqueeze_455), kwargs = {})
#   %relu_56 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_113,), kwargs = {})
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
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 72
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


# kernel path: /tmp/torchinductor_sahanp/si/csiikzpjdwfmgzvkpwx6m4mndbyfocxhemy6bzke43nmys7gb3ns.py
# Topologically Sorted Source Nodes: [cat_8], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_8 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_54, %relu_56, %relu_58], 1), kwargs = {})
triton_poi_fused_cat_15 = async_compile.triton('triton_poi_fused_cat_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 288
    x1 = (xindex // 288)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 144, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((144*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 216, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((72*x1) + ((-144) + x0)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 288, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + ((72*x1) + ((-216) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + ((-216) + x0), tmp11, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 - tmp15
    tmp17 = tl.load(in_ptr4 + ((-216) + x0), tmp11, eviction_policy='evict_last', other=0.0)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = tl.full([1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp16 * tmp24
    tmp26 = tl.load(in_ptr5 + ((-216) + x0), tmp11, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.load(in_ptr6 + ((-216) + x0), tmp11, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp11, tmp31, tmp32)
    tmp34 = tl.where(tmp9, tmp10, tmp33)
    tmp35 = tl.where(tmp4, tmp5, tmp34)
    tl.store(out_ptr0 + (x2), tmp35, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wo/cwovuakai7tftslq3qr2qkxbaixb5abcgoqxysfutgn6niqvwwcu.py
# Topologically Sorted Source Nodes: [input_181], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_181 => convolution_60
# Graph fragment:
#   %convolution_60 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_59, %arg96_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_16 = async_compile.triton('triton_poi_fused_convolution_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20736
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (144*x2) + (1296*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2s/c2scsbxwzjb4hy3sryywyz232cptanfvtdoqqixjdsqeij5qclo2.py
# Topologically Sorted Source Nodes: [cat_9], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_9 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_60, %relu_62, %relu_64, %relu_59], 1), kwargs = {})
triton_poi_fused_cat_17 = async_compile.triton('triton_poi_fused_cat_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 432
    x1 = (xindex // 432)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 144, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((144*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 216, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((72*x1) + ((-144) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 288, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + ((72*x1) + ((-216) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-216) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 - tmp16
    tmp18 = tl.load(in_ptr4 + ((-216) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tl.full([1], 1, tl.int32)
    tmp23 = tmp22 / tmp21
    tmp24 = 1.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp17 * tmp25
    tmp27 = tl.load(in_ptr5 + ((-216) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.load(in_ptr6 + ((-216) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full([1], 0, tl.int32)
    tmp32 = triton_helpers.maximum(tmp31, tmp30)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp14, tmp32, tmp33)
    tmp35 = tmp0 >= tmp12
    tmp36 = tl.full([1], 432, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tl.load(in_ptr7 + ((144*x1) + ((-288) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.where(tmp14, tmp34, tmp38)
    tmp40 = tl.where(tmp9, tmp10, tmp39)
    tmp41 = tl.where(tmp4, tmp5, tmp40)
    tl.store(out_ptr0 + (x2), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eu/ceuzowlpgoyew6oinfkziksuppb4ht3xb7mxvhulaunzjlbo32n3.py
# Topologically Sorted Source Nodes: [input_197, input_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_197 => add_131, mul_196, mul_197, sub_65
#   input_198 => relu_65
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_65, %unsqueeze_521), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_196, %unsqueeze_525), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_197, %unsqueeze_527), kwargs = {})
#   %relu_65 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_131,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 288
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


# kernel path: /tmp/torchinductor_sahanp/lj/clj4ntch6mhyjgo3h77lyx46xa4lwtgu5zisahdcdoqqkd6buya7.py
# Topologically Sorted Source Nodes: [input_197, input_198, input_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_197 => add_131, mul_196, mul_197, sub_65
#   input_198 => relu_65
#   input_199 => convolution_66
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_65, %unsqueeze_521), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_196, %unsqueeze_525), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_197, %unsqueeze_527), kwargs = {})
#   %relu_65 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_131,), kwargs = {})
#   %convolution_66 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_65, %arg126_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 87552
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 288
    y1 = (yindex // 288)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (288*x2) + (2592*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cx/ccxzc2qbvv7neuckg55mj6r4pphodyvmoat7sbo5kqxov4fuq3bw.py
# Topologically Sorted Source Nodes: [input_200, input_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_200 => add_133, mul_199, mul_200, sub_66
#   input_201 => relu_66
# Graph fragment:
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_66, %unsqueeze_529), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %unsqueeze_533), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %unsqueeze_535), kwargs = {})
#   %relu_66 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_133,), kwargs = {})
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
    xnumel = 476672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 304
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


# kernel path: /tmp/torchinductor_sahanp/i6/ci6lgp2zo6vzin45ux6y5e4twj47eaxxir63wvo4akr2vlyzded7.py
# Topologically Sorted Source Nodes: [input_203, input_204, input_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_203 => add_135, mul_202, mul_203, sub_67
#   input_204 => relu_67
#   input_205 => convolution_68
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_67, %unsqueeze_537), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %unsqueeze_541), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %unsqueeze_543), kwargs = {})
#   %relu_67 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_135,), kwargs = {})
#   %convolution_68 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_67, %arg136_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 46208
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 304
    y1 = (yindex // 304)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (304*x2) + (2736*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7c/c7cm5olgickyvvw4zatyq4tyhepagoahvft6liree2kmpctpc26u.py
# Topologically Sorted Source Nodes: [input_206, input_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_206 => add_137, mul_205, mul_206, sub_68
#   input_207 => relu_68
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_545), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_205, %unsqueeze_549), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_206, %unsqueeze_551), kwargs = {})
#   %relu_68 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_137,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/o7/co7wrawk3imgwfzq6foicxmtax46tv7dbghtzke6gnsfeh6sktil.py
# Topologically Sorted Source Nodes: [cat_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_10 => cat_10
# Graph fragment:
#   %cat_10 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_66, %relu_68, %relu_70], 1), kwargs = {})
triton_poi_fused_cat_23 = async_compile.triton('triton_poi_fused_cat_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 608
    x1 = (xindex // 608)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 304, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((304*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 456, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((152*x1) + ((-304) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 608, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + ((152*x1) + ((-456) + x0)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + ((-456) + x0), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 - tmp15
    tmp17 = tl.load(in_ptr4 + ((-456) + x0), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = tl.full([1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp16 * tmp24
    tmp26 = tl.load(in_ptr5 + ((-456) + x0), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.load(in_ptr6 + ((-456) + x0), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp11, tmp31, tmp32)
    tmp34 = tl.where(tmp9, tmp10, tmp33)
    tmp35 = tl.where(tmp4, tmp5, tmp34)
    tl.store(out_ptr0 + (x2), tmp35, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5m/c5m7qe5z2gem74slzhh7ysya6rwcjxtlulxcjfv62rdwmsb43udk.py
# Topologically Sorted Source Nodes: [input_217], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_217 => convolution_72
# Graph fragment:
#   %convolution_72 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_71, %arg156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_24 = async_compile.triton('triton_poi_fused_convolution_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 92416
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 304
    y1 = (yindex // 304)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (304*x2) + (2736*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eo/ceopsd3hl2sbwq222zifl7khx7w6loownfctii5mfkutsapdidfh.py
# Topologically Sorted Source Nodes: [cat_11], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_11 => cat_11
# Graph fragment:
#   %cat_11 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_72, %relu_74, %relu_76, %relu_71], 1), kwargs = {})
triton_poi_fused_cat_25 = async_compile.triton('triton_poi_fused_cat_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1430016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 912
    x1 = (xindex // 912)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 304, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((304*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 456, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((152*x1) + ((-304) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 608, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + ((152*x1) + ((-456) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-456) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 - tmp16
    tmp18 = tl.load(in_ptr4 + ((-456) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tl.full([1], 1, tl.int32)
    tmp23 = tmp22 / tmp21
    tmp24 = 1.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp17 * tmp25
    tmp27 = tl.load(in_ptr5 + ((-456) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.load(in_ptr6 + ((-456) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full([1], 0, tl.int32)
    tmp32 = triton_helpers.maximum(tmp31, tmp30)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp14, tmp32, tmp33)
    tmp35 = tmp0 >= tmp12
    tmp36 = tl.full([1], 912, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tl.load(in_ptr7 + ((304*x1) + ((-608) + x0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.where(tmp14, tmp34, tmp38)
    tmp40 = tl.where(tmp9, tmp10, tmp39)
    tmp41 = tl.where(tmp4, tmp5, tmp40)
    tl.store(out_ptr0 + (x2), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6z/c6zgofmwm5rglolgn46dn7jyc63pjl4mi23odhhabatcc7da32at.py
# Topologically Sorted Source Nodes: [input_233, input_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_233 => add_155, mul_232, mul_233, sub_77
#   input_234 => relu_77
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_617), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_232, %unsqueeze_621), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_233, %unsqueeze_623), kwargs = {})
#   %relu_77 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_155,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/or/corqpd7lzej5xdreah3saqauzcoqgd5dvywjh5gye3qctg2efdqy.py
# Topologically Sorted Source Nodes: [input_233, input_234, input_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_233 => add_155, mul_232, mul_233, sub_77
#   input_234 => relu_77
#   input_235 => convolution_78
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_617), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_232, %unsqueeze_621), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_233, %unsqueeze_623), kwargs = {})
#   %relu_77 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_155,), kwargs = {})
#   %convolution_78 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_77, %arg186_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 460800
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 480
    y1 = (yindex // 480)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (480*x2) + (4320*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/db/cdbdlctcir5xcruieu6mp4jk6zwedfbluaziqtrbnrcpdakn2c42.py
# Topologically Sorted Source Nodes: [input_236, input_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_236 => add_157, mul_235, mul_236, sub_78
#   input_237 => relu_78
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_78, %unsqueeze_625), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_235, %unsqueeze_629), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_236, %unsqueeze_631), kwargs = {})
#   %relu_78 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_157,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
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


# kernel path: /tmp/torchinductor_sahanp/ho/choo52h3civqcwcz2blpdpfp24av6af4kl4gs257fqsqvf66uo3r.py
# Topologically Sorted Source Nodes: [input_236, input_237, input_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_236 => add_157, mul_235, mul_236, sub_78
#   input_237 => relu_78
#   input_238 => convolution_79
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_78, %unsqueeze_625), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_235, %unsqueeze_629), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_236, %unsqueeze_631), kwargs = {})
#   %relu_78 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_157,), kwargs = {})
#   %convolution_79 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_78, %arg191_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1048576, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 983040
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 960
    y1 = (yindex // 960)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (960*x2) + (8640*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zf/czf62iaw2glfu732epi3dpuyias3l7nsfzi3hfaufdsoaex7fp65.py
# Topologically Sorted Source Nodes: [input_239, input_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_239 => add_159, mul_238, mul_239, sub_79
#   input_240 => relu_79
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_79, %unsqueeze_633), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_635), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_238, %unsqueeze_637), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_239, %unsqueeze_639), kwargs = {})
#   %relu_79 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_159,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
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


# kernel path: /tmp/torchinductor_sahanp/5j/c5jd63uzxfosibfpxcfmqnlel5hic7ythwhpwyctyx5tlrwmk6zk.py
# Topologically Sorted Source Nodes: [input_239, input_240, input_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_239 => add_159, mul_238, mul_239, sub_79
#   input_240 => relu_79
#   input_241 => convolution_80
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_79, %unsqueeze_633), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_635), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_238, %unsqueeze_637), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_239, %unsqueeze_639), kwargs = {})
#   %relu_79 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_159,), kwargs = {})
#   %convolution_80 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_79, %arg196_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2097152, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1310720
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1024*x2) + (9216*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/v7/cv7auttrhntvbv3m3tpi5usz3k3tw5yyhrwzibqvm64mqppxqwz6.py
# Topologically Sorted Source Nodes: [input_242, input_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_242 => add_161, mul_241, mul_242, sub_80
#   input_243 => relu_80
# Graph fragment:
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_80, %unsqueeze_641), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_643), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_241, %unsqueeze_645), kwargs = {})
#   %add_161 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_242, %unsqueeze_647), kwargs = {})
#   %relu_80 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_161,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1280
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


# kernel path: /tmp/torchinductor_sahanp/sq/csqkhairnijhvr4ktyp7twearlmht4xesxsah4ytmd4brxepu7yo.py
# Topologically Sorted Source Nodes: [input_245, input_246, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   input_245 => add_163, mul_244, mul_245, sub_81
#   input_246 => relu_81
#   x_4 => mean_1
# Graph fragment:
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_81, %unsqueeze_649), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_651), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_244, %unsqueeze_653), kwargs = {})
#   %add_163 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_245, %unsqueeze_655), kwargs = {})
#   %relu_81 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_163,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_81, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_33 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_mean_relu_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_mean_relu_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (16384*x1)), None)
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
    tmp21 = 16.0
    tmp22 = tmp20 / tmp21
    tl.store(out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg17_1, (32, ), (1, ))
    assert_size_stride(arg18_1, (32, ), (1, ))
    assert_size_stride(arg19_1, (32, ), (1, ))
    assert_size_stride(arg20_1, (32, ), (1, ))
    assert_size_stride(arg21_1, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg27_1, (32, ), (1, ))
    assert_size_stride(arg28_1, (32, ), (1, ))
    assert_size_stride(arg29_1, (32, ), (1, ))
    assert_size_stride(arg30_1, (32, ), (1, ))
    assert_size_stride(arg31_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg32_1, (64, ), (1, ))
    assert_size_stride(arg33_1, (64, ), (1, ))
    assert_size_stride(arg34_1, (64, ), (1, ))
    assert_size_stride(arg35_1, (64, ), (1, ))
    assert_size_stride(arg36_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (64, ), (1, ))
    assert_size_stride(arg39_1, (64, ), (1, ))
    assert_size_stride(arg40_1, (64, ), (1, ))
    assert_size_stride(arg41_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg47_1, (32, ), (1, ))
    assert_size_stride(arg48_1, (32, ), (1, ))
    assert_size_stride(arg49_1, (32, ), (1, ))
    assert_size_stride(arg50_1, (32, ), (1, ))
    assert_size_stride(arg51_1, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg52_1, (64, ), (1, ))
    assert_size_stride(arg53_1, (64, ), (1, ))
    assert_size_stride(arg54_1, (64, ), (1, ))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg57_1, (32, ), (1, ))
    assert_size_stride(arg58_1, (32, ), (1, ))
    assert_size_stride(arg59_1, (32, ), (1, ))
    assert_size_stride(arg60_1, (32, ), (1, ))
    assert_size_stride(arg61_1, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (144, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg67_1, (144, ), (1, ))
    assert_size_stride(arg68_1, (144, ), (1, ))
    assert_size_stride(arg69_1, (144, ), (1, ))
    assert_size_stride(arg70_1, (144, ), (1, ))
    assert_size_stride(arg71_1, (144, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg72_1, (144, ), (1, ))
    assert_size_stride(arg73_1, (144, ), (1, ))
    assert_size_stride(arg74_1, (144, ), (1, ))
    assert_size_stride(arg75_1, (144, ), (1, ))
    assert_size_stride(arg76_1, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg77_1, (72, ), (1, ))
    assert_size_stride(arg78_1, (72, ), (1, ))
    assert_size_stride(arg79_1, (72, ), (1, ))
    assert_size_stride(arg80_1, (72, ), (1, ))
    assert_size_stride(arg81_1, (144, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg82_1, (144, ), (1, ))
    assert_size_stride(arg83_1, (144, ), (1, ))
    assert_size_stride(arg84_1, (144, ), (1, ))
    assert_size_stride(arg85_1, (144, ), (1, ))
    assert_size_stride(arg86_1, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg87_1, (72, ), (1, ))
    assert_size_stride(arg88_1, (72, ), (1, ))
    assert_size_stride(arg89_1, (72, ), (1, ))
    assert_size_stride(arg90_1, (72, ), (1, ))
    assert_size_stride(arg91_1, (144, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg92_1, (144, ), (1, ))
    assert_size_stride(arg93_1, (144, ), (1, ))
    assert_size_stride(arg94_1, (144, ), (1, ))
    assert_size_stride(arg95_1, (144, ), (1, ))
    assert_size_stride(arg96_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg97_1, (144, ), (1, ))
    assert_size_stride(arg98_1, (144, ), (1, ))
    assert_size_stride(arg99_1, (144, ), (1, ))
    assert_size_stride(arg100_1, (144, ), (1, ))
    assert_size_stride(arg101_1, (144, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg102_1, (144, ), (1, ))
    assert_size_stride(arg103_1, (144, ), (1, ))
    assert_size_stride(arg104_1, (144, ), (1, ))
    assert_size_stride(arg105_1, (144, ), (1, ))
    assert_size_stride(arg106_1, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg107_1, (72, ), (1, ))
    assert_size_stride(arg108_1, (72, ), (1, ))
    assert_size_stride(arg109_1, (72, ), (1, ))
    assert_size_stride(arg110_1, (72, ), (1, ))
    assert_size_stride(arg111_1, (144, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg112_1, (144, ), (1, ))
    assert_size_stride(arg113_1, (144, ), (1, ))
    assert_size_stride(arg114_1, (144, ), (1, ))
    assert_size_stride(arg115_1, (144, ), (1, ))
    assert_size_stride(arg116_1, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg117_1, (72, ), (1, ))
    assert_size_stride(arg118_1, (72, ), (1, ))
    assert_size_stride(arg119_1, (72, ), (1, ))
    assert_size_stride(arg120_1, (72, ), (1, ))
    assert_size_stride(arg121_1, (288, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg122_1, (288, ), (1, ))
    assert_size_stride(arg123_1, (288, ), (1, ))
    assert_size_stride(arg124_1, (288, ), (1, ))
    assert_size_stride(arg125_1, (288, ), (1, ))
    assert_size_stride(arg126_1, (304, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(arg127_1, (304, ), (1, ))
    assert_size_stride(arg128_1, (304, ), (1, ))
    assert_size_stride(arg129_1, (304, ), (1, ))
    assert_size_stride(arg130_1, (304, ), (1, ))
    assert_size_stride(arg131_1, (304, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(arg132_1, (304, ), (1, ))
    assert_size_stride(arg133_1, (304, ), (1, ))
    assert_size_stride(arg134_1, (304, ), (1, ))
    assert_size_stride(arg135_1, (304, ), (1, ))
    assert_size_stride(arg136_1, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(arg137_1, (152, ), (1, ))
    assert_size_stride(arg138_1, (152, ), (1, ))
    assert_size_stride(arg139_1, (152, ), (1, ))
    assert_size_stride(arg140_1, (152, ), (1, ))
    assert_size_stride(arg141_1, (304, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg142_1, (304, ), (1, ))
    assert_size_stride(arg143_1, (304, ), (1, ))
    assert_size_stride(arg144_1, (304, ), (1, ))
    assert_size_stride(arg145_1, (304, ), (1, ))
    assert_size_stride(arg146_1, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(arg147_1, (152, ), (1, ))
    assert_size_stride(arg148_1, (152, ), (1, ))
    assert_size_stride(arg149_1, (152, ), (1, ))
    assert_size_stride(arg150_1, (152, ), (1, ))
    assert_size_stride(arg151_1, (304, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(arg152_1, (304, ), (1, ))
    assert_size_stride(arg153_1, (304, ), (1, ))
    assert_size_stride(arg154_1, (304, ), (1, ))
    assert_size_stride(arg155_1, (304, ), (1, ))
    assert_size_stride(arg156_1, (304, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(arg157_1, (304, ), (1, ))
    assert_size_stride(arg158_1, (304, ), (1, ))
    assert_size_stride(arg159_1, (304, ), (1, ))
    assert_size_stride(arg160_1, (304, ), (1, ))
    assert_size_stride(arg161_1, (304, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(arg162_1, (304, ), (1, ))
    assert_size_stride(arg163_1, (304, ), (1, ))
    assert_size_stride(arg164_1, (304, ), (1, ))
    assert_size_stride(arg165_1, (304, ), (1, ))
    assert_size_stride(arg166_1, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(arg167_1, (152, ), (1, ))
    assert_size_stride(arg168_1, (152, ), (1, ))
    assert_size_stride(arg169_1, (152, ), (1, ))
    assert_size_stride(arg170_1, (152, ), (1, ))
    assert_size_stride(arg171_1, (304, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg172_1, (304, ), (1, ))
    assert_size_stride(arg173_1, (304, ), (1, ))
    assert_size_stride(arg174_1, (304, ), (1, ))
    assert_size_stride(arg175_1, (304, ), (1, ))
    assert_size_stride(arg176_1, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(arg177_1, (152, ), (1, ))
    assert_size_stride(arg178_1, (152, ), (1, ))
    assert_size_stride(arg179_1, (152, ), (1, ))
    assert_size_stride(arg180_1, (152, ), (1, ))
    assert_size_stride(arg181_1, (480, 912, 1, 1), (912, 1, 1, 1))
    assert_size_stride(arg182_1, (480, ), (1, ))
    assert_size_stride(arg183_1, (480, ), (1, ))
    assert_size_stride(arg184_1, (480, ), (1, ))
    assert_size_stride(arg185_1, (480, ), (1, ))
    assert_size_stride(arg186_1, (960, 480, 3, 3), (4320, 9, 3, 1))
    assert_size_stride(arg187_1, (960, ), (1, ))
    assert_size_stride(arg188_1, (960, ), (1, ))
    assert_size_stride(arg189_1, (960, ), (1, ))
    assert_size_stride(arg190_1, (960, ), (1, ))
    assert_size_stride(arg191_1, (1024, 960, 3, 3), (8640, 9, 3, 1))
    assert_size_stride(arg192_1, (1024, ), (1, ))
    assert_size_stride(arg193_1, (1024, ), (1, ))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (1024, ), (1, ))
    assert_size_stride(arg196_1, (1280, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(arg197_1, (1280, ), (1, ))
    assert_size_stride(arg198_1, (1280, ), (1, ))
    assert_size_stride(arg199_1, (1280, ), (1, ))
    assert_size_stride(arg200_1, (1280, ), (1, ))
    assert_size_stride(arg201_1, (1024, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg202_1, (1024, ), (1, ))
    assert_size_stride(arg203_1, (1024, ), (1, ))
    assert_size_stride(arg204_1, (1024, ), (1, ))
    assert_size_stride(arg205_1, (1024, ), (1, ))
    assert_size_stride(arg206_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg207_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_125, input_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_125, input_126, input_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg6_1, buf4, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [input_125, input_126, input_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 64, 56, 56), (200704, 1, 3584, 64))
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_128, input_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [input_130], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg11_1
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [input_131, input_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf8, arg12_1, arg13_1, arg14_1, arg15_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        buf9 = reinterpret_tensor(buf4, (32, 64, 3, 3), (576, 1, 192, 64), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_131, input_132, input_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg16_1, buf9, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg16_1
        # Topologically Sorted Source Nodes: [input_131, input_132, input_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf10 = extern_kernels.convolution(buf8, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del buf8
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_134, input_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf11, arg17_1, arg18_1, arg19_1, arg20_1, 802816, grid=grid(802816), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        # Topologically Sorted Source Nodes: [input_136], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg21_1
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_137, input_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf13, arg22_1, arg23_1, arg24_1, arg25_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        buf14 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_137, input_138, input_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg26_1, buf14, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg26_1
        # Topologically Sorted Source Nodes: [input_137, input_138, input_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf15 = extern_kernels.convolution(buf13, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del buf13
        buf16 = reinterpret_tensor(buf3, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
        triton_poi_fused_cat_7.run(buf6, buf11, buf15, arg27_1, arg28_1, arg29_1, arg30_1, buf16, 3211264, grid=grid(3211264), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        del buf11
        del buf15
        del buf6
        # Topologically Sorted Source Nodes: [input_142], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg31_1
        del buf16
        buf18 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [input_143, input_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf18, arg32_1, arg33_1, arg34_1, arg35_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        buf19 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(arg36_1, buf19, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg36_1
        # Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf18, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del buf19
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_146, input_147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf21, arg37_1, arg38_1, arg39_1, arg40_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        # Topologically Sorted Source Nodes: [input_148], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg41_1
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_149, input_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf23, arg42_1, arg43_1, arg44_1, arg45_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        buf24 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_149, input_150, input_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg46_1, buf24, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg46_1
        # Topologically Sorted Source Nodes: [input_149, input_150, input_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf25 = extern_kernels.convolution(buf23, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del buf23
        buf26 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [input_152, input_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf26, arg47_1, arg48_1, arg49_1, arg50_1, 802816, grid=grid(802816), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        # Topologically Sorted Source Nodes: [input_154], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg51_1
        buf28 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [input_155, input_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf28, arg52_1, arg53_1, arg54_1, arg55_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        buf29 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [input_155, input_156, input_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg56_1, buf29, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg56_1
        # Topologically Sorted Source Nodes: [input_155, input_156, input_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf30 = extern_kernels.convolution(buf28, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del buf28
        del buf29
        buf31 = empty_strided_cuda((8, 192, 56, 56), (602112, 1, 10752, 192), torch.float32)
        # Topologically Sorted Source Nodes: [cat_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_9.run(buf21, buf26, buf30, arg57_1, arg58_1, arg59_1, arg60_1, buf18, buf31, 4816896, grid=grid(4816896), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf18
        del buf21
        del buf26
        del buf30
        # Topologically Sorted Source Nodes: [input_160], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 128, 56, 56), (401408, 1, 7168, 128))
        del arg61_1
        del buf31
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [input_161, input_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf33, arg62_1, arg63_1, arg64_1, arg65_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        buf34 = empty_strided_cuda((144, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_161, input_162, input_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(arg66_1, buf34, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg66_1
        # Topologically Sorted Source Nodes: [input_161, input_162, input_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf35 = extern_kernels.convolution(buf33, buf34, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 144, 28, 28), (112896, 1, 4032, 144))
        del buf33
        del buf34
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [input_164, input_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf36, arg67_1, arg68_1, arg69_1, arg70_1, 903168, grid=grid(903168), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        # Topologically Sorted Source Nodes: [input_166], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 144, 28, 28), (112896, 1, 4032, 144))
        del arg71_1
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [input_167, input_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf38, arg72_1, arg73_1, arg74_1, arg75_1, 903168, grid=grid(903168), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        buf39 = empty_strided_cuda((72, 144, 3, 3), (1296, 1, 432, 144), torch.float32)
        # Topologically Sorted Source Nodes: [input_167, input_168, input_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg76_1, buf39, 10368, 9, grid=grid(10368, 9), stream=stream0)
        del arg76_1
        # Topologically Sorted Source Nodes: [input_167, input_168, input_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf40 = extern_kernels.convolution(buf38, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 72, 28, 28), (56448, 1, 2016, 72))
        del buf38
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [input_170, input_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf41, arg77_1, arg78_1, arg79_1, arg80_1, 451584, grid=grid(451584), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        # Topologically Sorted Source Nodes: [input_172], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 144, 28, 28), (112896, 1, 4032, 144))
        del arg81_1
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [input_173, input_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf43, arg82_1, arg83_1, arg84_1, arg85_1, 903168, grid=grid(903168), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        buf44 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [input_173, input_174, input_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg86_1, buf44, 10368, 9, grid=grid(10368, 9), stream=stream0)
        del arg86_1
        # Topologically Sorted Source Nodes: [input_173, input_174, input_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf45 = extern_kernels.convolution(buf43, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 72, 28, 28), (56448, 1, 2016, 72))
        del buf43
        buf46 = empty_strided_cuda((8, 288, 28, 28), (225792, 1, 8064, 288), torch.float32)
        # Topologically Sorted Source Nodes: [cat_8], Original ATen: [aten.cat]
        triton_poi_fused_cat_15.run(buf36, buf41, buf45, arg87_1, arg88_1, arg89_1, arg90_1, buf46, 1806336, grid=grid(1806336), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        del buf36
        del buf41
        del buf45
        # Topologically Sorted Source Nodes: [input_178], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 144, 28, 28), (112896, 1, 4032, 144))
        del arg91_1
        del buf46
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [input_179, input_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf48, arg92_1, arg93_1, arg94_1, arg95_1, 903168, grid=grid(903168), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        buf49 = empty_strided_cuda((144, 144, 3, 3), (1296, 1, 432, 144), torch.float32)
        # Topologically Sorted Source Nodes: [input_181], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_16.run(arg96_1, buf49, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg96_1
        # Topologically Sorted Source Nodes: [input_181], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf48, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 144, 28, 28), (112896, 1, 4032, 144))
        del buf49
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [input_182, input_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf51, arg97_1, arg98_1, arg99_1, arg100_1, 903168, grid=grid(903168), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        # Topologically Sorted Source Nodes: [input_184], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 144, 28, 28), (112896, 1, 4032, 144))
        del arg101_1
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [input_185, input_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf53, arg102_1, arg103_1, arg104_1, arg105_1, 903168, grid=grid(903168), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        buf54 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [input_185, input_186, input_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg106_1, buf54, 10368, 9, grid=grid(10368, 9), stream=stream0)
        del arg106_1
        # Topologically Sorted Source Nodes: [input_185, input_186, input_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf55 = extern_kernels.convolution(buf53, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 72, 28, 28), (56448, 1, 2016, 72))
        del buf53
        buf56 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [input_188, input_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf56, arg107_1, arg108_1, arg109_1, arg110_1, 451584, grid=grid(451584), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        # Topologically Sorted Source Nodes: [input_190], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 144, 28, 28), (112896, 1, 4032, 144))
        del arg111_1
        buf58 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [input_191, input_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf58, arg112_1, arg113_1, arg114_1, arg115_1, 903168, grid=grid(903168), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        buf59 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [input_191, input_192, input_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg116_1, buf59, 10368, 9, grid=grid(10368, 9), stream=stream0)
        del arg116_1
        # Topologically Sorted Source Nodes: [input_191, input_192, input_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf60 = extern_kernels.convolution(buf58, buf59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 72, 28, 28), (56448, 1, 2016, 72))
        del buf58
        del buf59
        buf61 = empty_strided_cuda((8, 432, 28, 28), (338688, 1, 12096, 432), torch.float32)
        # Topologically Sorted Source Nodes: [cat_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_17.run(buf51, buf56, buf60, arg117_1, arg118_1, arg119_1, arg120_1, buf48, buf61, 2709504, grid=grid(2709504), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        del buf48
        del buf51
        del buf56
        del buf60
        # Topologically Sorted Source Nodes: [input_196], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 288, 28, 28), (225792, 1, 8064, 288))
        del arg121_1
        del buf61
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [input_197, input_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf63, arg122_1, arg123_1, arg124_1, arg125_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        buf64 = empty_strided_cuda((304, 288, 3, 3), (2592, 1, 864, 288), torch.float32)
        # Topologically Sorted Source Nodes: [input_197, input_198, input_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(arg126_1, buf64, 87552, 9, grid=grid(87552, 9), stream=stream0)
        del arg126_1
        # Topologically Sorted Source Nodes: [input_197, input_198, input_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf65 = extern_kernels.convolution(buf63, buf64, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 304, 14, 14), (59584, 1, 4256, 304))
        del buf63
        del buf64
        buf66 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [input_200, input_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf66, arg127_1, arg128_1, arg129_1, arg130_1, 476672, grid=grid(476672), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        # Topologically Sorted Source Nodes: [input_202], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 304, 14, 14), (59584, 1, 4256, 304))
        del arg131_1
        buf68 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [input_203, input_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf68, arg132_1, arg133_1, arg134_1, arg135_1, 476672, grid=grid(476672), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        buf69 = empty_strided_cuda((152, 304, 3, 3), (2736, 1, 912, 304), torch.float32)
        # Topologically Sorted Source Nodes: [input_203, input_204, input_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(arg136_1, buf69, 46208, 9, grid=grid(46208, 9), stream=stream0)
        del arg136_1
        # Topologically Sorted Source Nodes: [input_203, input_204, input_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf70 = extern_kernels.convolution(buf68, buf69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del buf68
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [input_206, input_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf71, arg137_1, arg138_1, arg139_1, arg140_1, 238336, grid=grid(238336), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        # Topologically Sorted Source Nodes: [input_208], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 304, 14, 14), (59584, 1, 4256, 304))
        del arg141_1
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [input_209, input_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf73, arg142_1, arg143_1, arg144_1, arg145_1, 476672, grid=grid(476672), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        buf74 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [input_209, input_210, input_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(arg146_1, buf74, 46208, 9, grid=grid(46208, 9), stream=stream0)
        del arg146_1
        # Topologically Sorted Source Nodes: [input_209, input_210, input_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf75 = extern_kernels.convolution(buf73, buf74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del buf73
        buf76 = empty_strided_cuda((8, 608, 14, 14), (119168, 1, 8512, 608), torch.float32)
        # Topologically Sorted Source Nodes: [cat_10], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf66, buf71, buf75, arg147_1, arg148_1, arg149_1, arg150_1, buf76, 953344, grid=grid(953344), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        del buf66
        del buf71
        del buf75
        # Topologically Sorted Source Nodes: [input_214], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 304, 14, 14), (59584, 1, 4256, 304))
        del arg151_1
        del buf76
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [input_215, input_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf78, arg152_1, arg153_1, arg154_1, arg155_1, 476672, grid=grid(476672), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        buf79 = empty_strided_cuda((304, 304, 3, 3), (2736, 1, 912, 304), torch.float32)
        # Topologically Sorted Source Nodes: [input_217], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(arg156_1, buf79, 92416, 9, grid=grid(92416, 9), stream=stream0)
        del arg156_1
        # Topologically Sorted Source Nodes: [input_217], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf78, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 304, 14, 14), (59584, 1, 4256, 304))
        del buf79
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [input_218, input_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf81, arg157_1, arg158_1, arg159_1, arg160_1, 476672, grid=grid(476672), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        # Topologically Sorted Source Nodes: [input_220], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 304, 14, 14), (59584, 1, 4256, 304))
        del arg161_1
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [input_221, input_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf83, arg162_1, arg163_1, arg164_1, arg165_1, 476672, grid=grid(476672), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        buf84 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [input_221, input_222, input_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(arg166_1, buf84, 46208, 9, grid=grid(46208, 9), stream=stream0)
        del arg166_1
        # Topologically Sorted Source Nodes: [input_221, input_222, input_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf85 = extern_kernels.convolution(buf83, buf84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del buf83
        buf86 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [input_224, input_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf86, arg167_1, arg168_1, arg169_1, arg170_1, 238336, grid=grid(238336), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        # Topologically Sorted Source Nodes: [input_226], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 304, 14, 14), (59584, 1, 4256, 304))
        del arg171_1
        buf88 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [input_227, input_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf88, arg172_1, arg173_1, arg174_1, arg175_1, 476672, grid=grid(476672), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        buf89 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [input_227, input_228, input_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(arg176_1, buf89, 46208, 9, grid=grid(46208, 9), stream=stream0)
        del arg176_1
        # Topologically Sorted Source Nodes: [input_227, input_228, input_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf90 = extern_kernels.convolution(buf88, buf89, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 152, 14, 14), (29792, 1, 2128, 152))
        del buf88
        del buf89
        buf91 = empty_strided_cuda((8, 912, 14, 14), (178752, 1, 12768, 912), torch.float32)
        # Topologically Sorted Source Nodes: [cat_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_25.run(buf81, buf86, buf90, arg177_1, arg178_1, arg179_1, arg180_1, buf78, buf91, 1430016, grid=grid(1430016), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        del buf78
        del buf81
        del buf86
        del buf90
        # Topologically Sorted Source Nodes: [input_232], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 480, 14, 14), (94080, 1, 6720, 480))
        del arg181_1
        del buf91
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [input_233, input_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf93, arg182_1, arg183_1, arg184_1, arg185_1, 752640, grid=grid(752640), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        buf94 = empty_strided_cuda((960, 480, 3, 3), (4320, 1, 1440, 480), torch.float32)
        # Topologically Sorted Source Nodes: [input_233, input_234, input_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(arg186_1, buf94, 460800, 9, grid=grid(460800, 9), stream=stream0)
        del arg186_1
        # Topologically Sorted Source Nodes: [input_233, input_234, input_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf95 = extern_kernels.convolution(buf93, buf94, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del buf93
        del buf94
        buf96 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [input_236, input_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf96, arg187_1, arg188_1, arg189_1, arg190_1, 376320, grid=grid(376320), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        buf97 = empty_strided_cuda((1024, 960, 3, 3), (8640, 1, 2880, 960), torch.float32)
        # Topologically Sorted Source Nodes: [input_236, input_237, input_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29.run(arg191_1, buf97, 983040, 9, grid=grid(983040, 9), stream=stream0)
        del arg191_1
        # Topologically Sorted Source Nodes: [input_236, input_237, input_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf98 = extern_kernels.convolution(buf96, buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
        del buf96
        del buf97
        buf99 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [input_239, input_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf99, arg192_1, arg193_1, arg194_1, arg195_1, 401408, grid=grid(401408), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        buf100 = empty_strided_cuda((1280, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [input_239, input_240, input_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31.run(arg196_1, buf100, 1310720, 9, grid=grid(1310720, 9), stream=stream0)
        del arg196_1
        # Topologically Sorted Source Nodes: [input_239, input_240, input_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf101 = extern_kernels.convolution(buf99, buf100, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 1280, 4, 4), (20480, 1, 5120, 1280))
        del buf100
        del buf99
        buf102 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [input_242, input_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf102, arg197_1, arg198_1, arg199_1, arg200_1, 163840, grid=grid(163840), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        # Topologically Sorted Source Nodes: [input_242, input_243, input_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 1024, 4, 4), (16384, 1, 4096, 1024))
        del arg201_1
        del buf102
        buf105 = empty_strided_cuda((8, 1024, 1, 1), (1024, 1, 8192, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [input_245, input_246, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_33.run(buf103, arg202_1, arg203_1, arg204_1, arg205_1, buf105, 8192, 16, grid=grid(8192), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        del buf103
        buf106 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg207_1, reinterpret_tensor(buf105, (8, 1024), (1024, 1), 0), reinterpret_tensor(arg206_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf106)
        del arg206_1
        del arg207_1
        del buf105
    return (buf106, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((144, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((144, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((144, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((144, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((144, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((144, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((288, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((304, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((304, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((304, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((304, 608, 1, 1), (608, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((304, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((304, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((304, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((480, 912, 1, 1), (912, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((960, 480, 3, 3), (4320, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, 960, 3, 3), (8640, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1280, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('selecsls42b', benchmark_compiled_module)
