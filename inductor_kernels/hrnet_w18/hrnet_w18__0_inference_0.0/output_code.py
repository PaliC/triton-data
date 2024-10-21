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
# Topologically Sorted Source Nodes: [x_818], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_818 => convolution_325
# Graph fragment:
#   %convolution_325 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/g4/cg44vkswcejqn6yhzia3bvq33k4wssu3n2hwq3ducaq2xqudsjyz.py
# Topologically Sorted Source Nodes: [x_818], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_818 => convolution_325
# Graph fragment:
#   %convolution_325 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cz/cczi5gg3b7juhbzml2qsxaep3irwgnmmqqno5qffufpvejua2ji4.py
# Topologically Sorted Source Nodes: [x_819, x_820], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_819 => add_952, mul_1100, mul_1101, sub_325
#   x_820 => relu_284
# Graph fragment:
#   %sub_325 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_325, %unsqueeze_2632), kwargs = {})
#   %mul_1100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_325, %unsqueeze_2634), kwargs = {})
#   %mul_1101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1100, %unsqueeze_2636), kwargs = {})
#   %add_952 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1101, %unsqueeze_2638), kwargs = {})
#   %relu_284 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_952,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_sahanp/5s/c5s2nb2mvs22plzfsn2cvj4qw4jfyr6v7alm46qr7cqpz47ukdve.py
# Topologically Sorted Source Nodes: [x_819, x_820, x_821], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_819 => add_952, mul_1100, mul_1101, sub_325
#   x_820 => relu_284
#   x_821 => convolution_326
# Graph fragment:
#   %sub_325 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_325, %unsqueeze_2632), kwargs = {})
#   %mul_1100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_325, %unsqueeze_2634), kwargs = {})
#   %mul_1101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1100, %unsqueeze_2636), kwargs = {})
#   %add_952 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1101, %unsqueeze_2638), kwargs = {})
#   %relu_284 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_952,), kwargs = {})
#   %convolution_326 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_284, %arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/rw/crwgtvcg7yovghatbwpk3asjp32xbuvhy3kgxny7iikvzx2qthpq.py
# Topologically Sorted Source Nodes: [x_822, x_823], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_822 => add_954, mul_1103, mul_1104, sub_326
#   x_823 => relu_285
# Graph fragment:
#   %sub_326 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_326, %unsqueeze_2640), kwargs = {})
#   %mul_1103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_326, %unsqueeze_2642), kwargs = {})
#   %mul_1104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1103, %unsqueeze_2644), kwargs = {})
#   %add_954 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1104, %unsqueeze_2646), kwargs = {})
#   %relu_285 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_954,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/uc/cuc2vth5rjktvg5iouxo3nbchdzv6xbaweze6qakitu5aevbqgpf.py
# Topologically Sorted Source Nodes: [x_831, input_239, x_832, x_833], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_239 => add_962, mul_1115, mul_1116, sub_330
#   x_831 => add_960, mul_1112, mul_1113, sub_329
#   x_832 => add_963
#   x_833 => relu_288
# Graph fragment:
#   %sub_329 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_329, %unsqueeze_2664), kwargs = {})
#   %mul_1112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_329, %unsqueeze_2666), kwargs = {})
#   %mul_1113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1112, %unsqueeze_2668), kwargs = {})
#   %add_960 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1113, %unsqueeze_2670), kwargs = {})
#   %sub_330 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_330, %unsqueeze_2672), kwargs = {})
#   %mul_1115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_330, %unsqueeze_2674), kwargs = {})
#   %mul_1116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1115, %unsqueeze_2676), kwargs = {})
#   %add_962 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1116, %unsqueeze_2678), kwargs = {})
#   %add_963 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_960, %add_962), kwargs = {})
#   %relu_288 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_963,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
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


# kernel path: /tmp/torchinductor_sahanp/k3/ck3jwtuarjqco25jddmsc3uevs366ow5ea233k67usgfgt6umj5b.py
# Topologically Sorted Source Nodes: [x_841, x_842, x_843], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_841 => add_969, mul_1124, mul_1125, sub_333
#   x_842 => add_970
#   x_843 => relu_291
# Graph fragment:
#   %sub_333 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_333, %unsqueeze_2696), kwargs = {})
#   %mul_1124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_333, %unsqueeze_2698), kwargs = {})
#   %mul_1125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1124, %unsqueeze_2700), kwargs = {})
#   %add_969 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1125, %unsqueeze_2702), kwargs = {})
#   %add_970 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_969, %relu_288), kwargs = {})
#   %relu_291 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_970,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
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


# kernel path: /tmp/torchinductor_sahanp/ex/cexo7pjejg6ndmybyylqnxsihqm7q76v6y4lzy5nzpf6udhmqj7q.py
# Topologically Sorted Source Nodes: [input_240], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_240 => convolution_340
# Graph fragment:
#   %convolution_340 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_297, %arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_7 = async_compile.triton('triton_poi_fused_convolution_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2c/c2cn2lp7mkfjba6qvn4jsoqgzb6t7c7nqbmnav5fjl3zlyslvgna.py
# Topologically Sorted Source Nodes: [input_241, input_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_241 => add_986, mul_1145, mul_1146, sub_340
#   input_242 => relu_298
# Graph fragment:
#   %sub_340 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_340, %unsqueeze_2752), kwargs = {})
#   %mul_1145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_340, %unsqueeze_2754), kwargs = {})
#   %mul_1146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1145, %unsqueeze_2756), kwargs = {})
#   %add_986 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1146, %unsqueeze_2758), kwargs = {})
#   %relu_298 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_986,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 18
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


# kernel path: /tmp/torchinductor_sahanp/qe/cqel4sq6x7l6x4tbveh7kdb6ofif56kimv76wq6qqqgcha74lzk4.py
# Topologically Sorted Source Nodes: [x_864], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_864 => convolution_342
# Graph fragment:
#   %convolution_342 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_298, %arg86_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_9 = async_compile.triton('triton_poi_fused_convolution_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 324
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 18
    y1 = (yindex // 18)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (18*x2) + (162*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zd/czduklm66qob2bkfskk73cd2jx47pkljkps2ujcueveqaj3gc6tu.py
# Topologically Sorted Source Nodes: [x_868, x_869, x_870], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_868 => add_992, mul_1154, mul_1155, sub_343
#   x_869 => add_993
#   x_870 => relu_301
# Graph fragment:
#   %sub_343 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_343, %unsqueeze_2776), kwargs = {})
#   %mul_1154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_343, %unsqueeze_2778), kwargs = {})
#   %mul_1155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1154, %unsqueeze_2780), kwargs = {})
#   %add_992 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1155, %unsqueeze_2782), kwargs = {})
#   %add_993 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_992, %relu_298), kwargs = {})
#   %relu_301 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_993,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 18
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


# kernel path: /tmp/torchinductor_sahanp/52/c52khomve2lczy3kj7k3tbxdybxi245mqhionrmjgn64ur5xpstk.py
# Topologically Sorted Source Nodes: [input_243], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_243 => convolution_341
# Graph fragment:
#   %convolution_341 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_297, %arg81_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_11 = async_compile.triton('triton_poi_fused_convolution_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/64/c64a6ghcrlk6fuylsfyftfsqrgai2gkyn6snu2atxvdmkeixjbnu.py
# Topologically Sorted Source Nodes: [input_244, input_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_244 => add_988, mul_1148, mul_1149, sub_341
#   input_245 => relu_299
# Graph fragment:
#   %sub_341 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_341, %unsqueeze_2760), kwargs = {})
#   %mul_1148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_341, %unsqueeze_2762), kwargs = {})
#   %mul_1149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1148, %unsqueeze_2764), kwargs = {})
#   %add_988 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1149, %unsqueeze_2766), kwargs = {})
#   %relu_299 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_988,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 36
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


# kernel path: /tmp/torchinductor_sahanp/av/caves4n62dtebjxcl6wcoa6xvosgh4fu7ashg53odwdrlbcdas2q.py
# Topologically Sorted Source Nodes: [x_892], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_892 => convolution_350
# Graph fragment:
#   %convolution_350 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_299, %arg126_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_13 = async_compile.triton('triton_poi_fused_convolution_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1296
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 36
    y1 = (yindex // 36)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (36*x2) + (324*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bh/cbh3adxf2vdgsx4xwko3u7r5z5f2wv7hstgznh2jdptd6ndbl36l.py
# Topologically Sorted Source Nodes: [x_896, x_897, x_898], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_896 => add_1012, mul_1178, mul_1179, sub_351
#   x_897 => add_1013
#   x_898 => relu_309
# Graph fragment:
#   %sub_351 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_351, %unsqueeze_2840), kwargs = {})
#   %mul_1178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_351, %unsqueeze_2842), kwargs = {})
#   %mul_1179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1178, %unsqueeze_2844), kwargs = {})
#   %add_1012 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1179, %unsqueeze_2846), kwargs = {})
#   %add_1013 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1012, %relu_299), kwargs = {})
#   %relu_309 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1013,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 36
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


# kernel path: /tmp/torchinductor_sahanp/fr/cfr3epz2atngszzdjgcmijcm7boegbxwvpfqarxbicyyc5zbiq5j.py
# Topologically Sorted Source Nodes: [input_247, input_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_247 => add_1030, mul_1199, mul_1200, sub_358
#   input_248 => _unsafe_index_31
# Graph fragment:
#   %sub_358 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_358, %unsqueeze_2896), kwargs = {})
#   %mul_1199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_358, %unsqueeze_2898), kwargs = {})
#   %mul_1200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1199, %unsqueeze_2900), kwargs = {})
#   %add_1030 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1200, %unsqueeze_2902), kwargs = {})
#   %_unsafe_index_31 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1030, [None, None, %unsqueeze_2903, %convert_element_type_845]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x2 = (xindex // 3136) % 18
    x3 = (xindex // 56448)
    x5 = xindex
    tmp10 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x2 + (18*tmp8) + (504*tmp4) + (14112*x3)), xmask, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr0 + (x5), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fg/cfg3etqbdgwzc2y7ipclqkab4gi6qm7qlx7rd74gy2oozwfgkbr3.py
# Topologically Sorted Source Nodes: [x_889, x_890, x_891, y_65, shortcut_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   shortcut_26 => relu_316
#   x_889 => add_1007, mul_1172, mul_1173, sub_349
#   x_890 => add_1008
#   x_891 => relu_307
#   y_65 => add_1035
# Graph fragment:
#   %sub_349 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_349, %unsqueeze_2824), kwargs = {})
#   %mul_1172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_349, %unsqueeze_2826), kwargs = {})
#   %mul_1173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1172, %unsqueeze_2828), kwargs = {})
#   %add_1007 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1173, %unsqueeze_2830), kwargs = {})
#   %add_1008 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1007, %relu_305), kwargs = {})
#   %relu_307 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1008,), kwargs = {})
#   %add_1035 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_307, %_unsafe_index_31), kwargs = {})
#   %relu_316 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1035,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 18
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 3136
    y3 = (yindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x1 + (18*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x1 + (18*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y2 + (3136*x1) + (56448*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp21 = tmp19 + tmp20
    tmp22 = triton_helpers.maximum(tmp18, tmp21)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + (18*y0)), tmp19, xmask & ymask)
    tl.store(out_ptr0 + (x1 + (18*y0)), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lw/clwehj4dtn64ylmuidoaxnflnmyrdfkmqypshuqfvwr5zufi7skg.py
# Topologically Sorted Source Nodes: [x_931, x_932, x_933], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_931 => add_1049, mul_1221, mul_1222, sub_364
#   x_932 => add_1050
#   x_933 => relu_322
# Graph fragment:
#   %sub_364 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_364, %unsqueeze_2945), kwargs = {})
#   %mul_1221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_364, %unsqueeze_2947), kwargs = {})
#   %mul_1222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1221, %unsqueeze_2949), kwargs = {})
#   %add_1049 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1222, %unsqueeze_2951), kwargs = {})
#   %add_1050 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1049, %relu_320), kwargs = {})
#   %relu_322 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1050,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 18
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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


# kernel path: /tmp/torchinductor_sahanp/qy/cqycu73ipznyfcnelnhll3asssqgai7zme4yupcjmt2hs6vbnpuy.py
# Topologically Sorted Source Nodes: [input_249], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_249 => convolution_359
# Graph fragment:
#   %convolution_359 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_307, %arg171_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_18 = async_compile.triton('triton_poi_fused_convolution_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 648
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 18
    y1 = (yindex // 18)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (18*x2) + (162*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dj/cdjkxrfqn4g3rriemb3xbtjkb6hsftuq6y2seww3banv35mmluvr.py
# Topologically Sorted Source Nodes: [input_250, y_66, shortcut_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_250 => add_1037, mul_1206, mul_1207, sub_359
#   shortcut_27 => relu_317
#   y_66 => add_1038
# Graph fragment:
#   %sub_359 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_359, %unsqueeze_2905), kwargs = {})
#   %mul_1206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_359, %unsqueeze_2907), kwargs = {})
#   %mul_1207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1206, %unsqueeze_2909), kwargs = {})
#   %add_1037 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1207, %unsqueeze_2911), kwargs = {})
#   %add_1038 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1037, %relu_315), kwargs = {})
#   %relu_317 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_1038,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 36
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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


# kernel path: /tmp/torchinductor_sahanp/ob/cobbnewwgmkmmjrqv5i6iwd7m5mamsknej4uq3yh5qy6italnqwg.py
# Topologically Sorted Source Nodes: [input_251], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_251 => convolution_360
# Graph fragment:
#   %convolution_360 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_317, %arg176_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_20 = async_compile.triton('triton_poi_fused_convolution_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2592
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 36
    y1 = (yindex // 36)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (36*x2) + (324*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jb/cjbpwifvsc36z2ajmwaqoqedzrwaoc5wiohaphiht7bsztmuzmul.py
# Topologically Sorted Source Nodes: [input_252, input_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_252 => add_1040, mul_1209, mul_1210, sub_360
#   input_253 => relu_318
# Graph fragment:
#   %sub_360 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_360, %unsqueeze_2913), kwargs = {})
#   %mul_1209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_360, %unsqueeze_2915), kwargs = {})
#   %mul_1210 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1209, %unsqueeze_2917), kwargs = {})
#   %add_1040 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1210, %unsqueeze_2919), kwargs = {})
#   %relu_318 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1040,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
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


# kernel path: /tmp/torchinductor_sahanp/pn/cpn6bqzyov7bny5r7asr2vzc6ouv6eafjn3gaby4gvj2e5ko4rxk.py
# Topologically Sorted Source Nodes: [x_976], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_976 => convolution_377
# Graph fragment:
#   %convolution_377 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_318, %arg261_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_22 = async_compile.triton('triton_poi_fused_convolution_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_22(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5184
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (72*x2) + (648*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6t/c6thqynhvj6rmfb7lvpgvqtvo5oark4egemdftn7loymxk5snjmt.py
# Topologically Sorted Source Nodes: [x_980, x_981, x_982], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_980 => add_1084, mul_1263, mul_1264, sub_378
#   x_981 => add_1085
#   x_982 => relu_336
# Graph fragment:
#   %sub_378 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_378, %unsqueeze_3057), kwargs = {})
#   %mul_1263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_378, %unsqueeze_3059), kwargs = {})
#   %mul_1264 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1263, %unsqueeze_3061), kwargs = {})
#   %add_1084 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1264, %unsqueeze_3063), kwargs = {})
#   %add_1085 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1084, %relu_318), kwargs = {})
#   %relu_336 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1085,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 72
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


# kernel path: /tmp/torchinductor_sahanp/2w/c2wvagcclgopqc4d7hxd4ex626hv7kcqrvaup7no7hnxfi4wwhkx.py
# Topologically Sorted Source Nodes: [input_263, input_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_263 => add_1119, mul_1301, mul_1302, sub_388
#   input_264 => _unsafe_index_34
# Graph fragment:
#   %sub_388 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_388, %unsqueeze_3139), kwargs = {})
#   %mul_1301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_388, %unsqueeze_3141), kwargs = {})
#   %mul_1302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1301, %unsqueeze_3143), kwargs = {})
#   %add_1119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1302, %unsqueeze_3145), kwargs = {})
#   %_unsafe_index_34 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1119, [None, None, %unsqueeze_3146, %convert_element_type_917]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x2 = (xindex // 784) % 36
    x3 = (xindex // 28224)
    x5 = xindex
    tmp10 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x2 + (36*tmp8) + (504*tmp4) + (7056*x3)), xmask, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr0 + (x5), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/i2/ci2p6zjgzdvhjixetz2mwmwzxweg73hqvqoz6lovbhy774gak6x4.py
# Topologically Sorted Source Nodes: [x_973, x_974, x_975, input_261, y_69, y_70, shortcut_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_261 => add_1116, mul_1298, mul_1299, sub_387
#   shortcut_29 => relu_344
#   x_973 => add_1079, mul_1257, mul_1258, sub_376
#   x_974 => add_1080
#   x_975 => relu_334
#   y_69 => add_1117
#   y_70 => add_1124
# Graph fragment:
#   %sub_376 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_376, %unsqueeze_3041), kwargs = {})
#   %mul_1257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_376, %unsqueeze_3043), kwargs = {})
#   %mul_1258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1257, %unsqueeze_3045), kwargs = {})
#   %add_1079 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1258, %unsqueeze_3047), kwargs = {})
#   %add_1080 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1079, %relu_332), kwargs = {})
#   %relu_334 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_1080,), kwargs = {})
#   %sub_387 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_387, %unsqueeze_3131), kwargs = {})
#   %mul_1298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_387, %unsqueeze_3133), kwargs = {})
#   %mul_1299 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1298, %unsqueeze_3135), kwargs = {})
#   %add_1116 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1299, %unsqueeze_3137), kwargs = {})
#   %add_1117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1116, %relu_334), kwargs = {})
#   %add_1124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1117, %_unsafe_index_34), kwargs = {})
#   %relu_344 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1124,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 784
    y3 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x1 + (36*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x1 + (36*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x1 + (36*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (y2 + (784*x1) + (28224*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp22 = tmp20 - tmp21
    tmp24 = tmp23 + tmp4
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp7 / tmp25
    tmp27 = tmp26 * tmp9
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp32 + tmp19
    tmp35 = tmp33 + tmp34
    tmp36 = triton_helpers.maximum(tmp18, tmp35)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + (36*y0)), tmp19, xmask & ymask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x1 + (36*y0)), tmp36, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cn/ccnf56aulxlxldetf22is2sb55zl4chhas473wn37qvzt5oduwll.py
# Topologically Sorted Source Nodes: [input_255, input_256, y_67, input_258, input_259, y_68, shortcut_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_255 => add_1102, mul_1284, mul_1285, sub_385
#   input_256 => _unsafe_index_32
#   input_258 => add_1109, mul_1291, mul_1292, sub_386
#   input_259 => _unsafe_index_33
#   shortcut_28 => relu_343
#   y_67 => add_1107
#   y_68 => add_1114
# Graph fragment:
#   %sub_385 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_385, %unsqueeze_3113), kwargs = {})
#   %mul_1284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_385, %unsqueeze_3115), kwargs = {})
#   %mul_1285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1284, %unsqueeze_3117), kwargs = {})
#   %add_1102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1285, %unsqueeze_3119), kwargs = {})
#   %_unsafe_index_32 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1102, [None, None, %unsqueeze_3120, %convert_element_type_903]), kwargs = {})
#   %add_1107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_326, %_unsafe_index_32), kwargs = {})
#   %sub_386 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_386, %unsqueeze_3122), kwargs = {})
#   %mul_1291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_386, %unsqueeze_3124), kwargs = {})
#   %mul_1292 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1291, %unsqueeze_3126), kwargs = {})
#   %add_1109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1292, %unsqueeze_3128), kwargs = {})
#   %_unsafe_index_33 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1109, [None, None, %unsqueeze_3129, %convert_element_type_909]), kwargs = {})
#   %add_1114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1107, %_unsafe_index_33), kwargs = {})
#   %relu_343 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1114,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 144
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 56)
    x2 = xindex % 56
    y0 = yindex % 18
    y1 = (yindex // 18)
    x4 = xindex
    y5 = yindex
    tmp10 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr10 + (y0 + (18*x4) + (56448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x3
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (y0 + (18*tmp8) + (504*tmp4) + (14112*y1)), xmask & ymask)
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1, 1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = 0.25
    tmp26 = tmp1 * tmp25
    tmp27 = tmp26.to(tl.int32)
    tmp28 = tmp6 * tmp25
    tmp29 = tmp28.to(tl.int32)
    tmp30 = tl.load(in_ptr5 + (y0 + (18*tmp29) + (252*tmp27) + (3528*y1)), xmask & ymask)
    tmp32 = tmp30 - tmp31
    tmp34 = tmp33 + tmp13
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tmp16 / tmp35
    tmp37 = tmp36 * tmp18
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp44 = tmp43 + tmp24
    tmp45 = tmp44 + tmp42
    tmp46 = tl.full([1, 1], 0, tl.int32)
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tl.store(out_ptr2 + (y0 + (18*x4) + (56448*y1)), tmp47, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2v/c2vcsoifcgkiz2rjoq3tcuvfhoj45ksbpe627m54n7qeypic5nkx.py
# Topologically Sorted Source Nodes: [input_266, input_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_266 => add_1126, mul_1308, mul_1309, sub_389
#   input_267 => relu_345
# Graph fragment:
#   %sub_389 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_389, %unsqueeze_3148), kwargs = {})
#   %mul_1308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_389, %unsqueeze_3150), kwargs = {})
#   %mul_1309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1308, %unsqueeze_3152), kwargs = {})
#   %add_1126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1309, %unsqueeze_3154), kwargs = {})
#   %relu_345 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1126,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 18
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


# kernel path: /tmp/torchinductor_sahanp/67/c67g4t7jkl4ccf2mmlrcqckofosghyqqpdqidhygkr6d2nw4svr4.py
# Topologically Sorted Source Nodes: [input_266, input_267, input_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_266 => add_1126, mul_1308, mul_1309, sub_389
#   input_267 => relu_345
#   input_268 => convolution_390
# Graph fragment:
#   %sub_389 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_389, %unsqueeze_3148), kwargs = {})
#   %mul_1308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_389, %unsqueeze_3150), kwargs = {})
#   %mul_1309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1308, %unsqueeze_3152), kwargs = {})
#   %add_1126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1309, %unsqueeze_3154), kwargs = {})
#   %relu_345 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1126,), kwargs = {})
#   %convolution_390 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_345, %arg326_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1296
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 18
    y1 = (yindex // 18)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (18*x2) + (162*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4x/c4xxmgaojewak2vkhq4n4mdxgqpcwjqqy2uiigsftdklwo5mjvft.py
# Topologically Sorted Source Nodes: [input_269, input_271, y_71, y_72, shortcut_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_269 => add_1128, mul_1311, mul_1312, sub_390
#   input_271 => add_1130, mul_1314, mul_1315, sub_391
#   shortcut_30 => relu_346
#   y_71 => add_1131
#   y_72 => add_1132
# Graph fragment:
#   %sub_390 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_390, %unsqueeze_3156), kwargs = {})
#   %mul_1311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_390, %unsqueeze_3158), kwargs = {})
#   %mul_1312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1311, %unsqueeze_3160), kwargs = {})
#   %add_1128 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1312, %unsqueeze_3162), kwargs = {})
#   %sub_391 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_391, %unsqueeze_3164), kwargs = {})
#   %mul_1314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_391, %unsqueeze_3166), kwargs = {})
#   %mul_1315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1314, %unsqueeze_3168), kwargs = {})
#   %add_1130 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1315, %unsqueeze_3170), kwargs = {})
#   %add_1131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1128, %add_1130), kwargs = {})
#   %add_1132 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1131, %relu_342), kwargs = {})
#   %relu_346 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1132,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
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
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_out_ptr1 + (x2), xmask)
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
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(in_out_ptr1 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/j5/cj5i5dlbtpi3rqdo4c4lzwl322orkup5csg544m56tirpwojln4i.py
# Topologically Sorted Source Nodes: [x_1316, x_1317, x_1318], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_1316 => add_1454, mul_1686, mul_1687, sub_503
#   x_1317 => add_1455
#   x_1318 => relu_449
# Graph fragment:
#   %sub_503 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_503, %unsqueeze_4069), kwargs = {})
#   %mul_1686 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_503, %unsqueeze_4071), kwargs = {})
#   %mul_1687 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1686, %unsqueeze_4073), kwargs = {})
#   %add_1454 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1687, %unsqueeze_4075), kwargs = {})
#   %add_1455 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1454, %relu_430), kwargs = {})
#   %relu_449 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1455,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
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
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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


# kernel path: /tmp/torchinductor_sahanp/ix/cixwphfu3nhfphyewcqrbtkelknex6tflbw2y6b24utd455aymo5.py
# Topologically Sorted Source Nodes: [input_326], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_326 => convolution_485
# Graph fragment:
#   %convolution_485 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_430, %arg801_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_31 = async_compile.triton('triton_poi_fused_convolution_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (72*x2) + (648*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kz/ckze6jb73ecnkxeirjag5qrow3rp6o6fsqw46yye6zjf3adoocsm.py
# Topologically Sorted Source Nodes: [input_327, input_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_327 => add_1410, mul_1632, mul_1633, sub_485
#   input_328 => relu_431
# Graph fragment:
#   %sub_485 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_485, %unsqueeze_3925), kwargs = {})
#   %mul_1632 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_485, %unsqueeze_3927), kwargs = {})
#   %mul_1633 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1632, %unsqueeze_3929), kwargs = {})
#   %add_1410 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1633, %unsqueeze_3931), kwargs = {})
#   %relu_431 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1410,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56448
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


# kernel path: /tmp/torchinductor_sahanp/cg/ccgb4p4d2pbx43eoemj2neys7pg3rkm6kq76y7xptxxco6zaukuv.py
# Topologically Sorted Source Nodes: [x_1340], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_1340 => convolution_510
# Graph fragment:
#   %convolution_510 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_431, %arg926_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_33 = async_compile.triton('triton_poi_fused_convolution_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_33(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/hp/chpqnzc4rcjcvrkjszlnnepcvumuixhmksswyjvib463duvfkcwn.py
# Topologically Sorted Source Nodes: [x_1344, x_1345, x_1346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_1344 => add_1474, mul_1710, mul_1711, sub_511
#   x_1345 => add_1475
#   x_1346 => relu_457
# Graph fragment:
#   %sub_511 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_511, %unsqueeze_4133), kwargs = {})
#   %mul_1710 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_511, %unsqueeze_4135), kwargs = {})
#   %mul_1711 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1710, %unsqueeze_4137), kwargs = {})
#   %add_1474 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1711, %unsqueeze_4139), kwargs = {})
#   %add_1475 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1474, %relu_431), kwargs = {})
#   %relu_457 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1475,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
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


# kernel path: /tmp/torchinductor_sahanp/bw/cbwifzwpwfvvgu56gyjakdrzwexfnwwveu2nknpflf2x4uumag2h.py
# Topologically Sorted Source Nodes: [input_344, input_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_344 => add_1523, mul_1762, mul_1763, sub_523
#   input_345 => _unsafe_index_48
# Graph fragment:
#   %sub_523 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_523, %unsqueeze_4233), kwargs = {})
#   %mul_1762 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_523, %unsqueeze_4235), kwargs = {})
#   %mul_1763 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1762, %unsqueeze_4237), kwargs = {})
#   %add_1523 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1763, %unsqueeze_4239), kwargs = {})
#   %_unsafe_index_48 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1523, [None, None, %unsqueeze_4240, %convert_element_type_1243]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x2 = (xindex // 784) % 36
    x3 = (xindex // 28224)
    x5 = xindex
    tmp10 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.25
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x2 + (36*tmp8) + (252*tmp4) + (1764*x3)), xmask, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr0 + (x5), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4f/c4fcbtqytwlk5fxq2gsnlaqrn7at5xuvntqjvft4cj45ele4dhjo.py
# Topologically Sorted Source Nodes: [x_1309, x_1310, x_1311, input_339, y_94, y_95, y_96, shortcut_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_339 => add_1513, mul_1752, mul_1753, sub_521
#   shortcut_41 => relu_465
#   x_1309 => add_1449, mul_1680, mul_1681, sub_501
#   x_1310 => add_1450
#   x_1311 => relu_447
#   y_94 => add_1514
#   y_95 => add_1521
#   y_96 => add_1528
# Graph fragment:
#   %sub_501 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_501, %unsqueeze_4053), kwargs = {})
#   %mul_1680 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_501, %unsqueeze_4055), kwargs = {})
#   %mul_1681 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1680, %unsqueeze_4057), kwargs = {})
#   %add_1449 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1681, %unsqueeze_4059), kwargs = {})
#   %add_1450 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1449, %relu_445), kwargs = {})
#   %relu_447 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_1450,), kwargs = {})
#   %sub_521 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_521, %unsqueeze_4216), kwargs = {})
#   %mul_1752 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_521, %unsqueeze_4218), kwargs = {})
#   %mul_1753 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1752, %unsqueeze_4220), kwargs = {})
#   %add_1513 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1753, %unsqueeze_4222), kwargs = {})
#   %add_1514 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1513, %relu_447), kwargs = {})
#   %add_1521 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1514, %_unsafe_index_47), kwargs = {})
#   %add_1528 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1521, %_unsafe_index_48), kwargs = {})
#   %relu_465 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1528,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 784
    y3 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x1 + (36*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x1 + (36*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x1 + (36*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (y2 + (784*x1) + (28224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr10 + (y2 + (784*x1) + (28224*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp22 = tmp20 - tmp21
    tmp24 = tmp23 + tmp4
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp7 / tmp25
    tmp27 = tmp26 * tmp9
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp32 + tmp19
    tmp35 = tmp33 + tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = triton_helpers.maximum(tmp18, tmp37)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + (36*y0)), tmp19, xmask & ymask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x1 + (36*y0)), tmp38, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ak/caka3neyhfxttb4fjtvfzaocagginya2eow7pnzthl2uc4e6pl5x.py
# Topologically Sorted Source Nodes: [input_330, input_331, y_91, input_333, input_334, y_92, input_336, input_337, y_93, shortcut_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_330 => add_1492, mul_1731, mul_1732, sub_518
#   input_331 => _unsafe_index_44
#   input_333 => add_1499, mul_1738, mul_1739, sub_519
#   input_334 => _unsafe_index_45
#   input_336 => add_1506, mul_1745, mul_1746, sub_520
#   input_337 => _unsafe_index_46
#   shortcut_40 => relu_464
#   y_91 => add_1497
#   y_92 => add_1504
#   y_93 => add_1511
# Graph fragment:
#   %sub_518 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_518, %unsqueeze_4189), kwargs = {})
#   %mul_1731 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_518, %unsqueeze_4191), kwargs = {})
#   %mul_1732 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1731, %unsqueeze_4193), kwargs = {})
#   %add_1492 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1732, %unsqueeze_4195), kwargs = {})
#   %_unsafe_index_44 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1492, [None, None, %unsqueeze_4196, %convert_element_type_1217]), kwargs = {})
#   %add_1497 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_439, %_unsafe_index_44), kwargs = {})
#   %sub_519 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_519, %unsqueeze_4198), kwargs = {})
#   %mul_1738 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_519, %unsqueeze_4200), kwargs = {})
#   %mul_1739 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1738, %unsqueeze_4202), kwargs = {})
#   %add_1499 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1739, %unsqueeze_4204), kwargs = {})
#   %_unsafe_index_45 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1499, [None, None, %unsqueeze_4205, %convert_element_type_1223]), kwargs = {})
#   %add_1504 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1497, %_unsafe_index_45), kwargs = {})
#   %sub_520 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_520, %unsqueeze_4207), kwargs = {})
#   %mul_1745 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_520, %unsqueeze_4209), kwargs = {})
#   %mul_1746 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1745, %unsqueeze_4211), kwargs = {})
#   %add_1506 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1746, %unsqueeze_4213), kwargs = {})
#   %_unsafe_index_46 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1506, [None, None, %unsqueeze_4214, %convert_element_type_1229]), kwargs = {})
#   %add_1511 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1504, %_unsafe_index_46), kwargs = {})
#   %relu_464 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1511,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 144
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 56)
    x2 = xindex % 56
    y0 = yindex % 18
    y1 = (yindex // 18)
    x4 = xindex
    y5 = yindex
    tmp10 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr11 + (y0), ymask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr12 + (y0), ymask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr13 + (y0), ymask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr14 + (y0), ymask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr15 + (y0 + (18*x4) + (56448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x3
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (y0 + (18*tmp8) + (504*tmp4) + (14112*y1)), xmask & ymask)
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1, 1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = 0.25
    tmp26 = tmp1 * tmp25
    tmp27 = tmp26.to(tl.int32)
    tmp28 = tmp6 * tmp25
    tmp29 = tmp28.to(tl.int32)
    tmp30 = tl.load(in_ptr5 + (y0 + (18*tmp29) + (252*tmp27) + (3528*y1)), xmask & ymask)
    tmp32 = tmp30 - tmp31
    tmp34 = tmp33 + tmp13
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tmp16 / tmp35
    tmp37 = tmp36 * tmp18
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = 0.125
    tmp44 = tmp1 * tmp43
    tmp45 = tmp44.to(tl.int32)
    tmp46 = tmp6 * tmp43
    tmp47 = tmp46.to(tl.int32)
    tmp48 = tl.load(in_ptr10 + (y0 + (18*tmp47) + (126*tmp45) + (882*y1)), xmask & ymask)
    tmp50 = tmp48 - tmp49
    tmp52 = tmp51 + tmp13
    tmp53 = libdevice.sqrt(tmp52)
    tmp54 = tmp16 / tmp53
    tmp55 = tmp54 * tmp18
    tmp56 = tmp50 * tmp55
    tmp58 = tmp56 * tmp57
    tmp60 = tmp58 + tmp59
    tmp62 = tmp61 + tmp24
    tmp63 = tmp62 + tmp42
    tmp64 = tmp63 + tmp60
    tmp65 = tl.full([1, 1], 0, tl.int32)
    tmp66 = triton_helpers.maximum(tmp65, tmp64)
    tl.store(out_ptr3 + (y0 + (18*x4) + (56448*y1)), tmp66, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oz/coz3iqc75j75ku4fu6nx7idsgab7gaopviq7fz4wfimvddvui5fe.py
# Topologically Sorted Source Nodes: [input_354, input_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_354 => add_1538, mul_1778, mul_1779, sub_527
#   input_355 => _unsafe_index_49
# Graph fragment:
#   %sub_527 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_527, %unsqueeze_4266), kwargs = {})
#   %mul_1778 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_527, %unsqueeze_4268), kwargs = {})
#   %mul_1779 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1778, %unsqueeze_4270), kwargs = {})
#   %add_1538 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1779, %unsqueeze_4272), kwargs = {})
#   %_unsafe_index_49 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1538, [None, None, %unsqueeze_4273, %convert_element_type_1255]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x2 = (xindex // 196) % 72
    x3 = (xindex // 14112)
    x5 = xindex
    tmp10 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x2 + (72*tmp8) + (504*tmp4) + (3528*x3)), xmask, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr0 + (x5), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/65/c65kjafvlpws4uuyzhxv4f4o2vx6ibki2j3jlklog2jbwwjo4syg.py
# Topologically Sorted Source Nodes: [input_350, input_352, y_97, y_98, y_99, shortcut_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_350 => add_1532, mul_1772, mul_1773, sub_525
#   input_352 => add_1534, mul_1775, mul_1776, sub_526
#   shortcut_42 => relu_467
#   y_97 => add_1535
#   y_98 => add_1536
#   y_99 => add_1543
# Graph fragment:
#   %sub_525 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_525, %unsqueeze_4250), kwargs = {})
#   %mul_1772 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_525, %unsqueeze_4252), kwargs = {})
#   %mul_1773 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1772, %unsqueeze_4254), kwargs = {})
#   %add_1532 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1773, %unsqueeze_4256), kwargs = {})
#   %sub_526 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_526, %unsqueeze_4258), kwargs = {})
#   %mul_1775 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_526, %unsqueeze_4260), kwargs = {})
#   %mul_1776 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1775, %unsqueeze_4262), kwargs = {})
#   %add_1534 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1776, %unsqueeze_4264), kwargs = {})
#   %add_1535 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1532, %add_1534), kwargs = {})
#   %add_1536 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1535, %relu_455), kwargs = {})
#   %add_1543 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1536, %_unsafe_index_49), kwargs = {})
#   %relu_467 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1543,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 72
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 196
    y3 = (yindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (x1 + (72*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1 + (72*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x1 + (72*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr10 + (y2 + (196*x1) + (14112*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1, 1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(out_ptr0 + (x1 + (72*y0)), tmp35, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/u2/cu2agup3hbaxpt23och6tyqgc7z3bmedoc7adn4ilyxej2p7oubc.py
# Topologically Sorted Source Nodes: [input_360, input_361], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_360 => add_1547, mul_1788, mul_1789, sub_529
#   input_361 => relu_469
# Graph fragment:
#   %sub_529 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_529, %unsqueeze_4283), kwargs = {})
#   %mul_1788 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_529, %unsqueeze_4285), kwargs = {})
#   %mul_1789 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1788, %unsqueeze_4287), kwargs = {})
#   %add_1547 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1789, %unsqueeze_4289), kwargs = {})
#   %relu_469 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1547,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 18
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


# kernel path: /tmp/torchinductor_sahanp/ac/cac3qgrcpsi6ozoau5cu6pbppw7xmtmyzuibel6qfhnvg6vj25yj.py
# Topologically Sorted Source Nodes: [input_360, input_361, input_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_360 => add_1547, mul_1788, mul_1789, sub_529
#   input_361 => relu_469
#   input_362 => convolution_530
# Graph fragment:
#   %sub_529 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_529, %unsqueeze_4283), kwargs = {})
#   %mul_1788 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_529, %unsqueeze_4285), kwargs = {})
#   %mul_1789 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1788, %unsqueeze_4287), kwargs = {})
#   %add_1547 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1789, %unsqueeze_4289), kwargs = {})
#   %relu_469 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1547,), kwargs = {})
#   %convolution_530 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_469, %arg1026_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2592
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 18
    y1 = (yindex // 18)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (18*x2) + (162*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/if/cif5vnd3ttfz2nqgajt5vk6wquqcryuho7ecoa4ncgzv7tbcqovl.py
# Topologically Sorted Source Nodes: [input_365, input_366], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_365 => add_1551, mul_1794, mul_1795, sub_531
#   input_366 => relu_470
# Graph fragment:
#   %sub_531 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_531, %unsqueeze_4299), kwargs = {})
#   %mul_1794 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_531, %unsqueeze_4301), kwargs = {})
#   %mul_1795 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1794, %unsqueeze_4303), kwargs = {})
#   %add_1551 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1795, %unsqueeze_4305), kwargs = {})
#   %relu_470 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1551,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 36
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


# kernel path: /tmp/torchinductor_sahanp/hx/chxevfqwzk765e2sfai3brgj4vl3tyq5pyh3txdyvptfwj3op4c4.py
# Topologically Sorted Source Nodes: [input_365, input_366, input_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_365 => add_1551, mul_1794, mul_1795, sub_531
#   input_366 => relu_470
#   input_367 => convolution_532
# Graph fragment:
#   %sub_531 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_531, %unsqueeze_4299), kwargs = {})
#   %mul_1794 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_531, %unsqueeze_4301), kwargs = {})
#   %mul_1795 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1794, %unsqueeze_4303), kwargs = {})
#   %add_1551 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1795, %unsqueeze_4305), kwargs = {})
#   %relu_470 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1551,), kwargs = {})
#   %convolution_532 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_470, %arg1036_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_43(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5184
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 36
    y1 = (yindex // 36)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (36*x2) + (324*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eu/ceu3vt6frffuh7csji4djtebrhtbfgsr77kwxo3ynqxnk7mfsfu4.py
# Topologically Sorted Source Nodes: [input_363, input_368, y_100, input_370, y_101, y_102, shortcut_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_363 => add_1549, mul_1791, mul_1792, sub_530
#   input_368 => add_1553, mul_1797, mul_1798, sub_532
#   input_370 => add_1556, mul_1800, mul_1801, sub_533
#   shortcut_43 => relu_471
#   y_100 => add_1554
#   y_101 => add_1557
#   y_102 => add_1558
# Graph fragment:
#   %sub_530 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_530, %unsqueeze_4291), kwargs = {})
#   %mul_1791 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_530, %unsqueeze_4293), kwargs = {})
#   %mul_1792 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1791, %unsqueeze_4295), kwargs = {})
#   %add_1549 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1792, %unsqueeze_4297), kwargs = {})
#   %sub_532 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_532, %unsqueeze_4307), kwargs = {})
#   %mul_1797 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_532, %unsqueeze_4309), kwargs = {})
#   %mul_1798 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1797, %unsqueeze_4311), kwargs = {})
#   %add_1553 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1798, %unsqueeze_4313), kwargs = {})
#   %add_1554 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1549, %add_1553), kwargs = {})
#   %sub_533 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_533, %unsqueeze_4315), kwargs = {})
#   %mul_1800 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_533, %unsqueeze_4317), kwargs = {})
#   %mul_1801 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1800, %unsqueeze_4319), kwargs = {})
#   %add_1556 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1801, %unsqueeze_4321), kwargs = {})
#   %add_1557 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1554, %add_1556), kwargs = {})
#   %add_1558 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1557, %relu_463), kwargs = {})
#   %relu_471 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1558,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56448
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
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask)
    tmp31 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr13 + (x0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_out_ptr1 + (x2), xmask)
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
    tmp32 = tmp30 - tmp31
    tmp34 = tmp33 + tmp4
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tmp7 / tmp35
    tmp37 = tmp36 * tmp9
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp29 + tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tl.full([1], 0, tl.int32)
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tl.store(in_out_ptr1 + (x2), tmp47, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5m/c5m35bc4zrbc6s6dksyiop5hv6trcvexnea6i5qaas2ie74lyn3l.py
# Topologically Sorted Source Nodes: [input_447, input_452, y_124, input_454, y_125, y_126, shortcut_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_447 => add_1845, mul_2127, mul_2128, sub_626
#   input_452 => add_1849, mul_2133, mul_2134, sub_628
#   input_454 => add_1852, mul_2136, mul_2137, sub_629
#   shortcut_51 => relu_551
#   y_124 => add_1850
#   y_125 => add_1853
#   y_126 => add_1854
# Graph fragment:
#   %sub_626 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_626, %unsqueeze_5071), kwargs = {})
#   %mul_2127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_626, %unsqueeze_5073), kwargs = {})
#   %mul_2128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2127, %unsqueeze_5075), kwargs = {})
#   %add_1845 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2128, %unsqueeze_5077), kwargs = {})
#   %sub_628 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_628, %unsqueeze_5087), kwargs = {})
#   %mul_2133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_628, %unsqueeze_5089), kwargs = {})
#   %mul_2134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2133, %unsqueeze_5091), kwargs = {})
#   %add_1849 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2134, %unsqueeze_5093), kwargs = {})
#   %add_1850 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1845, %add_1849), kwargs = {})
#   %sub_629 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_629, %unsqueeze_5095), kwargs = {})
#   %mul_2136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_629, %unsqueeze_5097), kwargs = {})
#   %mul_2137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2136, %unsqueeze_5099), kwargs = {})
#   %add_1852 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2137, %unsqueeze_5101), kwargs = {})
#   %add_1853 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1850, %add_1852), kwargs = {})
#   %add_1854 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1853, %relu_543), kwargs = {})
#   %relu_551 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1854,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56448
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
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr13 + (x2), xmask)
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
    tmp32 = tmp30 - tmp31
    tmp34 = tmp33 + tmp4
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tmp7 / tmp35
    tmp37 = tmp36 * tmp9
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp29 + tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tl.full([1], 0, tl.int32)
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tl.store(in_out_ptr1 + (x2), tmp47, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rc/crctiyvzaqtbknq64lcsbiei22jtvrzxngo5foqw7qjk3uqwumqq.py
# Topologically Sorted Source Nodes: [x_1623, x_1624], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1623 => add_1889, mul_2181, mul_2182, sub_644
#   x_1624 => relu_563
# Graph fragment:
#   %sub_644 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_644, %unsqueeze_5215), kwargs = {})
#   %mul_2181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_644, %unsqueeze_5217), kwargs = {})
#   %mul_2182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2181, %unsqueeze_5219), kwargs = {})
#   %add_1889 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2182, %unsqueeze_5221), kwargs = {})
#   %relu_563 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1889,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
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


# kernel path: /tmp/torchinductor_sahanp/ax/caxcleuqzwunujmoynq2vowvijkb6hafoh7gxlzp2jquo556drj2.py
# Topologically Sorted Source Nodes: [x_1623, x_1624, x_1625], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_1623 => add_1889, mul_2181, mul_2182, sub_644
#   x_1624 => relu_563
#   x_1625 => convolution_645
# Graph fragment:
#   %sub_644 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_644, %unsqueeze_5215), kwargs = {})
#   %mul_2181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_644, %unsqueeze_5217), kwargs = {})
#   %mul_2182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2181, %unsqueeze_5219), kwargs = {})
#   %add_1889 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2182, %unsqueeze_5221), kwargs = {})
#   %relu_563 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1889,), kwargs = {})
#   %convolution_645 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_563, %arg1603_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rm/crmnezoppklk5pxdrun7wcoxsbpu3vqe43yvbrr2dg5dsdznqy6i.py
# Topologically Sorted Source Nodes: [x_1613, x_1614], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1613 => add_1877, mul_2166, mul_2167, sub_639
#   x_1614 => relu_559
# Graph fragment:
#   %sub_639 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_639, %unsqueeze_5175), kwargs = {})
#   %mul_2166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_639, %unsqueeze_5177), kwargs = {})
#   %mul_2167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2166, %unsqueeze_5179), kwargs = {})
#   %add_1877 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2167, %unsqueeze_5181), kwargs = {})
#   %relu_559 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1877,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
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


# kernel path: /tmp/torchinductor_sahanp/qg/cqgzxlcabmojx6ucpu4zq5ub2bbux6ouull6kjd4pvf5mwtsko2g.py
# Topologically Sorted Source Nodes: [x_1613, x_1614, x_1615], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_1613 => add_1877, mul_2166, mul_2167, sub_639
#   x_1614 => relu_559
#   x_1615 => convolution_640
# Graph fragment:
#   %sub_639 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_639, %unsqueeze_5175), kwargs = {})
#   %mul_2166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_639, %unsqueeze_5177), kwargs = {})
#   %mul_2167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2166, %unsqueeze_5179), kwargs = {})
#   %add_1877 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2167, %unsqueeze_5181), kwargs = {})
#   %relu_559 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1877,), kwargs = {})
#   %convolution_640 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_559, %arg1577_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/oy/coyt5cmk7hdmeqwyd3ynzu6nmmmbjictydgsgoa7isymbhm6du37.py
# Topologically Sorted Source Nodes: [x_1603, x_1604], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1603 => add_1865, mul_2151, mul_2152, sub_634
#   x_1604 => relu_555
# Graph fragment:
#   %sub_634 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_634, %unsqueeze_5135), kwargs = {})
#   %mul_2151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_634, %unsqueeze_5137), kwargs = {})
#   %mul_2152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2151, %unsqueeze_5139), kwargs = {})
#   %add_1865 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2152, %unsqueeze_5141), kwargs = {})
#   %relu_555 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1865,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
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


# kernel path: /tmp/torchinductor_sahanp/k4/ck47buhecu7fohgcpt67bxj44y54q4urjaj3pujbgjx3tp7xjrwp.py
# Topologically Sorted Source Nodes: [input_414, input_415, y_115, input_417, input_418, y_116, input_420, input_421, y_117, shortcut_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_414 => add_1788, mul_2067, mul_2068, sub_614
#   input_415 => _unsafe_index_56
#   input_417 => add_1795, mul_2074, mul_2075, sub_615
#   input_418 => _unsafe_index_57
#   input_420 => add_1802, mul_2081, mul_2082, sub_616
#   input_421 => _unsafe_index_58
#   shortcut_48 => relu_544
#   y_115 => add_1793
#   y_116 => add_1800
#   y_117 => add_1807
# Graph fragment:
#   %sub_614 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_614, %unsqueeze_4969), kwargs = {})
#   %mul_2067 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_614, %unsqueeze_4971), kwargs = {})
#   %mul_2068 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2067, %unsqueeze_4973), kwargs = {})
#   %add_1788 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2068, %unsqueeze_4975), kwargs = {})
#   %_unsafe_index_56 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1788, [None, None, %unsqueeze_4976, %convert_element_type_1457]), kwargs = {})
#   %add_1793 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_519, %_unsafe_index_56), kwargs = {})
#   %sub_615 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_615, %unsqueeze_4978), kwargs = {})
#   %mul_2074 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_615, %unsqueeze_4980), kwargs = {})
#   %mul_2075 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2074, %unsqueeze_4982), kwargs = {})
#   %add_1795 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2075, %unsqueeze_4984), kwargs = {})
#   %_unsafe_index_57 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1795, [None, None, %unsqueeze_4985, %convert_element_type_1463]), kwargs = {})
#   %add_1800 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1793, %_unsafe_index_57), kwargs = {})
#   %sub_616 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_616, %unsqueeze_4987), kwargs = {})
#   %mul_2081 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_616, %unsqueeze_4989), kwargs = {})
#   %mul_2082 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2081, %unsqueeze_4991), kwargs = {})
#   %add_1802 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2082, %unsqueeze_4993), kwargs = {})
#   %_unsafe_index_58 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_1802, [None, None, %unsqueeze_4994, %convert_element_type_1469]), kwargs = {})
#   %add_1807 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1800, %_unsafe_index_58), kwargs = {})
#   %relu_544 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1807,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 144
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 56)
    x2 = xindex % 56
    y0 = yindex % 18
    y1 = (yindex // 18)
    x4 = xindex
    y5 = yindex
    tmp10 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr11 + (y0), ymask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr12 + (y0), ymask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr13 + (y0), ymask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr14 + (y0), ymask, eviction_policy='evict_last')
    tmp61 = tl.load(in_out_ptr0 + (y0 + (18*x4) + (56448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x3
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (y0 + (18*tmp8) + (504*tmp4) + (14112*y1)), xmask & ymask)
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1, 1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = 0.25
    tmp26 = tmp1 * tmp25
    tmp27 = tmp26.to(tl.int32)
    tmp28 = tmp6 * tmp25
    tmp29 = tmp28.to(tl.int32)
    tmp30 = tl.load(in_ptr5 + (y0 + (18*tmp29) + (252*tmp27) + (3528*y1)), xmask & ymask)
    tmp32 = tmp30 - tmp31
    tmp34 = tmp33 + tmp13
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tmp16 / tmp35
    tmp37 = tmp36 * tmp18
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = 0.125
    tmp44 = tmp1 * tmp43
    tmp45 = tmp44.to(tl.int32)
    tmp46 = tmp6 * tmp43
    tmp47 = tmp46.to(tl.int32)
    tmp48 = tl.load(in_ptr10 + (y0 + (18*tmp47) + (126*tmp45) + (882*y1)), xmask & ymask)
    tmp50 = tmp48 - tmp49
    tmp52 = tmp51 + tmp13
    tmp53 = libdevice.sqrt(tmp52)
    tmp54 = tmp16 / tmp53
    tmp55 = tmp54 * tmp18
    tmp56 = tmp50 * tmp55
    tmp58 = tmp56 * tmp57
    tmp60 = tmp58 + tmp59
    tmp62 = tmp61 + tmp24
    tmp63 = tmp62 + tmp42
    tmp64 = tmp63 + tmp60
    tmp65 = tl.full([1, 1], 0, tl.int32)
    tmp66 = triton_helpers.maximum(tmp65, tmp64)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (18*x4) + (56448*y1)), tmp66, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sr/csr4l5bukemia2ni5zebihfe3vnk6rbmftl2ujvwgfman4vkskyt.py
# Topologically Sorted Source Nodes: [x_1593, x_1594], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1593 => add_1856, mul_2139, mul_2140, sub_630
#   x_1594 => relu_552
# Graph fragment:
#   %sub_630 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_630, %unsqueeze_5103), kwargs = {})
#   %mul_2139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_630, %unsqueeze_5105), kwargs = {})
#   %mul_2140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2139, %unsqueeze_5107), kwargs = {})
#   %add_1856 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2140, %unsqueeze_5109), kwargs = {})
#   %relu_552 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1856,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_52(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/sc/csc2m7hjtgq4a4hovy3o3sgovdrfdkffhvfaev6iquydmodyns2j.py
# Topologically Sorted Source Nodes: [x_1593, x_1594, x_1595], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_1593 => add_1856, mul_2139, mul_2140, sub_630
#   x_1594 => relu_552
#   x_1595 => convolution_631
# Graph fragment:
#   %sub_630 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_630, %unsqueeze_5103), kwargs = {})
#   %mul_2139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_630, %unsqueeze_5105), kwargs = {})
#   %mul_2140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2139, %unsqueeze_5107), kwargs = {})
#   %add_1856 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2140, %unsqueeze_5109), kwargs = {})
#   %relu_552 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1856,), kwargs = {})
#   %convolution_631 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_552, %arg1531_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
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


# kernel path: /tmp/torchinductor_sahanp/65/c65klxbw3k2cniqdw7xgkkuhusjakv3slgzwhjw3t23g3ko2a6qk.py
# Topologically Sorted Source Nodes: [x_1599, input_456, x_1600, x_1601], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_456 => add_1862, mul_2148, mul_2149, sub_633
#   x_1599 => add_1860, mul_2145, mul_2146, sub_632
#   x_1600 => add_1863
#   x_1601 => relu_554
# Graph fragment:
#   %sub_632 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_632, %unsqueeze_5119), kwargs = {})
#   %mul_2145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_632, %unsqueeze_5121), kwargs = {})
#   %mul_2146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2145, %unsqueeze_5123), kwargs = {})
#   %add_1860 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2146, %unsqueeze_5125), kwargs = {})
#   %sub_633 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_633, %unsqueeze_5127), kwargs = {})
#   %mul_2148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_633, %unsqueeze_5129), kwargs = {})
#   %mul_2149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2148, %unsqueeze_5131), kwargs = {})
#   %add_1862 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2149, %unsqueeze_5133), kwargs = {})
#   %add_1863 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1860, %add_1862), kwargs = {})
#   %relu_554 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1863,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_54 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_54', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ey/ceyl2herqb45ydygvompdu5i336e3j3dxaesaeksl74vcr22uaxk.py
# Topologically Sorted Source Nodes: [x_1601, input_459], Original ATen: [aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_459 => convolution_638
#   x_1601 => relu_554
# Graph fragment:
#   %relu_554 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1863,), kwargs = {})
#   %convolution_638 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_554, %arg1566_1, %arg1567_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_relu_55 = async_compile.triton('triton_poi_fused_convolution_relu_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_55(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
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


# kernel path: /tmp/torchinductor_sahanp/ch/cchie5eieim6n6xdhdld2rl5tdsd6tiehe7vygy42vj2kr72kfri.py
# Topologically Sorted Source Nodes: [x_1609, input_458, x_1610, x_1611, x_1601, input_459, input_460, input_461, y_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_458 => add_1871, mul_2160, mul_2161, sub_637
#   input_459 => convolution_638
#   input_460 => add_1874, mul_2163, mul_2164, sub_638
#   input_461 => relu_558
#   x_1601 => relu_554
#   x_1609 => add_1869, mul_2157, mul_2158, sub_636
#   x_1610 => add_1872
#   x_1611 => relu_557
#   y_127 => add_1875
# Graph fragment:
#   %sub_636 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_636, %unsqueeze_5151), kwargs = {})
#   %mul_2157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_636, %unsqueeze_5153), kwargs = {})
#   %mul_2158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2157, %unsqueeze_5155), kwargs = {})
#   %add_1869 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2158, %unsqueeze_5157), kwargs = {})
#   %sub_637 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_637, %unsqueeze_5159), kwargs = {})
#   %mul_2160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_637, %unsqueeze_5161), kwargs = {})
#   %mul_2161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2160, %unsqueeze_5163), kwargs = {})
#   %add_1871 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2161, %unsqueeze_5165), kwargs = {})
#   %add_1872 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1869, %add_1871), kwargs = {})
#   %relu_557 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1872,), kwargs = {})
#   %relu_554 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1863,), kwargs = {})
#   %convolution_638 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_554, %arg1566_1, %arg1567_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_638 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_638, %unsqueeze_5167), kwargs = {})
#   %mul_2163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_638, %unsqueeze_5169), kwargs = {})
#   %mul_2164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2163, %unsqueeze_5171), kwargs = {})
#   %add_1874 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2164, %unsqueeze_5173), kwargs = {})
#   %relu_558 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1874,), kwargs = {})
#   %add_1875 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_557, %relu_558), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_56', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_56', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_56(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
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
    tmp32 = tl.load(in_out_ptr1 + (x2), None)
    tmp33 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
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
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 - tmp35
    tmp38 = tmp37 + tmp4
    tmp39 = libdevice.sqrt(tmp38)
    tmp40 = tmp7 / tmp39
    tmp41 = tmp40 * tmp9
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = triton_helpers.maximum(tmp30, tmp46)
    tmp48 = tmp31 + tmp47
    tl.store(in_out_ptr1 + (x2), tmp48, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6g/c6gutfp5ijygf35rbphyd534uanxbgbgeeldbz436dschuzaizwp.py
# Topologically Sorted Source Nodes: [x_1611, x_1601, input_459, input_460, input_461, y_127, input_464], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_459 => convolution_638
#   input_460 => add_1874, mul_2163, mul_2164, sub_638
#   input_461 => relu_558
#   input_464 => convolution_643
#   x_1601 => relu_554
#   x_1611 => relu_557
#   y_127 => add_1875
# Graph fragment:
#   %relu_557 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1872,), kwargs = {})
#   %relu_554 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1863,), kwargs = {})
#   %convolution_638 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_554, %arg1566_1, %arg1567_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_638 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_638, %unsqueeze_5167), kwargs = {})
#   %mul_2163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_638, %unsqueeze_5169), kwargs = {})
#   %mul_2164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2163, %unsqueeze_5171), kwargs = {})
#   %add_1874 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2164, %unsqueeze_5173), kwargs = {})
#   %relu_558 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1874,), kwargs = {})
#   %add_1875 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_557, %relu_558), kwargs = {})
#   %convolution_643 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1875, %arg1592_1, %arg1593_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_57', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_57(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5s/c5sdugss33hcm5q26ajishggd4oyg3ntpmvmopovaw3x3cq7vuj4.py
# Topologically Sorted Source Nodes: [x_1619, input_463, x_1620, x_1621, x_1611, x_1601, input_459, input_460, input_461, y_127, input_464, input_465, input_466, y_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_459 => convolution_638
#   input_460 => add_1874, mul_2163, mul_2164, sub_638
#   input_461 => relu_558
#   input_463 => add_1883, mul_2175, mul_2176, sub_642
#   input_464 => convolution_643
#   input_465 => add_1886, mul_2178, mul_2179, sub_643
#   input_466 => relu_562
#   x_1601 => relu_554
#   x_1611 => relu_557
#   x_1619 => add_1881, mul_2172, mul_2173, sub_641
#   x_1620 => add_1884
#   x_1621 => relu_561
#   y_127 => add_1875
#   y_128 => add_1887
# Graph fragment:
#   %sub_641 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_641, %unsqueeze_5191), kwargs = {})
#   %mul_2172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_641, %unsqueeze_5193), kwargs = {})
#   %mul_2173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2172, %unsqueeze_5195), kwargs = {})
#   %add_1881 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2173, %unsqueeze_5197), kwargs = {})
#   %sub_642 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_642, %unsqueeze_5199), kwargs = {})
#   %mul_2175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_642, %unsqueeze_5201), kwargs = {})
#   %mul_2176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2175, %unsqueeze_5203), kwargs = {})
#   %add_1883 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2176, %unsqueeze_5205), kwargs = {})
#   %add_1884 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1881, %add_1883), kwargs = {})
#   %relu_561 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1884,), kwargs = {})
#   %relu_557 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1872,), kwargs = {})
#   %relu_554 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1863,), kwargs = {})
#   %convolution_638 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_554, %arg1566_1, %arg1567_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_638 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_638, %unsqueeze_5167), kwargs = {})
#   %mul_2163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_638, %unsqueeze_5169), kwargs = {})
#   %mul_2164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2163, %unsqueeze_5171), kwargs = {})
#   %add_1874 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2164, %unsqueeze_5173), kwargs = {})
#   %relu_558 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1874,), kwargs = {})
#   %add_1875 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_557, %relu_558), kwargs = {})
#   %convolution_643 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1875, %arg1592_1, %arg1593_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_643 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_643, %unsqueeze_5207), kwargs = {})
#   %mul_2178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_643, %unsqueeze_5209), kwargs = {})
#   %mul_2179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2178, %unsqueeze_5211), kwargs = {})
#   %add_1886 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2179, %unsqueeze_5213), kwargs = {})
#   %relu_562 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1886,), kwargs = {})
#   %add_1887 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_561, %relu_562), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_58 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_58', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_58', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_58(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
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
    tmp32 = tl.load(in_out_ptr1 + (x2), None)
    tmp33 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
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
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 - tmp35
    tmp38 = tmp37 + tmp4
    tmp39 = libdevice.sqrt(tmp38)
    tmp40 = tmp7 / tmp39
    tmp41 = tmp40 * tmp9
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = triton_helpers.maximum(tmp30, tmp46)
    tmp48 = tmp31 + tmp47
    tl.store(in_out_ptr1 + (x2), tmp48, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bq/cbqgqlx7on4weoqo3ewj7mkuq7vf2krsig6cwlnfmlndrtg3irid.py
# Topologically Sorted Source Nodes: [x_1621, x_1611, x_1601, input_459, input_460, input_461, y_127, input_464, input_465, input_466, y_128, input_469], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_459 => convolution_638
#   input_460 => add_1874, mul_2163, mul_2164, sub_638
#   input_461 => relu_558
#   input_464 => convolution_643
#   input_465 => add_1886, mul_2178, mul_2179, sub_643
#   input_466 => relu_562
#   input_469 => convolution_648
#   x_1601 => relu_554
#   x_1611 => relu_557
#   x_1621 => relu_561
#   y_127 => add_1875
#   y_128 => add_1887
# Graph fragment:
#   %relu_561 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1884,), kwargs = {})
#   %relu_557 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1872,), kwargs = {})
#   %relu_554 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1863,), kwargs = {})
#   %convolution_638 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_554, %arg1566_1, %arg1567_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_638 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_638, %unsqueeze_5167), kwargs = {})
#   %mul_2163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_638, %unsqueeze_5169), kwargs = {})
#   %mul_2164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2163, %unsqueeze_5171), kwargs = {})
#   %add_1874 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2164, %unsqueeze_5173), kwargs = {})
#   %relu_558 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1874,), kwargs = {})
#   %add_1875 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_557, %relu_558), kwargs = {})
#   %convolution_643 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1875, %arg1592_1, %arg1593_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_643 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_643, %unsqueeze_5207), kwargs = {})
#   %mul_2178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_643, %unsqueeze_5209), kwargs = {})
#   %mul_2179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2178, %unsqueeze_5211), kwargs = {})
#   %add_1886 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2179, %unsqueeze_5213), kwargs = {})
#   %relu_562 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1886,), kwargs = {})
#   %add_1887 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_561, %relu_562), kwargs = {})
#   %convolution_648 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1887, %arg1618_1, %arg1619_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_59 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_59', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_59(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (4608*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yg/cygflkxhu4pwlfgknmioiboznrcldakuaw4stm25vxoyezm6sasw.py
# Topologically Sorted Source Nodes: [x_1629, input_468, x_1630, x_1631, x_1621, x_1611, x_1601, input_459, input_460, input_461, y_127, input_464, input_465, input_466, y_128, input_469, input_470, input_471, y_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_459 => convolution_638
#   input_460 => add_1874, mul_2163, mul_2164, sub_638
#   input_461 => relu_558
#   input_464 => convolution_643
#   input_465 => add_1886, mul_2178, mul_2179, sub_643
#   input_466 => relu_562
#   input_468 => add_1895, mul_2190, mul_2191, sub_647
#   input_469 => convolution_648
#   input_470 => add_1898, mul_2193, mul_2194, sub_648
#   input_471 => relu_566
#   x_1601 => relu_554
#   x_1611 => relu_557
#   x_1621 => relu_561
#   x_1629 => add_1893, mul_2187, mul_2188, sub_646
#   x_1630 => add_1896
#   x_1631 => relu_565
#   y_127 => add_1875
#   y_128 => add_1887
#   y_129 => add_1899
# Graph fragment:
#   %sub_646 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_646, %unsqueeze_5231), kwargs = {})
#   %mul_2187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_646, %unsqueeze_5233), kwargs = {})
#   %mul_2188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2187, %unsqueeze_5235), kwargs = {})
#   %add_1893 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2188, %unsqueeze_5237), kwargs = {})
#   %sub_647 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_647, %unsqueeze_5239), kwargs = {})
#   %mul_2190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_647, %unsqueeze_5241), kwargs = {})
#   %mul_2191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2190, %unsqueeze_5243), kwargs = {})
#   %add_1895 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2191, %unsqueeze_5245), kwargs = {})
#   %add_1896 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1893, %add_1895), kwargs = {})
#   %relu_565 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1896,), kwargs = {})
#   %relu_561 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1884,), kwargs = {})
#   %relu_557 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1872,), kwargs = {})
#   %relu_554 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1863,), kwargs = {})
#   %convolution_638 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_554, %arg1566_1, %arg1567_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_638 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_638, %unsqueeze_5167), kwargs = {})
#   %mul_2163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_638, %unsqueeze_5169), kwargs = {})
#   %mul_2164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2163, %unsqueeze_5171), kwargs = {})
#   %add_1874 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2164, %unsqueeze_5173), kwargs = {})
#   %relu_558 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1874,), kwargs = {})
#   %add_1875 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_557, %relu_558), kwargs = {})
#   %convolution_643 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1875, %arg1592_1, %arg1593_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_643 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_643, %unsqueeze_5207), kwargs = {})
#   %mul_2178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_643, %unsqueeze_5209), kwargs = {})
#   %mul_2179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2178, %unsqueeze_5211), kwargs = {})
#   %add_1886 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2179, %unsqueeze_5213), kwargs = {})
#   %relu_562 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1886,), kwargs = {})
#   %add_1887 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_561, %relu_562), kwargs = {})
#   %convolution_648 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1887, %arg1618_1, %arg1619_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_648 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_648, %unsqueeze_5247), kwargs = {})
#   %mul_2193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_648, %unsqueeze_5249), kwargs = {})
#   %mul_2194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2193, %unsqueeze_5251), kwargs = {})
#   %add_1898 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2194, %unsqueeze_5253), kwargs = {})
#   %relu_566 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1898,), kwargs = {})
#   %add_1899 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_565, %relu_566), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_60 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_60', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_60', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_60(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_out_ptr1 + (x2), None)
    tmp33 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
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
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 - tmp35
    tmp38 = tmp37 + tmp4
    tmp39 = libdevice.sqrt(tmp38)
    tmp40 = tmp7 / tmp39
    tmp41 = tmp40 * tmp9
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = triton_helpers.maximum(tmp30, tmp46)
    tmp48 = tmp31 + tmp47
    tl.store(in_out_ptr1 + (x2), tmp48, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yp/cypu6g6fmz5ebt2vtkirn7bnfuin34oluof2him6bn735pgvuluh.py
# Topologically Sorted Source Nodes: [x_1631, x_1621, x_1611, x_1601, input_459, input_460, input_461, y_127, input_464, input_465, input_466, y_128, input_469, input_470, input_471, y_129, input_472, input_473, input_474, x_1632], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
# Source node to ATen node mapping:
#   input_459 => convolution_638
#   input_460 => add_1874, mul_2163, mul_2164, sub_638
#   input_461 => relu_558
#   input_464 => convolution_643
#   input_465 => add_1886, mul_2178, mul_2179, sub_643
#   input_466 => relu_562
#   input_469 => convolution_648
#   input_470 => add_1898, mul_2193, mul_2194, sub_648
#   input_471 => relu_566
#   input_472 => convolution_649
#   input_473 => add_1901, mul_2196, mul_2197, sub_649
#   input_474 => relu_567
#   x_1601 => relu_554
#   x_1611 => relu_557
#   x_1621 => relu_561
#   x_1631 => relu_565
#   x_1632 => mean_1
#   y_127 => add_1875
#   y_128 => add_1887
#   y_129 => add_1899
# Graph fragment:
#   %relu_565 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1896,), kwargs = {})
#   %relu_561 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1884,), kwargs = {})
#   %relu_557 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1872,), kwargs = {})
#   %relu_554 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1863,), kwargs = {})
#   %convolution_638 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_554, %arg1566_1, %arg1567_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_638 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_638, %unsqueeze_5167), kwargs = {})
#   %mul_2163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_638, %unsqueeze_5169), kwargs = {})
#   %mul_2164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2163, %unsqueeze_5171), kwargs = {})
#   %add_1874 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2164, %unsqueeze_5173), kwargs = {})
#   %relu_558 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1874,), kwargs = {})
#   %add_1875 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_557, %relu_558), kwargs = {})
#   %convolution_643 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1875, %arg1592_1, %arg1593_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_643 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_643, %unsqueeze_5207), kwargs = {})
#   %mul_2178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_643, %unsqueeze_5209), kwargs = {})
#   %mul_2179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2178, %unsqueeze_5211), kwargs = {})
#   %add_1886 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2179, %unsqueeze_5213), kwargs = {})
#   %relu_562 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1886,), kwargs = {})
#   %add_1887 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_561, %relu_562), kwargs = {})
#   %convolution_648 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1887, %arg1618_1, %arg1619_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_648 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_648, %unsqueeze_5247), kwargs = {})
#   %mul_2193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_648, %unsqueeze_5249), kwargs = {})
#   %mul_2194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2193, %unsqueeze_5251), kwargs = {})
#   %add_1898 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2194, %unsqueeze_5253), kwargs = {})
#   %relu_566 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1898,), kwargs = {})
#   %add_1899 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_565, %relu_566), kwargs = {})
#   %convolution_649 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1899, %arg1624_1, %arg1625_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_649 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_649, %unsqueeze_5255), kwargs = {})
#   %mul_2196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_649, %unsqueeze_5257), kwargs = {})
#   %mul_2197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2196, %unsqueeze_5259), kwargs = {})
#   %add_1901 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2197, %unsqueeze_5261), kwargs = {})
#   %relu_567 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1901,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_567, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_61 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_61', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (100352*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tl.full([1, 1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = 49.0
    tmp25 = tmp23 / tmp24
    tl.store(out_ptr1 + (x3), tmp25, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1, arg1180_1, arg1181_1, arg1182_1, arg1183_1, arg1184_1, arg1185_1, arg1186_1, arg1187_1, arg1188_1, arg1189_1, arg1190_1, arg1191_1, arg1192_1, arg1193_1, arg1194_1, arg1195_1, arg1196_1, arg1197_1, arg1198_1, arg1199_1, arg1200_1, arg1201_1, arg1202_1, arg1203_1, arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1234_1, arg1235_1, arg1236_1, arg1237_1, arg1238_1, arg1239_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1276_1, arg1277_1, arg1278_1, arg1279_1, arg1280_1, arg1281_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1318_1, arg1319_1, arg1320_1, arg1321_1, arg1322_1, arg1323_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1360_1, arg1361_1, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1, arg1380_1, arg1381_1, arg1382_1, arg1383_1, arg1384_1, arg1385_1, arg1386_1, arg1387_1, arg1388_1, arg1389_1, arg1390_1, arg1391_1, arg1392_1, arg1393_1, arg1394_1, arg1395_1, arg1396_1, arg1397_1, arg1398_1, arg1399_1, arg1400_1, arg1401_1, arg1402_1, arg1403_1, arg1404_1, arg1405_1, arg1406_1, arg1407_1, arg1408_1, arg1409_1, arg1410_1, arg1411_1, arg1412_1, arg1413_1, arg1414_1, arg1415_1, arg1416_1, arg1417_1, arg1418_1, arg1419_1, arg1420_1, arg1421_1, arg1422_1, arg1423_1, arg1424_1, arg1425_1, arg1426_1, arg1427_1, arg1428_1, arg1429_1, arg1430_1, arg1431_1, arg1432_1, arg1433_1, arg1434_1, arg1435_1, arg1436_1, arg1437_1, arg1438_1, arg1439_1, arg1440_1, arg1441_1, arg1442_1, arg1443_1, arg1444_1, arg1445_1, arg1446_1, arg1447_1, arg1448_1, arg1449_1, arg1450_1, arg1451_1, arg1452_1, arg1453_1, arg1454_1, arg1455_1, arg1456_1, arg1457_1, arg1458_1, arg1459_1, arg1460_1, arg1461_1, arg1462_1, arg1463_1, arg1464_1, arg1465_1, arg1466_1, arg1467_1, arg1468_1, arg1469_1, arg1470_1, arg1471_1, arg1472_1, arg1473_1, arg1474_1, arg1475_1, arg1476_1, arg1477_1, arg1478_1, arg1479_1, arg1480_1, arg1481_1, arg1482_1, arg1483_1, arg1484_1, arg1485_1, arg1486_1, arg1487_1, arg1488_1, arg1489_1, arg1490_1, arg1491_1, arg1492_1, arg1493_1, arg1494_1, arg1495_1, arg1496_1, arg1497_1, arg1498_1, arg1499_1, arg1500_1, arg1501_1, arg1502_1, arg1503_1, arg1504_1, arg1505_1, arg1506_1, arg1507_1, arg1508_1, arg1509_1, arg1510_1, arg1511_1, arg1512_1, arg1513_1, arg1514_1, arg1515_1, arg1516_1, arg1517_1, arg1518_1, arg1519_1, arg1520_1, arg1521_1, arg1522_1, arg1523_1, arg1524_1, arg1525_1, arg1526_1, arg1527_1, arg1528_1, arg1529_1, arg1530_1, arg1531_1, arg1532_1, arg1533_1, arg1534_1, arg1535_1, arg1536_1, arg1537_1, arg1538_1, arg1539_1, arg1540_1, arg1541_1, arg1542_1, arg1543_1, arg1544_1, arg1545_1, arg1546_1, arg1547_1, arg1548_1, arg1549_1, arg1550_1, arg1551_1, arg1552_1, arg1553_1, arg1554_1, arg1555_1, arg1556_1, arg1557_1, arg1558_1, arg1559_1, arg1560_1, arg1561_1, arg1562_1, arg1563_1, arg1564_1, arg1565_1, arg1566_1, arg1567_1, arg1568_1, arg1569_1, arg1570_1, arg1571_1, arg1572_1, arg1573_1, arg1574_1, arg1575_1, arg1576_1, arg1577_1, arg1578_1, arg1579_1, arg1580_1, arg1581_1, arg1582_1, arg1583_1, arg1584_1, arg1585_1, arg1586_1, arg1587_1, arg1588_1, arg1589_1, arg1590_1, arg1591_1, arg1592_1, arg1593_1, arg1594_1, arg1595_1, arg1596_1, arg1597_1, arg1598_1, arg1599_1, arg1600_1, arg1601_1, arg1602_1, arg1603_1, arg1604_1, arg1605_1, arg1606_1, arg1607_1, arg1608_1, arg1609_1, arg1610_1, arg1611_1, arg1612_1, arg1613_1, arg1614_1, arg1615_1, arg1616_1, arg1617_1, arg1618_1, arg1619_1, arg1620_1, arg1621_1, arg1622_1, arg1623_1, arg1624_1, arg1625_1, arg1626_1, arg1627_1, arg1628_1, arg1629_1, arg1630_1, arg1631_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg32_1, (64, ), (1, ))
    assert_size_stride(arg33_1, (64, ), (1, ))
    assert_size_stride(arg34_1, (64, ), (1, ))
    assert_size_stride(arg35_1, (64, ), (1, ))
    assert_size_stride(arg36_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (64, ), (1, ))
    assert_size_stride(arg39_1, (64, ), (1, ))
    assert_size_stride(arg40_1, (64, ), (1, ))
    assert_size_stride(arg41_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg47_1, (64, ), (1, ))
    assert_size_stride(arg48_1, (64, ), (1, ))
    assert_size_stride(arg49_1, (64, ), (1, ))
    assert_size_stride(arg50_1, (64, ), (1, ))
    assert_size_stride(arg51_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg52_1, (64, ), (1, ))
    assert_size_stride(arg53_1, (64, ), (1, ))
    assert_size_stride(arg54_1, (64, ), (1, ))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (256, ), (1, ))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg62_1, (64, ), (1, ))
    assert_size_stride(arg63_1, (64, ), (1, ))
    assert_size_stride(arg64_1, (64, ), (1, ))
    assert_size_stride(arg65_1, (64, ), (1, ))
    assert_size_stride(arg66_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg67_1, (64, ), (1, ))
    assert_size_stride(arg68_1, (64, ), (1, ))
    assert_size_stride(arg69_1, (64, ), (1, ))
    assert_size_stride(arg70_1, (64, ), (1, ))
    assert_size_stride(arg71_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, ), (1, ))
    assert_size_stride(arg76_1, (18, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg77_1, (18, ), (1, ))
    assert_size_stride(arg78_1, (18, ), (1, ))
    assert_size_stride(arg79_1, (18, ), (1, ))
    assert_size_stride(arg80_1, (18, ), (1, ))
    assert_size_stride(arg81_1, (36, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg82_1, (36, ), (1, ))
    assert_size_stride(arg83_1, (36, ), (1, ))
    assert_size_stride(arg84_1, (36, ), (1, ))
    assert_size_stride(arg85_1, (36, ), (1, ))
    assert_size_stride(arg86_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg87_1, (18, ), (1, ))
    assert_size_stride(arg88_1, (18, ), (1, ))
    assert_size_stride(arg89_1, (18, ), (1, ))
    assert_size_stride(arg90_1, (18, ), (1, ))
    assert_size_stride(arg91_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg92_1, (18, ), (1, ))
    assert_size_stride(arg93_1, (18, ), (1, ))
    assert_size_stride(arg94_1, (18, ), (1, ))
    assert_size_stride(arg95_1, (18, ), (1, ))
    assert_size_stride(arg96_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg97_1, (18, ), (1, ))
    assert_size_stride(arg98_1, (18, ), (1, ))
    assert_size_stride(arg99_1, (18, ), (1, ))
    assert_size_stride(arg100_1, (18, ), (1, ))
    assert_size_stride(arg101_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg102_1, (18, ), (1, ))
    assert_size_stride(arg103_1, (18, ), (1, ))
    assert_size_stride(arg104_1, (18, ), (1, ))
    assert_size_stride(arg105_1, (18, ), (1, ))
    assert_size_stride(arg106_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg107_1, (18, ), (1, ))
    assert_size_stride(arg108_1, (18, ), (1, ))
    assert_size_stride(arg109_1, (18, ), (1, ))
    assert_size_stride(arg110_1, (18, ), (1, ))
    assert_size_stride(arg111_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg112_1, (18, ), (1, ))
    assert_size_stride(arg113_1, (18, ), (1, ))
    assert_size_stride(arg114_1, (18, ), (1, ))
    assert_size_stride(arg115_1, (18, ), (1, ))
    assert_size_stride(arg116_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg117_1, (18, ), (1, ))
    assert_size_stride(arg118_1, (18, ), (1, ))
    assert_size_stride(arg119_1, (18, ), (1, ))
    assert_size_stride(arg120_1, (18, ), (1, ))
    assert_size_stride(arg121_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg122_1, (18, ), (1, ))
    assert_size_stride(arg123_1, (18, ), (1, ))
    assert_size_stride(arg124_1, (18, ), (1, ))
    assert_size_stride(arg125_1, (18, ), (1, ))
    assert_size_stride(arg126_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg127_1, (36, ), (1, ))
    assert_size_stride(arg128_1, (36, ), (1, ))
    assert_size_stride(arg129_1, (36, ), (1, ))
    assert_size_stride(arg130_1, (36, ), (1, ))
    assert_size_stride(arg131_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg132_1, (36, ), (1, ))
    assert_size_stride(arg133_1, (36, ), (1, ))
    assert_size_stride(arg134_1, (36, ), (1, ))
    assert_size_stride(arg135_1, (36, ), (1, ))
    assert_size_stride(arg136_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg137_1, (36, ), (1, ))
    assert_size_stride(arg138_1, (36, ), (1, ))
    assert_size_stride(arg139_1, (36, ), (1, ))
    assert_size_stride(arg140_1, (36, ), (1, ))
    assert_size_stride(arg141_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg142_1, (36, ), (1, ))
    assert_size_stride(arg143_1, (36, ), (1, ))
    assert_size_stride(arg144_1, (36, ), (1, ))
    assert_size_stride(arg145_1, (36, ), (1, ))
    assert_size_stride(arg146_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg147_1, (36, ), (1, ))
    assert_size_stride(arg148_1, (36, ), (1, ))
    assert_size_stride(arg149_1, (36, ), (1, ))
    assert_size_stride(arg150_1, (36, ), (1, ))
    assert_size_stride(arg151_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg152_1, (36, ), (1, ))
    assert_size_stride(arg153_1, (36, ), (1, ))
    assert_size_stride(arg154_1, (36, ), (1, ))
    assert_size_stride(arg155_1, (36, ), (1, ))
    assert_size_stride(arg156_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg157_1, (36, ), (1, ))
    assert_size_stride(arg158_1, (36, ), (1, ))
    assert_size_stride(arg159_1, (36, ), (1, ))
    assert_size_stride(arg160_1, (36, ), (1, ))
    assert_size_stride(arg161_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg162_1, (36, ), (1, ))
    assert_size_stride(arg163_1, (36, ), (1, ))
    assert_size_stride(arg164_1, (36, ), (1, ))
    assert_size_stride(arg165_1, (36, ), (1, ))
    assert_size_stride(arg166_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg167_1, (18, ), (1, ))
    assert_size_stride(arg168_1, (18, ), (1, ))
    assert_size_stride(arg169_1, (18, ), (1, ))
    assert_size_stride(arg170_1, (18, ), (1, ))
    assert_size_stride(arg171_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg172_1, (36, ), (1, ))
    assert_size_stride(arg173_1, (36, ), (1, ))
    assert_size_stride(arg174_1, (36, ), (1, ))
    assert_size_stride(arg175_1, (36, ), (1, ))
    assert_size_stride(arg176_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg177_1, (72, ), (1, ))
    assert_size_stride(arg178_1, (72, ), (1, ))
    assert_size_stride(arg179_1, (72, ), (1, ))
    assert_size_stride(arg180_1, (72, ), (1, ))
    assert_size_stride(arg181_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg182_1, (18, ), (1, ))
    assert_size_stride(arg183_1, (18, ), (1, ))
    assert_size_stride(arg184_1, (18, ), (1, ))
    assert_size_stride(arg185_1, (18, ), (1, ))
    assert_size_stride(arg186_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg187_1, (18, ), (1, ))
    assert_size_stride(arg188_1, (18, ), (1, ))
    assert_size_stride(arg189_1, (18, ), (1, ))
    assert_size_stride(arg190_1, (18, ), (1, ))
    assert_size_stride(arg191_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg192_1, (18, ), (1, ))
    assert_size_stride(arg193_1, (18, ), (1, ))
    assert_size_stride(arg194_1, (18, ), (1, ))
    assert_size_stride(arg195_1, (18, ), (1, ))
    assert_size_stride(arg196_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg197_1, (18, ), (1, ))
    assert_size_stride(arg198_1, (18, ), (1, ))
    assert_size_stride(arg199_1, (18, ), (1, ))
    assert_size_stride(arg200_1, (18, ), (1, ))
    assert_size_stride(arg201_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg202_1, (18, ), (1, ))
    assert_size_stride(arg203_1, (18, ), (1, ))
    assert_size_stride(arg204_1, (18, ), (1, ))
    assert_size_stride(arg205_1, (18, ), (1, ))
    assert_size_stride(arg206_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg207_1, (18, ), (1, ))
    assert_size_stride(arg208_1, (18, ), (1, ))
    assert_size_stride(arg209_1, (18, ), (1, ))
    assert_size_stride(arg210_1, (18, ), (1, ))
    assert_size_stride(arg211_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg212_1, (18, ), (1, ))
    assert_size_stride(arg213_1, (18, ), (1, ))
    assert_size_stride(arg214_1, (18, ), (1, ))
    assert_size_stride(arg215_1, (18, ), (1, ))
    assert_size_stride(arg216_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg217_1, (18, ), (1, ))
    assert_size_stride(arg218_1, (18, ), (1, ))
    assert_size_stride(arg219_1, (18, ), (1, ))
    assert_size_stride(arg220_1, (18, ), (1, ))
    assert_size_stride(arg221_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg222_1, (36, ), (1, ))
    assert_size_stride(arg223_1, (36, ), (1, ))
    assert_size_stride(arg224_1, (36, ), (1, ))
    assert_size_stride(arg225_1, (36, ), (1, ))
    assert_size_stride(arg226_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg227_1, (36, ), (1, ))
    assert_size_stride(arg228_1, (36, ), (1, ))
    assert_size_stride(arg229_1, (36, ), (1, ))
    assert_size_stride(arg230_1, (36, ), (1, ))
    assert_size_stride(arg231_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg232_1, (36, ), (1, ))
    assert_size_stride(arg233_1, (36, ), (1, ))
    assert_size_stride(arg234_1, (36, ), (1, ))
    assert_size_stride(arg235_1, (36, ), (1, ))
    assert_size_stride(arg236_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg237_1, (36, ), (1, ))
    assert_size_stride(arg238_1, (36, ), (1, ))
    assert_size_stride(arg239_1, (36, ), (1, ))
    assert_size_stride(arg240_1, (36, ), (1, ))
    assert_size_stride(arg241_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg242_1, (36, ), (1, ))
    assert_size_stride(arg243_1, (36, ), (1, ))
    assert_size_stride(arg244_1, (36, ), (1, ))
    assert_size_stride(arg245_1, (36, ), (1, ))
    assert_size_stride(arg246_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg247_1, (36, ), (1, ))
    assert_size_stride(arg248_1, (36, ), (1, ))
    assert_size_stride(arg249_1, (36, ), (1, ))
    assert_size_stride(arg250_1, (36, ), (1, ))
    assert_size_stride(arg251_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg252_1, (36, ), (1, ))
    assert_size_stride(arg253_1, (36, ), (1, ))
    assert_size_stride(arg254_1, (36, ), (1, ))
    assert_size_stride(arg255_1, (36, ), (1, ))
    assert_size_stride(arg256_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg257_1, (36, ), (1, ))
    assert_size_stride(arg258_1, (36, ), (1, ))
    assert_size_stride(arg259_1, (36, ), (1, ))
    assert_size_stride(arg260_1, (36, ), (1, ))
    assert_size_stride(arg261_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg262_1, (72, ), (1, ))
    assert_size_stride(arg263_1, (72, ), (1, ))
    assert_size_stride(arg264_1, (72, ), (1, ))
    assert_size_stride(arg265_1, (72, ), (1, ))
    assert_size_stride(arg266_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg267_1, (72, ), (1, ))
    assert_size_stride(arg268_1, (72, ), (1, ))
    assert_size_stride(arg269_1, (72, ), (1, ))
    assert_size_stride(arg270_1, (72, ), (1, ))
    assert_size_stride(arg271_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg272_1, (72, ), (1, ))
    assert_size_stride(arg273_1, (72, ), (1, ))
    assert_size_stride(arg274_1, (72, ), (1, ))
    assert_size_stride(arg275_1, (72, ), (1, ))
    assert_size_stride(arg276_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg277_1, (72, ), (1, ))
    assert_size_stride(arg278_1, (72, ), (1, ))
    assert_size_stride(arg279_1, (72, ), (1, ))
    assert_size_stride(arg280_1, (72, ), (1, ))
    assert_size_stride(arg281_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg282_1, (72, ), (1, ))
    assert_size_stride(arg283_1, (72, ), (1, ))
    assert_size_stride(arg284_1, (72, ), (1, ))
    assert_size_stride(arg285_1, (72, ), (1, ))
    assert_size_stride(arg286_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg287_1, (72, ), (1, ))
    assert_size_stride(arg288_1, (72, ), (1, ))
    assert_size_stride(arg289_1, (72, ), (1, ))
    assert_size_stride(arg290_1, (72, ), (1, ))
    assert_size_stride(arg291_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg292_1, (72, ), (1, ))
    assert_size_stride(arg293_1, (72, ), (1, ))
    assert_size_stride(arg294_1, (72, ), (1, ))
    assert_size_stride(arg295_1, (72, ), (1, ))
    assert_size_stride(arg296_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg297_1, (72, ), (1, ))
    assert_size_stride(arg298_1, (72, ), (1, ))
    assert_size_stride(arg299_1, (72, ), (1, ))
    assert_size_stride(arg300_1, (72, ), (1, ))
    assert_size_stride(arg301_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg302_1, (18, ), (1, ))
    assert_size_stride(arg303_1, (18, ), (1, ))
    assert_size_stride(arg304_1, (18, ), (1, ))
    assert_size_stride(arg305_1, (18, ), (1, ))
    assert_size_stride(arg306_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg307_1, (18, ), (1, ))
    assert_size_stride(arg308_1, (18, ), (1, ))
    assert_size_stride(arg309_1, (18, ), (1, ))
    assert_size_stride(arg310_1, (18, ), (1, ))
    assert_size_stride(arg311_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg312_1, (36, ), (1, ))
    assert_size_stride(arg313_1, (36, ), (1, ))
    assert_size_stride(arg314_1, (36, ), (1, ))
    assert_size_stride(arg315_1, (36, ), (1, ))
    assert_size_stride(arg316_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg317_1, (36, ), (1, ))
    assert_size_stride(arg318_1, (36, ), (1, ))
    assert_size_stride(arg319_1, (36, ), (1, ))
    assert_size_stride(arg320_1, (36, ), (1, ))
    assert_size_stride(arg321_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg322_1, (18, ), (1, ))
    assert_size_stride(arg323_1, (18, ), (1, ))
    assert_size_stride(arg324_1, (18, ), (1, ))
    assert_size_stride(arg325_1, (18, ), (1, ))
    assert_size_stride(arg326_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg327_1, (72, ), (1, ))
    assert_size_stride(arg328_1, (72, ), (1, ))
    assert_size_stride(arg329_1, (72, ), (1, ))
    assert_size_stride(arg330_1, (72, ), (1, ))
    assert_size_stride(arg331_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg332_1, (72, ), (1, ))
    assert_size_stride(arg333_1, (72, ), (1, ))
    assert_size_stride(arg334_1, (72, ), (1, ))
    assert_size_stride(arg335_1, (72, ), (1, ))
    assert_size_stride(arg336_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg337_1, (18, ), (1, ))
    assert_size_stride(arg338_1, (18, ), (1, ))
    assert_size_stride(arg339_1, (18, ), (1, ))
    assert_size_stride(arg340_1, (18, ), (1, ))
    assert_size_stride(arg341_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg342_1, (18, ), (1, ))
    assert_size_stride(arg343_1, (18, ), (1, ))
    assert_size_stride(arg344_1, (18, ), (1, ))
    assert_size_stride(arg345_1, (18, ), (1, ))
    assert_size_stride(arg346_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg347_1, (18, ), (1, ))
    assert_size_stride(arg348_1, (18, ), (1, ))
    assert_size_stride(arg349_1, (18, ), (1, ))
    assert_size_stride(arg350_1, (18, ), (1, ))
    assert_size_stride(arg351_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg352_1, (18, ), (1, ))
    assert_size_stride(arg353_1, (18, ), (1, ))
    assert_size_stride(arg354_1, (18, ), (1, ))
    assert_size_stride(arg355_1, (18, ), (1, ))
    assert_size_stride(arg356_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg357_1, (18, ), (1, ))
    assert_size_stride(arg358_1, (18, ), (1, ))
    assert_size_stride(arg359_1, (18, ), (1, ))
    assert_size_stride(arg360_1, (18, ), (1, ))
    assert_size_stride(arg361_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg362_1, (18, ), (1, ))
    assert_size_stride(arg363_1, (18, ), (1, ))
    assert_size_stride(arg364_1, (18, ), (1, ))
    assert_size_stride(arg365_1, (18, ), (1, ))
    assert_size_stride(arg366_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg367_1, (18, ), (1, ))
    assert_size_stride(arg368_1, (18, ), (1, ))
    assert_size_stride(arg369_1, (18, ), (1, ))
    assert_size_stride(arg370_1, (18, ), (1, ))
    assert_size_stride(arg371_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg372_1, (18, ), (1, ))
    assert_size_stride(arg373_1, (18, ), (1, ))
    assert_size_stride(arg374_1, (18, ), (1, ))
    assert_size_stride(arg375_1, (18, ), (1, ))
    assert_size_stride(arg376_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg377_1, (36, ), (1, ))
    assert_size_stride(arg378_1, (36, ), (1, ))
    assert_size_stride(arg379_1, (36, ), (1, ))
    assert_size_stride(arg380_1, (36, ), (1, ))
    assert_size_stride(arg381_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg382_1, (36, ), (1, ))
    assert_size_stride(arg383_1, (36, ), (1, ))
    assert_size_stride(arg384_1, (36, ), (1, ))
    assert_size_stride(arg385_1, (36, ), (1, ))
    assert_size_stride(arg386_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg387_1, (36, ), (1, ))
    assert_size_stride(arg388_1, (36, ), (1, ))
    assert_size_stride(arg389_1, (36, ), (1, ))
    assert_size_stride(arg390_1, (36, ), (1, ))
    assert_size_stride(arg391_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg392_1, (36, ), (1, ))
    assert_size_stride(arg393_1, (36, ), (1, ))
    assert_size_stride(arg394_1, (36, ), (1, ))
    assert_size_stride(arg395_1, (36, ), (1, ))
    assert_size_stride(arg396_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg397_1, (36, ), (1, ))
    assert_size_stride(arg398_1, (36, ), (1, ))
    assert_size_stride(arg399_1, (36, ), (1, ))
    assert_size_stride(arg400_1, (36, ), (1, ))
    assert_size_stride(arg401_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg402_1, (36, ), (1, ))
    assert_size_stride(arg403_1, (36, ), (1, ))
    assert_size_stride(arg404_1, (36, ), (1, ))
    assert_size_stride(arg405_1, (36, ), (1, ))
    assert_size_stride(arg406_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg407_1, (36, ), (1, ))
    assert_size_stride(arg408_1, (36, ), (1, ))
    assert_size_stride(arg409_1, (36, ), (1, ))
    assert_size_stride(arg410_1, (36, ), (1, ))
    assert_size_stride(arg411_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg412_1, (36, ), (1, ))
    assert_size_stride(arg413_1, (36, ), (1, ))
    assert_size_stride(arg414_1, (36, ), (1, ))
    assert_size_stride(arg415_1, (36, ), (1, ))
    assert_size_stride(arg416_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg417_1, (72, ), (1, ))
    assert_size_stride(arg418_1, (72, ), (1, ))
    assert_size_stride(arg419_1, (72, ), (1, ))
    assert_size_stride(arg420_1, (72, ), (1, ))
    assert_size_stride(arg421_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg422_1, (72, ), (1, ))
    assert_size_stride(arg423_1, (72, ), (1, ))
    assert_size_stride(arg424_1, (72, ), (1, ))
    assert_size_stride(arg425_1, (72, ), (1, ))
    assert_size_stride(arg426_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg427_1, (72, ), (1, ))
    assert_size_stride(arg428_1, (72, ), (1, ))
    assert_size_stride(arg429_1, (72, ), (1, ))
    assert_size_stride(arg430_1, (72, ), (1, ))
    assert_size_stride(arg431_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg432_1, (72, ), (1, ))
    assert_size_stride(arg433_1, (72, ), (1, ))
    assert_size_stride(arg434_1, (72, ), (1, ))
    assert_size_stride(arg435_1, (72, ), (1, ))
    assert_size_stride(arg436_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg437_1, (72, ), (1, ))
    assert_size_stride(arg438_1, (72, ), (1, ))
    assert_size_stride(arg439_1, (72, ), (1, ))
    assert_size_stride(arg440_1, (72, ), (1, ))
    assert_size_stride(arg441_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg442_1, (72, ), (1, ))
    assert_size_stride(arg443_1, (72, ), (1, ))
    assert_size_stride(arg444_1, (72, ), (1, ))
    assert_size_stride(arg445_1, (72, ), (1, ))
    assert_size_stride(arg446_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg447_1, (72, ), (1, ))
    assert_size_stride(arg448_1, (72, ), (1, ))
    assert_size_stride(arg449_1, (72, ), (1, ))
    assert_size_stride(arg450_1, (72, ), (1, ))
    assert_size_stride(arg451_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg452_1, (72, ), (1, ))
    assert_size_stride(arg453_1, (72, ), (1, ))
    assert_size_stride(arg454_1, (72, ), (1, ))
    assert_size_stride(arg455_1, (72, ), (1, ))
    assert_size_stride(arg456_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg457_1, (18, ), (1, ))
    assert_size_stride(arg458_1, (18, ), (1, ))
    assert_size_stride(arg459_1, (18, ), (1, ))
    assert_size_stride(arg460_1, (18, ), (1, ))
    assert_size_stride(arg461_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg462_1, (18, ), (1, ))
    assert_size_stride(arg463_1, (18, ), (1, ))
    assert_size_stride(arg464_1, (18, ), (1, ))
    assert_size_stride(arg465_1, (18, ), (1, ))
    assert_size_stride(arg466_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg467_1, (36, ), (1, ))
    assert_size_stride(arg468_1, (36, ), (1, ))
    assert_size_stride(arg469_1, (36, ), (1, ))
    assert_size_stride(arg470_1, (36, ), (1, ))
    assert_size_stride(arg471_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg472_1, (36, ), (1, ))
    assert_size_stride(arg473_1, (36, ), (1, ))
    assert_size_stride(arg474_1, (36, ), (1, ))
    assert_size_stride(arg475_1, (36, ), (1, ))
    assert_size_stride(arg476_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg477_1, (18, ), (1, ))
    assert_size_stride(arg478_1, (18, ), (1, ))
    assert_size_stride(arg479_1, (18, ), (1, ))
    assert_size_stride(arg480_1, (18, ), (1, ))
    assert_size_stride(arg481_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg482_1, (72, ), (1, ))
    assert_size_stride(arg483_1, (72, ), (1, ))
    assert_size_stride(arg484_1, (72, ), (1, ))
    assert_size_stride(arg485_1, (72, ), (1, ))
    assert_size_stride(arg486_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg487_1, (72, ), (1, ))
    assert_size_stride(arg488_1, (72, ), (1, ))
    assert_size_stride(arg489_1, (72, ), (1, ))
    assert_size_stride(arg490_1, (72, ), (1, ))
    assert_size_stride(arg491_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg492_1, (18, ), (1, ))
    assert_size_stride(arg493_1, (18, ), (1, ))
    assert_size_stride(arg494_1, (18, ), (1, ))
    assert_size_stride(arg495_1, (18, ), (1, ))
    assert_size_stride(arg496_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg497_1, (18, ), (1, ))
    assert_size_stride(arg498_1, (18, ), (1, ))
    assert_size_stride(arg499_1, (18, ), (1, ))
    assert_size_stride(arg500_1, (18, ), (1, ))
    assert_size_stride(arg501_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg502_1, (18, ), (1, ))
    assert_size_stride(arg503_1, (18, ), (1, ))
    assert_size_stride(arg504_1, (18, ), (1, ))
    assert_size_stride(arg505_1, (18, ), (1, ))
    assert_size_stride(arg506_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg507_1, (18, ), (1, ))
    assert_size_stride(arg508_1, (18, ), (1, ))
    assert_size_stride(arg509_1, (18, ), (1, ))
    assert_size_stride(arg510_1, (18, ), (1, ))
    assert_size_stride(arg511_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg512_1, (18, ), (1, ))
    assert_size_stride(arg513_1, (18, ), (1, ))
    assert_size_stride(arg514_1, (18, ), (1, ))
    assert_size_stride(arg515_1, (18, ), (1, ))
    assert_size_stride(arg516_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg517_1, (18, ), (1, ))
    assert_size_stride(arg518_1, (18, ), (1, ))
    assert_size_stride(arg519_1, (18, ), (1, ))
    assert_size_stride(arg520_1, (18, ), (1, ))
    assert_size_stride(arg521_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg522_1, (18, ), (1, ))
    assert_size_stride(arg523_1, (18, ), (1, ))
    assert_size_stride(arg524_1, (18, ), (1, ))
    assert_size_stride(arg525_1, (18, ), (1, ))
    assert_size_stride(arg526_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg527_1, (18, ), (1, ))
    assert_size_stride(arg528_1, (18, ), (1, ))
    assert_size_stride(arg529_1, (18, ), (1, ))
    assert_size_stride(arg530_1, (18, ), (1, ))
    assert_size_stride(arg531_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg532_1, (36, ), (1, ))
    assert_size_stride(arg533_1, (36, ), (1, ))
    assert_size_stride(arg534_1, (36, ), (1, ))
    assert_size_stride(arg535_1, (36, ), (1, ))
    assert_size_stride(arg536_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg537_1, (36, ), (1, ))
    assert_size_stride(arg538_1, (36, ), (1, ))
    assert_size_stride(arg539_1, (36, ), (1, ))
    assert_size_stride(arg540_1, (36, ), (1, ))
    assert_size_stride(arg541_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg542_1, (36, ), (1, ))
    assert_size_stride(arg543_1, (36, ), (1, ))
    assert_size_stride(arg544_1, (36, ), (1, ))
    assert_size_stride(arg545_1, (36, ), (1, ))
    assert_size_stride(arg546_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg547_1, (36, ), (1, ))
    assert_size_stride(arg548_1, (36, ), (1, ))
    assert_size_stride(arg549_1, (36, ), (1, ))
    assert_size_stride(arg550_1, (36, ), (1, ))
    assert_size_stride(arg551_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg552_1, (36, ), (1, ))
    assert_size_stride(arg553_1, (36, ), (1, ))
    assert_size_stride(arg554_1, (36, ), (1, ))
    assert_size_stride(arg555_1, (36, ), (1, ))
    assert_size_stride(arg556_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg557_1, (36, ), (1, ))
    assert_size_stride(arg558_1, (36, ), (1, ))
    assert_size_stride(arg559_1, (36, ), (1, ))
    assert_size_stride(arg560_1, (36, ), (1, ))
    assert_size_stride(arg561_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg562_1, (36, ), (1, ))
    assert_size_stride(arg563_1, (36, ), (1, ))
    assert_size_stride(arg564_1, (36, ), (1, ))
    assert_size_stride(arg565_1, (36, ), (1, ))
    assert_size_stride(arg566_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg567_1, (36, ), (1, ))
    assert_size_stride(arg568_1, (36, ), (1, ))
    assert_size_stride(arg569_1, (36, ), (1, ))
    assert_size_stride(arg570_1, (36, ), (1, ))
    assert_size_stride(arg571_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg572_1, (72, ), (1, ))
    assert_size_stride(arg573_1, (72, ), (1, ))
    assert_size_stride(arg574_1, (72, ), (1, ))
    assert_size_stride(arg575_1, (72, ), (1, ))
    assert_size_stride(arg576_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg577_1, (72, ), (1, ))
    assert_size_stride(arg578_1, (72, ), (1, ))
    assert_size_stride(arg579_1, (72, ), (1, ))
    assert_size_stride(arg580_1, (72, ), (1, ))
    assert_size_stride(arg581_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg582_1, (72, ), (1, ))
    assert_size_stride(arg583_1, (72, ), (1, ))
    assert_size_stride(arg584_1, (72, ), (1, ))
    assert_size_stride(arg585_1, (72, ), (1, ))
    assert_size_stride(arg586_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg587_1, (72, ), (1, ))
    assert_size_stride(arg588_1, (72, ), (1, ))
    assert_size_stride(arg589_1, (72, ), (1, ))
    assert_size_stride(arg590_1, (72, ), (1, ))
    assert_size_stride(arg591_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg592_1, (72, ), (1, ))
    assert_size_stride(arg593_1, (72, ), (1, ))
    assert_size_stride(arg594_1, (72, ), (1, ))
    assert_size_stride(arg595_1, (72, ), (1, ))
    assert_size_stride(arg596_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg597_1, (72, ), (1, ))
    assert_size_stride(arg598_1, (72, ), (1, ))
    assert_size_stride(arg599_1, (72, ), (1, ))
    assert_size_stride(arg600_1, (72, ), (1, ))
    assert_size_stride(arg601_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg602_1, (72, ), (1, ))
    assert_size_stride(arg603_1, (72, ), (1, ))
    assert_size_stride(arg604_1, (72, ), (1, ))
    assert_size_stride(arg605_1, (72, ), (1, ))
    assert_size_stride(arg606_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg607_1, (72, ), (1, ))
    assert_size_stride(arg608_1, (72, ), (1, ))
    assert_size_stride(arg609_1, (72, ), (1, ))
    assert_size_stride(arg610_1, (72, ), (1, ))
    assert_size_stride(arg611_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg612_1, (18, ), (1, ))
    assert_size_stride(arg613_1, (18, ), (1, ))
    assert_size_stride(arg614_1, (18, ), (1, ))
    assert_size_stride(arg615_1, (18, ), (1, ))
    assert_size_stride(arg616_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg617_1, (18, ), (1, ))
    assert_size_stride(arg618_1, (18, ), (1, ))
    assert_size_stride(arg619_1, (18, ), (1, ))
    assert_size_stride(arg620_1, (18, ), (1, ))
    assert_size_stride(arg621_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg622_1, (36, ), (1, ))
    assert_size_stride(arg623_1, (36, ), (1, ))
    assert_size_stride(arg624_1, (36, ), (1, ))
    assert_size_stride(arg625_1, (36, ), (1, ))
    assert_size_stride(arg626_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg627_1, (36, ), (1, ))
    assert_size_stride(arg628_1, (36, ), (1, ))
    assert_size_stride(arg629_1, (36, ), (1, ))
    assert_size_stride(arg630_1, (36, ), (1, ))
    assert_size_stride(arg631_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg632_1, (18, ), (1, ))
    assert_size_stride(arg633_1, (18, ), (1, ))
    assert_size_stride(arg634_1, (18, ), (1, ))
    assert_size_stride(arg635_1, (18, ), (1, ))
    assert_size_stride(arg636_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg637_1, (72, ), (1, ))
    assert_size_stride(arg638_1, (72, ), (1, ))
    assert_size_stride(arg639_1, (72, ), (1, ))
    assert_size_stride(arg640_1, (72, ), (1, ))
    assert_size_stride(arg641_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg642_1, (72, ), (1, ))
    assert_size_stride(arg643_1, (72, ), (1, ))
    assert_size_stride(arg644_1, (72, ), (1, ))
    assert_size_stride(arg645_1, (72, ), (1, ))
    assert_size_stride(arg646_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg647_1, (18, ), (1, ))
    assert_size_stride(arg648_1, (18, ), (1, ))
    assert_size_stride(arg649_1, (18, ), (1, ))
    assert_size_stride(arg650_1, (18, ), (1, ))
    assert_size_stride(arg651_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg652_1, (18, ), (1, ))
    assert_size_stride(arg653_1, (18, ), (1, ))
    assert_size_stride(arg654_1, (18, ), (1, ))
    assert_size_stride(arg655_1, (18, ), (1, ))
    assert_size_stride(arg656_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg657_1, (18, ), (1, ))
    assert_size_stride(arg658_1, (18, ), (1, ))
    assert_size_stride(arg659_1, (18, ), (1, ))
    assert_size_stride(arg660_1, (18, ), (1, ))
    assert_size_stride(arg661_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg662_1, (18, ), (1, ))
    assert_size_stride(arg663_1, (18, ), (1, ))
    assert_size_stride(arg664_1, (18, ), (1, ))
    assert_size_stride(arg665_1, (18, ), (1, ))
    assert_size_stride(arg666_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg667_1, (18, ), (1, ))
    assert_size_stride(arg668_1, (18, ), (1, ))
    assert_size_stride(arg669_1, (18, ), (1, ))
    assert_size_stride(arg670_1, (18, ), (1, ))
    assert_size_stride(arg671_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg672_1, (18, ), (1, ))
    assert_size_stride(arg673_1, (18, ), (1, ))
    assert_size_stride(arg674_1, (18, ), (1, ))
    assert_size_stride(arg675_1, (18, ), (1, ))
    assert_size_stride(arg676_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg677_1, (18, ), (1, ))
    assert_size_stride(arg678_1, (18, ), (1, ))
    assert_size_stride(arg679_1, (18, ), (1, ))
    assert_size_stride(arg680_1, (18, ), (1, ))
    assert_size_stride(arg681_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg682_1, (18, ), (1, ))
    assert_size_stride(arg683_1, (18, ), (1, ))
    assert_size_stride(arg684_1, (18, ), (1, ))
    assert_size_stride(arg685_1, (18, ), (1, ))
    assert_size_stride(arg686_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg687_1, (36, ), (1, ))
    assert_size_stride(arg688_1, (36, ), (1, ))
    assert_size_stride(arg689_1, (36, ), (1, ))
    assert_size_stride(arg690_1, (36, ), (1, ))
    assert_size_stride(arg691_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg692_1, (36, ), (1, ))
    assert_size_stride(arg693_1, (36, ), (1, ))
    assert_size_stride(arg694_1, (36, ), (1, ))
    assert_size_stride(arg695_1, (36, ), (1, ))
    assert_size_stride(arg696_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg697_1, (36, ), (1, ))
    assert_size_stride(arg698_1, (36, ), (1, ))
    assert_size_stride(arg699_1, (36, ), (1, ))
    assert_size_stride(arg700_1, (36, ), (1, ))
    assert_size_stride(arg701_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg702_1, (36, ), (1, ))
    assert_size_stride(arg703_1, (36, ), (1, ))
    assert_size_stride(arg704_1, (36, ), (1, ))
    assert_size_stride(arg705_1, (36, ), (1, ))
    assert_size_stride(arg706_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg707_1, (36, ), (1, ))
    assert_size_stride(arg708_1, (36, ), (1, ))
    assert_size_stride(arg709_1, (36, ), (1, ))
    assert_size_stride(arg710_1, (36, ), (1, ))
    assert_size_stride(arg711_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg712_1, (36, ), (1, ))
    assert_size_stride(arg713_1, (36, ), (1, ))
    assert_size_stride(arg714_1, (36, ), (1, ))
    assert_size_stride(arg715_1, (36, ), (1, ))
    assert_size_stride(arg716_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg717_1, (36, ), (1, ))
    assert_size_stride(arg718_1, (36, ), (1, ))
    assert_size_stride(arg719_1, (36, ), (1, ))
    assert_size_stride(arg720_1, (36, ), (1, ))
    assert_size_stride(arg721_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg722_1, (36, ), (1, ))
    assert_size_stride(arg723_1, (36, ), (1, ))
    assert_size_stride(arg724_1, (36, ), (1, ))
    assert_size_stride(arg725_1, (36, ), (1, ))
    assert_size_stride(arg726_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg727_1, (72, ), (1, ))
    assert_size_stride(arg728_1, (72, ), (1, ))
    assert_size_stride(arg729_1, (72, ), (1, ))
    assert_size_stride(arg730_1, (72, ), (1, ))
    assert_size_stride(arg731_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg732_1, (72, ), (1, ))
    assert_size_stride(arg733_1, (72, ), (1, ))
    assert_size_stride(arg734_1, (72, ), (1, ))
    assert_size_stride(arg735_1, (72, ), (1, ))
    assert_size_stride(arg736_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg737_1, (72, ), (1, ))
    assert_size_stride(arg738_1, (72, ), (1, ))
    assert_size_stride(arg739_1, (72, ), (1, ))
    assert_size_stride(arg740_1, (72, ), (1, ))
    assert_size_stride(arg741_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg742_1, (72, ), (1, ))
    assert_size_stride(arg743_1, (72, ), (1, ))
    assert_size_stride(arg744_1, (72, ), (1, ))
    assert_size_stride(arg745_1, (72, ), (1, ))
    assert_size_stride(arg746_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg747_1, (72, ), (1, ))
    assert_size_stride(arg748_1, (72, ), (1, ))
    assert_size_stride(arg749_1, (72, ), (1, ))
    assert_size_stride(arg750_1, (72, ), (1, ))
    assert_size_stride(arg751_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg752_1, (72, ), (1, ))
    assert_size_stride(arg753_1, (72, ), (1, ))
    assert_size_stride(arg754_1, (72, ), (1, ))
    assert_size_stride(arg755_1, (72, ), (1, ))
    assert_size_stride(arg756_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg757_1, (72, ), (1, ))
    assert_size_stride(arg758_1, (72, ), (1, ))
    assert_size_stride(arg759_1, (72, ), (1, ))
    assert_size_stride(arg760_1, (72, ), (1, ))
    assert_size_stride(arg761_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg762_1, (72, ), (1, ))
    assert_size_stride(arg763_1, (72, ), (1, ))
    assert_size_stride(arg764_1, (72, ), (1, ))
    assert_size_stride(arg765_1, (72, ), (1, ))
    assert_size_stride(arg766_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg767_1, (18, ), (1, ))
    assert_size_stride(arg768_1, (18, ), (1, ))
    assert_size_stride(arg769_1, (18, ), (1, ))
    assert_size_stride(arg770_1, (18, ), (1, ))
    assert_size_stride(arg771_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg772_1, (18, ), (1, ))
    assert_size_stride(arg773_1, (18, ), (1, ))
    assert_size_stride(arg774_1, (18, ), (1, ))
    assert_size_stride(arg775_1, (18, ), (1, ))
    assert_size_stride(arg776_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg777_1, (36, ), (1, ))
    assert_size_stride(arg778_1, (36, ), (1, ))
    assert_size_stride(arg779_1, (36, ), (1, ))
    assert_size_stride(arg780_1, (36, ), (1, ))
    assert_size_stride(arg781_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg782_1, (36, ), (1, ))
    assert_size_stride(arg783_1, (36, ), (1, ))
    assert_size_stride(arg784_1, (36, ), (1, ))
    assert_size_stride(arg785_1, (36, ), (1, ))
    assert_size_stride(arg786_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg787_1, (18, ), (1, ))
    assert_size_stride(arg788_1, (18, ), (1, ))
    assert_size_stride(arg789_1, (18, ), (1, ))
    assert_size_stride(arg790_1, (18, ), (1, ))
    assert_size_stride(arg791_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg792_1, (72, ), (1, ))
    assert_size_stride(arg793_1, (72, ), (1, ))
    assert_size_stride(arg794_1, (72, ), (1, ))
    assert_size_stride(arg795_1, (72, ), (1, ))
    assert_size_stride(arg796_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg797_1, (72, ), (1, ))
    assert_size_stride(arg798_1, (72, ), (1, ))
    assert_size_stride(arg799_1, (72, ), (1, ))
    assert_size_stride(arg800_1, (72, ), (1, ))
    assert_size_stride(arg801_1, (144, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg802_1, (144, ), (1, ))
    assert_size_stride(arg803_1, (144, ), (1, ))
    assert_size_stride(arg804_1, (144, ), (1, ))
    assert_size_stride(arg805_1, (144, ), (1, ))
    assert_size_stride(arg806_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg807_1, (18, ), (1, ))
    assert_size_stride(arg808_1, (18, ), (1, ))
    assert_size_stride(arg809_1, (18, ), (1, ))
    assert_size_stride(arg810_1, (18, ), (1, ))
    assert_size_stride(arg811_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg812_1, (18, ), (1, ))
    assert_size_stride(arg813_1, (18, ), (1, ))
    assert_size_stride(arg814_1, (18, ), (1, ))
    assert_size_stride(arg815_1, (18, ), (1, ))
    assert_size_stride(arg816_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg817_1, (18, ), (1, ))
    assert_size_stride(arg818_1, (18, ), (1, ))
    assert_size_stride(arg819_1, (18, ), (1, ))
    assert_size_stride(arg820_1, (18, ), (1, ))
    assert_size_stride(arg821_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg822_1, (18, ), (1, ))
    assert_size_stride(arg823_1, (18, ), (1, ))
    assert_size_stride(arg824_1, (18, ), (1, ))
    assert_size_stride(arg825_1, (18, ), (1, ))
    assert_size_stride(arg826_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg827_1, (18, ), (1, ))
    assert_size_stride(arg828_1, (18, ), (1, ))
    assert_size_stride(arg829_1, (18, ), (1, ))
    assert_size_stride(arg830_1, (18, ), (1, ))
    assert_size_stride(arg831_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg832_1, (18, ), (1, ))
    assert_size_stride(arg833_1, (18, ), (1, ))
    assert_size_stride(arg834_1, (18, ), (1, ))
    assert_size_stride(arg835_1, (18, ), (1, ))
    assert_size_stride(arg836_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg837_1, (18, ), (1, ))
    assert_size_stride(arg838_1, (18, ), (1, ))
    assert_size_stride(arg839_1, (18, ), (1, ))
    assert_size_stride(arg840_1, (18, ), (1, ))
    assert_size_stride(arg841_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg842_1, (18, ), (1, ))
    assert_size_stride(arg843_1, (18, ), (1, ))
    assert_size_stride(arg844_1, (18, ), (1, ))
    assert_size_stride(arg845_1, (18, ), (1, ))
    assert_size_stride(arg846_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg847_1, (36, ), (1, ))
    assert_size_stride(arg848_1, (36, ), (1, ))
    assert_size_stride(arg849_1, (36, ), (1, ))
    assert_size_stride(arg850_1, (36, ), (1, ))
    assert_size_stride(arg851_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg852_1, (36, ), (1, ))
    assert_size_stride(arg853_1, (36, ), (1, ))
    assert_size_stride(arg854_1, (36, ), (1, ))
    assert_size_stride(arg855_1, (36, ), (1, ))
    assert_size_stride(arg856_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg857_1, (36, ), (1, ))
    assert_size_stride(arg858_1, (36, ), (1, ))
    assert_size_stride(arg859_1, (36, ), (1, ))
    assert_size_stride(arg860_1, (36, ), (1, ))
    assert_size_stride(arg861_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg862_1, (36, ), (1, ))
    assert_size_stride(arg863_1, (36, ), (1, ))
    assert_size_stride(arg864_1, (36, ), (1, ))
    assert_size_stride(arg865_1, (36, ), (1, ))
    assert_size_stride(arg866_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg867_1, (36, ), (1, ))
    assert_size_stride(arg868_1, (36, ), (1, ))
    assert_size_stride(arg869_1, (36, ), (1, ))
    assert_size_stride(arg870_1, (36, ), (1, ))
    assert_size_stride(arg871_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg872_1, (36, ), (1, ))
    assert_size_stride(arg873_1, (36, ), (1, ))
    assert_size_stride(arg874_1, (36, ), (1, ))
    assert_size_stride(arg875_1, (36, ), (1, ))
    assert_size_stride(arg876_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg877_1, (36, ), (1, ))
    assert_size_stride(arg878_1, (36, ), (1, ))
    assert_size_stride(arg879_1, (36, ), (1, ))
    assert_size_stride(arg880_1, (36, ), (1, ))
    assert_size_stride(arg881_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg882_1, (36, ), (1, ))
    assert_size_stride(arg883_1, (36, ), (1, ))
    assert_size_stride(arg884_1, (36, ), (1, ))
    assert_size_stride(arg885_1, (36, ), (1, ))
    assert_size_stride(arg886_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg887_1, (72, ), (1, ))
    assert_size_stride(arg888_1, (72, ), (1, ))
    assert_size_stride(arg889_1, (72, ), (1, ))
    assert_size_stride(arg890_1, (72, ), (1, ))
    assert_size_stride(arg891_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg892_1, (72, ), (1, ))
    assert_size_stride(arg893_1, (72, ), (1, ))
    assert_size_stride(arg894_1, (72, ), (1, ))
    assert_size_stride(arg895_1, (72, ), (1, ))
    assert_size_stride(arg896_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg897_1, (72, ), (1, ))
    assert_size_stride(arg898_1, (72, ), (1, ))
    assert_size_stride(arg899_1, (72, ), (1, ))
    assert_size_stride(arg900_1, (72, ), (1, ))
    assert_size_stride(arg901_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg902_1, (72, ), (1, ))
    assert_size_stride(arg903_1, (72, ), (1, ))
    assert_size_stride(arg904_1, (72, ), (1, ))
    assert_size_stride(arg905_1, (72, ), (1, ))
    assert_size_stride(arg906_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg907_1, (72, ), (1, ))
    assert_size_stride(arg908_1, (72, ), (1, ))
    assert_size_stride(arg909_1, (72, ), (1, ))
    assert_size_stride(arg910_1, (72, ), (1, ))
    assert_size_stride(arg911_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg912_1, (72, ), (1, ))
    assert_size_stride(arg913_1, (72, ), (1, ))
    assert_size_stride(arg914_1, (72, ), (1, ))
    assert_size_stride(arg915_1, (72, ), (1, ))
    assert_size_stride(arg916_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg917_1, (72, ), (1, ))
    assert_size_stride(arg918_1, (72, ), (1, ))
    assert_size_stride(arg919_1, (72, ), (1, ))
    assert_size_stride(arg920_1, (72, ), (1, ))
    assert_size_stride(arg921_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg922_1, (72, ), (1, ))
    assert_size_stride(arg923_1, (72, ), (1, ))
    assert_size_stride(arg924_1, (72, ), (1, ))
    assert_size_stride(arg925_1, (72, ), (1, ))
    assert_size_stride(arg926_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg927_1, (144, ), (1, ))
    assert_size_stride(arg928_1, (144, ), (1, ))
    assert_size_stride(arg929_1, (144, ), (1, ))
    assert_size_stride(arg930_1, (144, ), (1, ))
    assert_size_stride(arg931_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg932_1, (144, ), (1, ))
    assert_size_stride(arg933_1, (144, ), (1, ))
    assert_size_stride(arg934_1, (144, ), (1, ))
    assert_size_stride(arg935_1, (144, ), (1, ))
    assert_size_stride(arg936_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg937_1, (144, ), (1, ))
    assert_size_stride(arg938_1, (144, ), (1, ))
    assert_size_stride(arg939_1, (144, ), (1, ))
    assert_size_stride(arg940_1, (144, ), (1, ))
    assert_size_stride(arg941_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg942_1, (144, ), (1, ))
    assert_size_stride(arg943_1, (144, ), (1, ))
    assert_size_stride(arg944_1, (144, ), (1, ))
    assert_size_stride(arg945_1, (144, ), (1, ))
    assert_size_stride(arg946_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg947_1, (144, ), (1, ))
    assert_size_stride(arg948_1, (144, ), (1, ))
    assert_size_stride(arg949_1, (144, ), (1, ))
    assert_size_stride(arg950_1, (144, ), (1, ))
    assert_size_stride(arg951_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg952_1, (144, ), (1, ))
    assert_size_stride(arg953_1, (144, ), (1, ))
    assert_size_stride(arg954_1, (144, ), (1, ))
    assert_size_stride(arg955_1, (144, ), (1, ))
    assert_size_stride(arg956_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg957_1, (144, ), (1, ))
    assert_size_stride(arg958_1, (144, ), (1, ))
    assert_size_stride(arg959_1, (144, ), (1, ))
    assert_size_stride(arg960_1, (144, ), (1, ))
    assert_size_stride(arg961_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg962_1, (144, ), (1, ))
    assert_size_stride(arg963_1, (144, ), (1, ))
    assert_size_stride(arg964_1, (144, ), (1, ))
    assert_size_stride(arg965_1, (144, ), (1, ))
    assert_size_stride(arg966_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg967_1, (18, ), (1, ))
    assert_size_stride(arg968_1, (18, ), (1, ))
    assert_size_stride(arg969_1, (18, ), (1, ))
    assert_size_stride(arg970_1, (18, ), (1, ))
    assert_size_stride(arg971_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg972_1, (18, ), (1, ))
    assert_size_stride(arg973_1, (18, ), (1, ))
    assert_size_stride(arg974_1, (18, ), (1, ))
    assert_size_stride(arg975_1, (18, ), (1, ))
    assert_size_stride(arg976_1, (18, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg977_1, (18, ), (1, ))
    assert_size_stride(arg978_1, (18, ), (1, ))
    assert_size_stride(arg979_1, (18, ), (1, ))
    assert_size_stride(arg980_1, (18, ), (1, ))
    assert_size_stride(arg981_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg982_1, (36, ), (1, ))
    assert_size_stride(arg983_1, (36, ), (1, ))
    assert_size_stride(arg984_1, (36, ), (1, ))
    assert_size_stride(arg985_1, (36, ), (1, ))
    assert_size_stride(arg986_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg987_1, (36, ), (1, ))
    assert_size_stride(arg988_1, (36, ), (1, ))
    assert_size_stride(arg989_1, (36, ), (1, ))
    assert_size_stride(arg990_1, (36, ), (1, ))
    assert_size_stride(arg991_1, (36, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg992_1, (36, ), (1, ))
    assert_size_stride(arg993_1, (36, ), (1, ))
    assert_size_stride(arg994_1, (36, ), (1, ))
    assert_size_stride(arg995_1, (36, ), (1, ))
    assert_size_stride(arg996_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg997_1, (18, ), (1, ))
    assert_size_stride(arg998_1, (18, ), (1, ))
    assert_size_stride(arg999_1, (18, ), (1, ))
    assert_size_stride(arg1000_1, (18, ), (1, ))
    assert_size_stride(arg1001_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1002_1, (72, ), (1, ))
    assert_size_stride(arg1003_1, (72, ), (1, ))
    assert_size_stride(arg1004_1, (72, ), (1, ))
    assert_size_stride(arg1005_1, (72, ), (1, ))
    assert_size_stride(arg1006_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1007_1, (72, ), (1, ))
    assert_size_stride(arg1008_1, (72, ), (1, ))
    assert_size_stride(arg1009_1, (72, ), (1, ))
    assert_size_stride(arg1010_1, (72, ), (1, ))
    assert_size_stride(arg1011_1, (72, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg1012_1, (72, ), (1, ))
    assert_size_stride(arg1013_1, (72, ), (1, ))
    assert_size_stride(arg1014_1, (72, ), (1, ))
    assert_size_stride(arg1015_1, (72, ), (1, ))
    assert_size_stride(arg1016_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1017_1, (18, ), (1, ))
    assert_size_stride(arg1018_1, (18, ), (1, ))
    assert_size_stride(arg1019_1, (18, ), (1, ))
    assert_size_stride(arg1020_1, (18, ), (1, ))
    assert_size_stride(arg1021_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1022_1, (18, ), (1, ))
    assert_size_stride(arg1023_1, (18, ), (1, ))
    assert_size_stride(arg1024_1, (18, ), (1, ))
    assert_size_stride(arg1025_1, (18, ), (1, ))
    assert_size_stride(arg1026_1, (144, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1027_1, (144, ), (1, ))
    assert_size_stride(arg1028_1, (144, ), (1, ))
    assert_size_stride(arg1029_1, (144, ), (1, ))
    assert_size_stride(arg1030_1, (144, ), (1, ))
    assert_size_stride(arg1031_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1032_1, (36, ), (1, ))
    assert_size_stride(arg1033_1, (36, ), (1, ))
    assert_size_stride(arg1034_1, (36, ), (1, ))
    assert_size_stride(arg1035_1, (36, ), (1, ))
    assert_size_stride(arg1036_1, (144, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1037_1, (144, ), (1, ))
    assert_size_stride(arg1038_1, (144, ), (1, ))
    assert_size_stride(arg1039_1, (144, ), (1, ))
    assert_size_stride(arg1040_1, (144, ), (1, ))
    assert_size_stride(arg1041_1, (144, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1042_1, (144, ), (1, ))
    assert_size_stride(arg1043_1, (144, ), (1, ))
    assert_size_stride(arg1044_1, (144, ), (1, ))
    assert_size_stride(arg1045_1, (144, ), (1, ))
    assert_size_stride(arg1046_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1047_1, (18, ), (1, ))
    assert_size_stride(arg1048_1, (18, ), (1, ))
    assert_size_stride(arg1049_1, (18, ), (1, ))
    assert_size_stride(arg1050_1, (18, ), (1, ))
    assert_size_stride(arg1051_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1052_1, (18, ), (1, ))
    assert_size_stride(arg1053_1, (18, ), (1, ))
    assert_size_stride(arg1054_1, (18, ), (1, ))
    assert_size_stride(arg1055_1, (18, ), (1, ))
    assert_size_stride(arg1056_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1057_1, (18, ), (1, ))
    assert_size_stride(arg1058_1, (18, ), (1, ))
    assert_size_stride(arg1059_1, (18, ), (1, ))
    assert_size_stride(arg1060_1, (18, ), (1, ))
    assert_size_stride(arg1061_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1062_1, (18, ), (1, ))
    assert_size_stride(arg1063_1, (18, ), (1, ))
    assert_size_stride(arg1064_1, (18, ), (1, ))
    assert_size_stride(arg1065_1, (18, ), (1, ))
    assert_size_stride(arg1066_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1067_1, (18, ), (1, ))
    assert_size_stride(arg1068_1, (18, ), (1, ))
    assert_size_stride(arg1069_1, (18, ), (1, ))
    assert_size_stride(arg1070_1, (18, ), (1, ))
    assert_size_stride(arg1071_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1072_1, (18, ), (1, ))
    assert_size_stride(arg1073_1, (18, ), (1, ))
    assert_size_stride(arg1074_1, (18, ), (1, ))
    assert_size_stride(arg1075_1, (18, ), (1, ))
    assert_size_stride(arg1076_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1077_1, (18, ), (1, ))
    assert_size_stride(arg1078_1, (18, ), (1, ))
    assert_size_stride(arg1079_1, (18, ), (1, ))
    assert_size_stride(arg1080_1, (18, ), (1, ))
    assert_size_stride(arg1081_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1082_1, (18, ), (1, ))
    assert_size_stride(arg1083_1, (18, ), (1, ))
    assert_size_stride(arg1084_1, (18, ), (1, ))
    assert_size_stride(arg1085_1, (18, ), (1, ))
    assert_size_stride(arg1086_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1087_1, (36, ), (1, ))
    assert_size_stride(arg1088_1, (36, ), (1, ))
    assert_size_stride(arg1089_1, (36, ), (1, ))
    assert_size_stride(arg1090_1, (36, ), (1, ))
    assert_size_stride(arg1091_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1092_1, (36, ), (1, ))
    assert_size_stride(arg1093_1, (36, ), (1, ))
    assert_size_stride(arg1094_1, (36, ), (1, ))
    assert_size_stride(arg1095_1, (36, ), (1, ))
    assert_size_stride(arg1096_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1097_1, (36, ), (1, ))
    assert_size_stride(arg1098_1, (36, ), (1, ))
    assert_size_stride(arg1099_1, (36, ), (1, ))
    assert_size_stride(arg1100_1, (36, ), (1, ))
    assert_size_stride(arg1101_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1102_1, (36, ), (1, ))
    assert_size_stride(arg1103_1, (36, ), (1, ))
    assert_size_stride(arg1104_1, (36, ), (1, ))
    assert_size_stride(arg1105_1, (36, ), (1, ))
    assert_size_stride(arg1106_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1107_1, (36, ), (1, ))
    assert_size_stride(arg1108_1, (36, ), (1, ))
    assert_size_stride(arg1109_1, (36, ), (1, ))
    assert_size_stride(arg1110_1, (36, ), (1, ))
    assert_size_stride(arg1111_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1112_1, (36, ), (1, ))
    assert_size_stride(arg1113_1, (36, ), (1, ))
    assert_size_stride(arg1114_1, (36, ), (1, ))
    assert_size_stride(arg1115_1, (36, ), (1, ))
    assert_size_stride(arg1116_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1117_1, (36, ), (1, ))
    assert_size_stride(arg1118_1, (36, ), (1, ))
    assert_size_stride(arg1119_1, (36, ), (1, ))
    assert_size_stride(arg1120_1, (36, ), (1, ))
    assert_size_stride(arg1121_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1122_1, (36, ), (1, ))
    assert_size_stride(arg1123_1, (36, ), (1, ))
    assert_size_stride(arg1124_1, (36, ), (1, ))
    assert_size_stride(arg1125_1, (36, ), (1, ))
    assert_size_stride(arg1126_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1127_1, (72, ), (1, ))
    assert_size_stride(arg1128_1, (72, ), (1, ))
    assert_size_stride(arg1129_1, (72, ), (1, ))
    assert_size_stride(arg1130_1, (72, ), (1, ))
    assert_size_stride(arg1131_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1132_1, (72, ), (1, ))
    assert_size_stride(arg1133_1, (72, ), (1, ))
    assert_size_stride(arg1134_1, (72, ), (1, ))
    assert_size_stride(arg1135_1, (72, ), (1, ))
    assert_size_stride(arg1136_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1137_1, (72, ), (1, ))
    assert_size_stride(arg1138_1, (72, ), (1, ))
    assert_size_stride(arg1139_1, (72, ), (1, ))
    assert_size_stride(arg1140_1, (72, ), (1, ))
    assert_size_stride(arg1141_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1142_1, (72, ), (1, ))
    assert_size_stride(arg1143_1, (72, ), (1, ))
    assert_size_stride(arg1144_1, (72, ), (1, ))
    assert_size_stride(arg1145_1, (72, ), (1, ))
    assert_size_stride(arg1146_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1147_1, (72, ), (1, ))
    assert_size_stride(arg1148_1, (72, ), (1, ))
    assert_size_stride(arg1149_1, (72, ), (1, ))
    assert_size_stride(arg1150_1, (72, ), (1, ))
    assert_size_stride(arg1151_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1152_1, (72, ), (1, ))
    assert_size_stride(arg1153_1, (72, ), (1, ))
    assert_size_stride(arg1154_1, (72, ), (1, ))
    assert_size_stride(arg1155_1, (72, ), (1, ))
    assert_size_stride(arg1156_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1157_1, (72, ), (1, ))
    assert_size_stride(arg1158_1, (72, ), (1, ))
    assert_size_stride(arg1159_1, (72, ), (1, ))
    assert_size_stride(arg1160_1, (72, ), (1, ))
    assert_size_stride(arg1161_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1162_1, (72, ), (1, ))
    assert_size_stride(arg1163_1, (72, ), (1, ))
    assert_size_stride(arg1164_1, (72, ), (1, ))
    assert_size_stride(arg1165_1, (72, ), (1, ))
    assert_size_stride(arg1166_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1167_1, (144, ), (1, ))
    assert_size_stride(arg1168_1, (144, ), (1, ))
    assert_size_stride(arg1169_1, (144, ), (1, ))
    assert_size_stride(arg1170_1, (144, ), (1, ))
    assert_size_stride(arg1171_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1172_1, (144, ), (1, ))
    assert_size_stride(arg1173_1, (144, ), (1, ))
    assert_size_stride(arg1174_1, (144, ), (1, ))
    assert_size_stride(arg1175_1, (144, ), (1, ))
    assert_size_stride(arg1176_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1177_1, (144, ), (1, ))
    assert_size_stride(arg1178_1, (144, ), (1, ))
    assert_size_stride(arg1179_1, (144, ), (1, ))
    assert_size_stride(arg1180_1, (144, ), (1, ))
    assert_size_stride(arg1181_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1182_1, (144, ), (1, ))
    assert_size_stride(arg1183_1, (144, ), (1, ))
    assert_size_stride(arg1184_1, (144, ), (1, ))
    assert_size_stride(arg1185_1, (144, ), (1, ))
    assert_size_stride(arg1186_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1187_1, (144, ), (1, ))
    assert_size_stride(arg1188_1, (144, ), (1, ))
    assert_size_stride(arg1189_1, (144, ), (1, ))
    assert_size_stride(arg1190_1, (144, ), (1, ))
    assert_size_stride(arg1191_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1192_1, (144, ), (1, ))
    assert_size_stride(arg1193_1, (144, ), (1, ))
    assert_size_stride(arg1194_1, (144, ), (1, ))
    assert_size_stride(arg1195_1, (144, ), (1, ))
    assert_size_stride(arg1196_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1197_1, (144, ), (1, ))
    assert_size_stride(arg1198_1, (144, ), (1, ))
    assert_size_stride(arg1199_1, (144, ), (1, ))
    assert_size_stride(arg1200_1, (144, ), (1, ))
    assert_size_stride(arg1201_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1202_1, (144, ), (1, ))
    assert_size_stride(arg1203_1, (144, ), (1, ))
    assert_size_stride(arg1204_1, (144, ), (1, ))
    assert_size_stride(arg1205_1, (144, ), (1, ))
    assert_size_stride(arg1206_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg1207_1, (18, ), (1, ))
    assert_size_stride(arg1208_1, (18, ), (1, ))
    assert_size_stride(arg1209_1, (18, ), (1, ))
    assert_size_stride(arg1210_1, (18, ), (1, ))
    assert_size_stride(arg1211_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg1212_1, (18, ), (1, ))
    assert_size_stride(arg1213_1, (18, ), (1, ))
    assert_size_stride(arg1214_1, (18, ), (1, ))
    assert_size_stride(arg1215_1, (18, ), (1, ))
    assert_size_stride(arg1216_1, (18, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg1217_1, (18, ), (1, ))
    assert_size_stride(arg1218_1, (18, ), (1, ))
    assert_size_stride(arg1219_1, (18, ), (1, ))
    assert_size_stride(arg1220_1, (18, ), (1, ))
    assert_size_stride(arg1221_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1222_1, (36, ), (1, ))
    assert_size_stride(arg1223_1, (36, ), (1, ))
    assert_size_stride(arg1224_1, (36, ), (1, ))
    assert_size_stride(arg1225_1, (36, ), (1, ))
    assert_size_stride(arg1226_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg1227_1, (36, ), (1, ))
    assert_size_stride(arg1228_1, (36, ), (1, ))
    assert_size_stride(arg1229_1, (36, ), (1, ))
    assert_size_stride(arg1230_1, (36, ), (1, ))
    assert_size_stride(arg1231_1, (36, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg1232_1, (36, ), (1, ))
    assert_size_stride(arg1233_1, (36, ), (1, ))
    assert_size_stride(arg1234_1, (36, ), (1, ))
    assert_size_stride(arg1235_1, (36, ), (1, ))
    assert_size_stride(arg1236_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1237_1, (18, ), (1, ))
    assert_size_stride(arg1238_1, (18, ), (1, ))
    assert_size_stride(arg1239_1, (18, ), (1, ))
    assert_size_stride(arg1240_1, (18, ), (1, ))
    assert_size_stride(arg1241_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1242_1, (72, ), (1, ))
    assert_size_stride(arg1243_1, (72, ), (1, ))
    assert_size_stride(arg1244_1, (72, ), (1, ))
    assert_size_stride(arg1245_1, (72, ), (1, ))
    assert_size_stride(arg1246_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1247_1, (72, ), (1, ))
    assert_size_stride(arg1248_1, (72, ), (1, ))
    assert_size_stride(arg1249_1, (72, ), (1, ))
    assert_size_stride(arg1250_1, (72, ), (1, ))
    assert_size_stride(arg1251_1, (72, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg1252_1, (72, ), (1, ))
    assert_size_stride(arg1253_1, (72, ), (1, ))
    assert_size_stride(arg1254_1, (72, ), (1, ))
    assert_size_stride(arg1255_1, (72, ), (1, ))
    assert_size_stride(arg1256_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1257_1, (18, ), (1, ))
    assert_size_stride(arg1258_1, (18, ), (1, ))
    assert_size_stride(arg1259_1, (18, ), (1, ))
    assert_size_stride(arg1260_1, (18, ), (1, ))
    assert_size_stride(arg1261_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1262_1, (18, ), (1, ))
    assert_size_stride(arg1263_1, (18, ), (1, ))
    assert_size_stride(arg1264_1, (18, ), (1, ))
    assert_size_stride(arg1265_1, (18, ), (1, ))
    assert_size_stride(arg1266_1, (144, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1267_1, (144, ), (1, ))
    assert_size_stride(arg1268_1, (144, ), (1, ))
    assert_size_stride(arg1269_1, (144, ), (1, ))
    assert_size_stride(arg1270_1, (144, ), (1, ))
    assert_size_stride(arg1271_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1272_1, (36, ), (1, ))
    assert_size_stride(arg1273_1, (36, ), (1, ))
    assert_size_stride(arg1274_1, (36, ), (1, ))
    assert_size_stride(arg1275_1, (36, ), (1, ))
    assert_size_stride(arg1276_1, (144, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1277_1, (144, ), (1, ))
    assert_size_stride(arg1278_1, (144, ), (1, ))
    assert_size_stride(arg1279_1, (144, ), (1, ))
    assert_size_stride(arg1280_1, (144, ), (1, ))
    assert_size_stride(arg1281_1, (144, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1282_1, (144, ), (1, ))
    assert_size_stride(arg1283_1, (144, ), (1, ))
    assert_size_stride(arg1284_1, (144, ), (1, ))
    assert_size_stride(arg1285_1, (144, ), (1, ))
    assert_size_stride(arg1286_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1287_1, (18, ), (1, ))
    assert_size_stride(arg1288_1, (18, ), (1, ))
    assert_size_stride(arg1289_1, (18, ), (1, ))
    assert_size_stride(arg1290_1, (18, ), (1, ))
    assert_size_stride(arg1291_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1292_1, (18, ), (1, ))
    assert_size_stride(arg1293_1, (18, ), (1, ))
    assert_size_stride(arg1294_1, (18, ), (1, ))
    assert_size_stride(arg1295_1, (18, ), (1, ))
    assert_size_stride(arg1296_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1297_1, (18, ), (1, ))
    assert_size_stride(arg1298_1, (18, ), (1, ))
    assert_size_stride(arg1299_1, (18, ), (1, ))
    assert_size_stride(arg1300_1, (18, ), (1, ))
    assert_size_stride(arg1301_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1302_1, (18, ), (1, ))
    assert_size_stride(arg1303_1, (18, ), (1, ))
    assert_size_stride(arg1304_1, (18, ), (1, ))
    assert_size_stride(arg1305_1, (18, ), (1, ))
    assert_size_stride(arg1306_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1307_1, (18, ), (1, ))
    assert_size_stride(arg1308_1, (18, ), (1, ))
    assert_size_stride(arg1309_1, (18, ), (1, ))
    assert_size_stride(arg1310_1, (18, ), (1, ))
    assert_size_stride(arg1311_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1312_1, (18, ), (1, ))
    assert_size_stride(arg1313_1, (18, ), (1, ))
    assert_size_stride(arg1314_1, (18, ), (1, ))
    assert_size_stride(arg1315_1, (18, ), (1, ))
    assert_size_stride(arg1316_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1317_1, (18, ), (1, ))
    assert_size_stride(arg1318_1, (18, ), (1, ))
    assert_size_stride(arg1319_1, (18, ), (1, ))
    assert_size_stride(arg1320_1, (18, ), (1, ))
    assert_size_stride(arg1321_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1322_1, (18, ), (1, ))
    assert_size_stride(arg1323_1, (18, ), (1, ))
    assert_size_stride(arg1324_1, (18, ), (1, ))
    assert_size_stride(arg1325_1, (18, ), (1, ))
    assert_size_stride(arg1326_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1327_1, (36, ), (1, ))
    assert_size_stride(arg1328_1, (36, ), (1, ))
    assert_size_stride(arg1329_1, (36, ), (1, ))
    assert_size_stride(arg1330_1, (36, ), (1, ))
    assert_size_stride(arg1331_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1332_1, (36, ), (1, ))
    assert_size_stride(arg1333_1, (36, ), (1, ))
    assert_size_stride(arg1334_1, (36, ), (1, ))
    assert_size_stride(arg1335_1, (36, ), (1, ))
    assert_size_stride(arg1336_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1337_1, (36, ), (1, ))
    assert_size_stride(arg1338_1, (36, ), (1, ))
    assert_size_stride(arg1339_1, (36, ), (1, ))
    assert_size_stride(arg1340_1, (36, ), (1, ))
    assert_size_stride(arg1341_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1342_1, (36, ), (1, ))
    assert_size_stride(arg1343_1, (36, ), (1, ))
    assert_size_stride(arg1344_1, (36, ), (1, ))
    assert_size_stride(arg1345_1, (36, ), (1, ))
    assert_size_stride(arg1346_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1347_1, (36, ), (1, ))
    assert_size_stride(arg1348_1, (36, ), (1, ))
    assert_size_stride(arg1349_1, (36, ), (1, ))
    assert_size_stride(arg1350_1, (36, ), (1, ))
    assert_size_stride(arg1351_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1352_1, (36, ), (1, ))
    assert_size_stride(arg1353_1, (36, ), (1, ))
    assert_size_stride(arg1354_1, (36, ), (1, ))
    assert_size_stride(arg1355_1, (36, ), (1, ))
    assert_size_stride(arg1356_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1357_1, (36, ), (1, ))
    assert_size_stride(arg1358_1, (36, ), (1, ))
    assert_size_stride(arg1359_1, (36, ), (1, ))
    assert_size_stride(arg1360_1, (36, ), (1, ))
    assert_size_stride(arg1361_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1362_1, (36, ), (1, ))
    assert_size_stride(arg1363_1, (36, ), (1, ))
    assert_size_stride(arg1364_1, (36, ), (1, ))
    assert_size_stride(arg1365_1, (36, ), (1, ))
    assert_size_stride(arg1366_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1367_1, (72, ), (1, ))
    assert_size_stride(arg1368_1, (72, ), (1, ))
    assert_size_stride(arg1369_1, (72, ), (1, ))
    assert_size_stride(arg1370_1, (72, ), (1, ))
    assert_size_stride(arg1371_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1372_1, (72, ), (1, ))
    assert_size_stride(arg1373_1, (72, ), (1, ))
    assert_size_stride(arg1374_1, (72, ), (1, ))
    assert_size_stride(arg1375_1, (72, ), (1, ))
    assert_size_stride(arg1376_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1377_1, (72, ), (1, ))
    assert_size_stride(arg1378_1, (72, ), (1, ))
    assert_size_stride(arg1379_1, (72, ), (1, ))
    assert_size_stride(arg1380_1, (72, ), (1, ))
    assert_size_stride(arg1381_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1382_1, (72, ), (1, ))
    assert_size_stride(arg1383_1, (72, ), (1, ))
    assert_size_stride(arg1384_1, (72, ), (1, ))
    assert_size_stride(arg1385_1, (72, ), (1, ))
    assert_size_stride(arg1386_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1387_1, (72, ), (1, ))
    assert_size_stride(arg1388_1, (72, ), (1, ))
    assert_size_stride(arg1389_1, (72, ), (1, ))
    assert_size_stride(arg1390_1, (72, ), (1, ))
    assert_size_stride(arg1391_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1392_1, (72, ), (1, ))
    assert_size_stride(arg1393_1, (72, ), (1, ))
    assert_size_stride(arg1394_1, (72, ), (1, ))
    assert_size_stride(arg1395_1, (72, ), (1, ))
    assert_size_stride(arg1396_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1397_1, (72, ), (1, ))
    assert_size_stride(arg1398_1, (72, ), (1, ))
    assert_size_stride(arg1399_1, (72, ), (1, ))
    assert_size_stride(arg1400_1, (72, ), (1, ))
    assert_size_stride(arg1401_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1402_1, (72, ), (1, ))
    assert_size_stride(arg1403_1, (72, ), (1, ))
    assert_size_stride(arg1404_1, (72, ), (1, ))
    assert_size_stride(arg1405_1, (72, ), (1, ))
    assert_size_stride(arg1406_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1407_1, (144, ), (1, ))
    assert_size_stride(arg1408_1, (144, ), (1, ))
    assert_size_stride(arg1409_1, (144, ), (1, ))
    assert_size_stride(arg1410_1, (144, ), (1, ))
    assert_size_stride(arg1411_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1412_1, (144, ), (1, ))
    assert_size_stride(arg1413_1, (144, ), (1, ))
    assert_size_stride(arg1414_1, (144, ), (1, ))
    assert_size_stride(arg1415_1, (144, ), (1, ))
    assert_size_stride(arg1416_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1417_1, (144, ), (1, ))
    assert_size_stride(arg1418_1, (144, ), (1, ))
    assert_size_stride(arg1419_1, (144, ), (1, ))
    assert_size_stride(arg1420_1, (144, ), (1, ))
    assert_size_stride(arg1421_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1422_1, (144, ), (1, ))
    assert_size_stride(arg1423_1, (144, ), (1, ))
    assert_size_stride(arg1424_1, (144, ), (1, ))
    assert_size_stride(arg1425_1, (144, ), (1, ))
    assert_size_stride(arg1426_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1427_1, (144, ), (1, ))
    assert_size_stride(arg1428_1, (144, ), (1, ))
    assert_size_stride(arg1429_1, (144, ), (1, ))
    assert_size_stride(arg1430_1, (144, ), (1, ))
    assert_size_stride(arg1431_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1432_1, (144, ), (1, ))
    assert_size_stride(arg1433_1, (144, ), (1, ))
    assert_size_stride(arg1434_1, (144, ), (1, ))
    assert_size_stride(arg1435_1, (144, ), (1, ))
    assert_size_stride(arg1436_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1437_1, (144, ), (1, ))
    assert_size_stride(arg1438_1, (144, ), (1, ))
    assert_size_stride(arg1439_1, (144, ), (1, ))
    assert_size_stride(arg1440_1, (144, ), (1, ))
    assert_size_stride(arg1441_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg1442_1, (144, ), (1, ))
    assert_size_stride(arg1443_1, (144, ), (1, ))
    assert_size_stride(arg1444_1, (144, ), (1, ))
    assert_size_stride(arg1445_1, (144, ), (1, ))
    assert_size_stride(arg1446_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg1447_1, (18, ), (1, ))
    assert_size_stride(arg1448_1, (18, ), (1, ))
    assert_size_stride(arg1449_1, (18, ), (1, ))
    assert_size_stride(arg1450_1, (18, ), (1, ))
    assert_size_stride(arg1451_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg1452_1, (18, ), (1, ))
    assert_size_stride(arg1453_1, (18, ), (1, ))
    assert_size_stride(arg1454_1, (18, ), (1, ))
    assert_size_stride(arg1455_1, (18, ), (1, ))
    assert_size_stride(arg1456_1, (18, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg1457_1, (18, ), (1, ))
    assert_size_stride(arg1458_1, (18, ), (1, ))
    assert_size_stride(arg1459_1, (18, ), (1, ))
    assert_size_stride(arg1460_1, (18, ), (1, ))
    assert_size_stride(arg1461_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1462_1, (36, ), (1, ))
    assert_size_stride(arg1463_1, (36, ), (1, ))
    assert_size_stride(arg1464_1, (36, ), (1, ))
    assert_size_stride(arg1465_1, (36, ), (1, ))
    assert_size_stride(arg1466_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg1467_1, (36, ), (1, ))
    assert_size_stride(arg1468_1, (36, ), (1, ))
    assert_size_stride(arg1469_1, (36, ), (1, ))
    assert_size_stride(arg1470_1, (36, ), (1, ))
    assert_size_stride(arg1471_1, (36, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg1472_1, (36, ), (1, ))
    assert_size_stride(arg1473_1, (36, ), (1, ))
    assert_size_stride(arg1474_1, (36, ), (1, ))
    assert_size_stride(arg1475_1, (36, ), (1, ))
    assert_size_stride(arg1476_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1477_1, (18, ), (1, ))
    assert_size_stride(arg1478_1, (18, ), (1, ))
    assert_size_stride(arg1479_1, (18, ), (1, ))
    assert_size_stride(arg1480_1, (18, ), (1, ))
    assert_size_stride(arg1481_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1482_1, (72, ), (1, ))
    assert_size_stride(arg1483_1, (72, ), (1, ))
    assert_size_stride(arg1484_1, (72, ), (1, ))
    assert_size_stride(arg1485_1, (72, ), (1, ))
    assert_size_stride(arg1486_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1487_1, (72, ), (1, ))
    assert_size_stride(arg1488_1, (72, ), (1, ))
    assert_size_stride(arg1489_1, (72, ), (1, ))
    assert_size_stride(arg1490_1, (72, ), (1, ))
    assert_size_stride(arg1491_1, (72, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg1492_1, (72, ), (1, ))
    assert_size_stride(arg1493_1, (72, ), (1, ))
    assert_size_stride(arg1494_1, (72, ), (1, ))
    assert_size_stride(arg1495_1, (72, ), (1, ))
    assert_size_stride(arg1496_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1497_1, (18, ), (1, ))
    assert_size_stride(arg1498_1, (18, ), (1, ))
    assert_size_stride(arg1499_1, (18, ), (1, ))
    assert_size_stride(arg1500_1, (18, ), (1, ))
    assert_size_stride(arg1501_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1502_1, (18, ), (1, ))
    assert_size_stride(arg1503_1, (18, ), (1, ))
    assert_size_stride(arg1504_1, (18, ), (1, ))
    assert_size_stride(arg1505_1, (18, ), (1, ))
    assert_size_stride(arg1506_1, (144, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg1507_1, (144, ), (1, ))
    assert_size_stride(arg1508_1, (144, ), (1, ))
    assert_size_stride(arg1509_1, (144, ), (1, ))
    assert_size_stride(arg1510_1, (144, ), (1, ))
    assert_size_stride(arg1511_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1512_1, (36, ), (1, ))
    assert_size_stride(arg1513_1, (36, ), (1, ))
    assert_size_stride(arg1514_1, (36, ), (1, ))
    assert_size_stride(arg1515_1, (36, ), (1, ))
    assert_size_stride(arg1516_1, (144, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg1517_1, (144, ), (1, ))
    assert_size_stride(arg1518_1, (144, ), (1, ))
    assert_size_stride(arg1519_1, (144, ), (1, ))
    assert_size_stride(arg1520_1, (144, ), (1, ))
    assert_size_stride(arg1521_1, (144, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg1522_1, (144, ), (1, ))
    assert_size_stride(arg1523_1, (144, ), (1, ))
    assert_size_stride(arg1524_1, (144, ), (1, ))
    assert_size_stride(arg1525_1, (144, ), (1, ))
    assert_size_stride(arg1526_1, (32, 18, 1, 1), (18, 1, 1, 1))
    assert_size_stride(arg1527_1, (32, ), (1, ))
    assert_size_stride(arg1528_1, (32, ), (1, ))
    assert_size_stride(arg1529_1, (32, ), (1, ))
    assert_size_stride(arg1530_1, (32, ), (1, ))
    assert_size_stride(arg1531_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg1532_1, (32, ), (1, ))
    assert_size_stride(arg1533_1, (32, ), (1, ))
    assert_size_stride(arg1534_1, (32, ), (1, ))
    assert_size_stride(arg1535_1, (32, ), (1, ))
    assert_size_stride(arg1536_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg1537_1, (128, ), (1, ))
    assert_size_stride(arg1538_1, (128, ), (1, ))
    assert_size_stride(arg1539_1, (128, ), (1, ))
    assert_size_stride(arg1540_1, (128, ), (1, ))
    assert_size_stride(arg1541_1, (128, 18, 1, 1), (18, 1, 1, 1))
    assert_size_stride(arg1542_1, (128, ), (1, ))
    assert_size_stride(arg1543_1, (128, ), (1, ))
    assert_size_stride(arg1544_1, (128, ), (1, ))
    assert_size_stride(arg1545_1, (128, ), (1, ))
    assert_size_stride(arg1546_1, (64, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg1547_1, (64, ), (1, ))
    assert_size_stride(arg1548_1, (64, ), (1, ))
    assert_size_stride(arg1549_1, (64, ), (1, ))
    assert_size_stride(arg1550_1, (64, ), (1, ))
    assert_size_stride(arg1551_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg1552_1, (64, ), (1, ))
    assert_size_stride(arg1553_1, (64, ), (1, ))
    assert_size_stride(arg1554_1, (64, ), (1, ))
    assert_size_stride(arg1555_1, (64, ), (1, ))
    assert_size_stride(arg1556_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg1557_1, (256, ), (1, ))
    assert_size_stride(arg1558_1, (256, ), (1, ))
    assert_size_stride(arg1559_1, (256, ), (1, ))
    assert_size_stride(arg1560_1, (256, ), (1, ))
    assert_size_stride(arg1561_1, (256, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg1562_1, (256, ), (1, ))
    assert_size_stride(arg1563_1, (256, ), (1, ))
    assert_size_stride(arg1564_1, (256, ), (1, ))
    assert_size_stride(arg1565_1, (256, ), (1, ))
    assert_size_stride(arg1566_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg1567_1, (256, ), (1, ))
    assert_size_stride(arg1568_1, (256, ), (1, ))
    assert_size_stride(arg1569_1, (256, ), (1, ))
    assert_size_stride(arg1570_1, (256, ), (1, ))
    assert_size_stride(arg1571_1, (256, ), (1, ))
    assert_size_stride(arg1572_1, (128, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg1573_1, (128, ), (1, ))
    assert_size_stride(arg1574_1, (128, ), (1, ))
    assert_size_stride(arg1575_1, (128, ), (1, ))
    assert_size_stride(arg1576_1, (128, ), (1, ))
    assert_size_stride(arg1577_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg1578_1, (128, ), (1, ))
    assert_size_stride(arg1579_1, (128, ), (1, ))
    assert_size_stride(arg1580_1, (128, ), (1, ))
    assert_size_stride(arg1581_1, (128, ), (1, ))
    assert_size_stride(arg1582_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg1583_1, (512, ), (1, ))
    assert_size_stride(arg1584_1, (512, ), (1, ))
    assert_size_stride(arg1585_1, (512, ), (1, ))
    assert_size_stride(arg1586_1, (512, ), (1, ))
    assert_size_stride(arg1587_1, (512, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg1588_1, (512, ), (1, ))
    assert_size_stride(arg1589_1, (512, ), (1, ))
    assert_size_stride(arg1590_1, (512, ), (1, ))
    assert_size_stride(arg1591_1, (512, ), (1, ))
    assert_size_stride(arg1592_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg1593_1, (512, ), (1, ))
    assert_size_stride(arg1594_1, (512, ), (1, ))
    assert_size_stride(arg1595_1, (512, ), (1, ))
    assert_size_stride(arg1596_1, (512, ), (1, ))
    assert_size_stride(arg1597_1, (512, ), (1, ))
    assert_size_stride(arg1598_1, (256, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg1599_1, (256, ), (1, ))
    assert_size_stride(arg1600_1, (256, ), (1, ))
    assert_size_stride(arg1601_1, (256, ), (1, ))
    assert_size_stride(arg1602_1, (256, ), (1, ))
    assert_size_stride(arg1603_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg1604_1, (256, ), (1, ))
    assert_size_stride(arg1605_1, (256, ), (1, ))
    assert_size_stride(arg1606_1, (256, ), (1, ))
    assert_size_stride(arg1607_1, (256, ), (1, ))
    assert_size_stride(arg1608_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg1609_1, (1024, ), (1, ))
    assert_size_stride(arg1610_1, (1024, ), (1, ))
    assert_size_stride(arg1611_1, (1024, ), (1, ))
    assert_size_stride(arg1612_1, (1024, ), (1, ))
    assert_size_stride(arg1613_1, (1024, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg1614_1, (1024, ), (1, ))
    assert_size_stride(arg1615_1, (1024, ), (1, ))
    assert_size_stride(arg1616_1, (1024, ), (1, ))
    assert_size_stride(arg1617_1, (1024, ), (1, ))
    assert_size_stride(arg1618_1, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg1619_1, (1024, ), (1, ))
    assert_size_stride(arg1620_1, (1024, ), (1, ))
    assert_size_stride(arg1621_1, (1024, ), (1, ))
    assert_size_stride(arg1622_1, (1024, ), (1, ))
    assert_size_stride(arg1623_1, (1024, ), (1, ))
    assert_size_stride(arg1624_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg1625_1, (2048, ), (1, ))
    assert_size_stride(arg1626_1, (2048, ), (1, ))
    assert_size_stride(arg1627_1, (2048, ), (1, ))
    assert_size_stride(arg1628_1, (2048, ), (1, ))
    assert_size_stride(arg1629_1, (2048, ), (1, ))
    assert_size_stride(arg1630_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg1631_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_818], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_818], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_818], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 112, 112), (802816, 1, 7168, 64))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_819, x_820], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_819, x_820, x_821], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg6_1, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [x_819, x_820, x_821], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 64, 56, 56), (200704, 1, 3584, 64))
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_822, x_823], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [x_824], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg11_1
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_825, x_826], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf8, arg12_1, arg13_1, arg14_1, arg15_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        buf9 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_825, x_826, x_827], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg16_1, buf9, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg16_1
        # Topologically Sorted Source Nodes: [x_825, x_826, x_827], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf10 = extern_kernels.convolution(buf8, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del buf8
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_828, x_829], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf11, arg17_1, arg18_1, arg19_1, arg20_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        # Topologically Sorted Source Nodes: [x_828, x_829, x_830], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg21_1
        del buf11
        # Topologically Sorted Source Nodes: [input_238], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf6, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg26_1
        del buf6
        buf14 = buf12; del buf12  # reuse
        buf15 = reinterpret_tensor(buf3, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_831, input_239, x_832, x_833], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf14, arg22_1, arg23_1, arg24_1, arg25_1, buf13, arg27_1, arg28_1, arg29_1, arg30_1, buf15, 6422528, grid=grid(6422528), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        del buf13
        del buf14
        # Topologically Sorted Source Nodes: [x_833, x_834], Original ATen: [aten.relu, aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg31_1
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_835, x_836], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf17, arg32_1, arg33_1, arg34_1, arg35_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        buf18 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [x_835, x_836, x_837], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg36_1, buf18, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg36_1
        # Topologically Sorted Source Nodes: [x_835, x_836, x_837], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del buf17
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_838, x_839], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf20, arg37_1, arg38_1, arg39_1, arg40_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        # Topologically Sorted Source Nodes: [x_838, x_839, x_840], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg41_1
        del buf20
        buf22 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_841, x_842, x_843], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf22, buf21, arg42_1, arg43_1, arg44_1, arg45_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        del buf21
        # Topologically Sorted Source Nodes: [x_844], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg46_1
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_845, x_846], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf24, arg47_1, arg48_1, arg49_1, arg50_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        buf25 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_845, x_846, x_847], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg51_1, buf25, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg51_1
        # Topologically Sorted Source Nodes: [x_845, x_846, x_847], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf26 = extern_kernels.convolution(buf24, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del buf24
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_848, x_849], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf27, arg52_1, arg53_1, arg54_1, arg55_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        # Topologically Sorted Source Nodes: [x_848, x_849, x_850], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg56_1
        del buf27
        buf29 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_851, x_852, x_853], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf29, buf28, arg57_1, arg58_1, arg59_1, arg60_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf28
        # Topologically Sorted Source Nodes: [x_854], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg61_1
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_855, x_856], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf31, arg62_1, arg63_1, arg64_1, arg65_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        buf32 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_855, x_856, x_857], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg66_1, buf32, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg66_1
        # Topologically Sorted Source Nodes: [x_855, x_856, x_857], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf33 = extern_kernels.convolution(buf31, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del buf31
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_858, x_859], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf34, arg67_1, arg68_1, arg69_1, arg70_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        # Topologically Sorted Source Nodes: [x_858, x_859, x_860], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg71_1
        del buf34
        buf36 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_861, x_862, x_863], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf36, buf35, arg72_1, arg73_1, arg74_1, arg75_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        del buf35
        buf37 = empty_strided_cuda((18, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_240], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(arg76_1, buf37, 4608, 9, grid=grid(4608, 9), stream=stream0)
        del arg76_1
        # Topologically Sorted Source Nodes: [input_240], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf36, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf37
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [input_241, input_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf39, arg77_1, arg78_1, arg79_1, arg80_1, 451584, grid=grid(451584), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        buf40 = empty_strided_cuda((18, 18, 3, 3), (162, 1, 54, 18), torch.float32)
        # Topologically Sorted Source Nodes: [x_864], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg86_1, buf40, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg86_1
        # Topologically Sorted Source Nodes: [x_864], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf39, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_865, x_866], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf42, arg87_1, arg88_1, arg89_1, arg90_1, 451584, grid=grid(451584), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        buf43 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_865, x_866, x_867], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg91_1, buf43, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg91_1
        # Topologically Sorted Source Nodes: [x_865, x_866, x_867], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf44 = extern_kernels.convolution(buf42, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf42
        buf45 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_868, x_869, x_870], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf45, buf44, arg92_1, arg93_1, arg94_1, arg95_1, 451584, grid=grid(451584), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf44
        buf46 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_871], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg96_1, buf46, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg96_1
        # Topologically Sorted Source Nodes: [x_871], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf45, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_872, x_873], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf48, arg97_1, arg98_1, arg99_1, arg100_1, 451584, grid=grid(451584), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        buf49 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_872, x_873, x_874], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg101_1, buf49, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg101_1
        # Topologically Sorted Source Nodes: [x_872, x_873, x_874], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf50 = extern_kernels.convolution(buf48, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf48
        buf51 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [x_875, x_876, x_877], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf51, buf50, arg102_1, arg103_1, arg104_1, arg105_1, 451584, grid=grid(451584), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        del buf50
        buf52 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [x_878], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg106_1, buf52, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg106_1
        # Topologically Sorted Source Nodes: [x_878], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf51, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf54 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [x_879, x_880], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf54, arg107_1, arg108_1, arg109_1, arg110_1, 451584, grid=grid(451584), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        buf55 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_879, x_880, x_881], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg111_1, buf55, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg111_1
        # Topologically Sorted Source Nodes: [x_879, x_880, x_881], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf56 = extern_kernels.convolution(buf54, buf55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf54
        buf57 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_882, x_883, x_884], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf57, buf56, arg112_1, arg113_1, arg114_1, arg115_1, 451584, grid=grid(451584), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        buf58 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_885], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg116_1, buf58, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg116_1
        # Topologically Sorted Source Nodes: [x_885], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf57, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_886, x_887], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf60, arg117_1, arg118_1, arg119_1, arg120_1, 451584, grid=grid(451584), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        buf61 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_886, x_887, x_888], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg121_1, buf61, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg121_1
        # Topologically Sorted Source Nodes: [x_886, x_887, x_888], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf62 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf64 = empty_strided_cuda((36, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_243], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(arg81_1, buf64, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg81_1
        # Topologically Sorted Source Nodes: [input_243], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf36, buf64, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf36
        del buf64
        buf66 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [input_244, input_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf66, arg82_1, arg83_1, arg84_1, arg85_1, 225792, grid=grid(225792), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        buf67 = empty_strided_cuda((36, 36, 3, 3), (324, 1, 108, 36), torch.float32)
        # Topologically Sorted Source Nodes: [x_892], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg126_1, buf67, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg126_1
        # Topologically Sorted Source Nodes: [x_892], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf66, buf67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf69 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_893, x_894], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf69, arg127_1, arg128_1, arg129_1, arg130_1, 225792, grid=grid(225792), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        buf70 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_893, x_894, x_895], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg131_1, buf70, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg131_1
        # Topologically Sorted Source Nodes: [x_893, x_894, x_895], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf71 = extern_kernels.convolution(buf69, buf70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf69
        buf72 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_896, x_897, x_898], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf72, buf71, arg132_1, arg133_1, arg134_1, arg135_1, 225792, grid=grid(225792), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        del buf71
        buf73 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_899], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg136_1, buf73, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg136_1
        # Topologically Sorted Source Nodes: [x_899], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf72, buf73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_900, x_901], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf75, arg137_1, arg138_1, arg139_1, arg140_1, 225792, grid=grid(225792), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        buf76 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_900, x_901, x_902], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg141_1, buf76, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg141_1
        # Topologically Sorted Source Nodes: [x_900, x_901, x_902], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf77 = extern_kernels.convolution(buf75, buf76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf75
        buf78 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_903, x_904, x_905], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf78, buf77, arg142_1, arg143_1, arg144_1, arg145_1, 225792, grid=grid(225792), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        del buf77
        buf79 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_906], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg146_1, buf79, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg146_1
        # Topologically Sorted Source Nodes: [x_906], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf78, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_907, x_908], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf81, arg147_1, arg148_1, arg149_1, arg150_1, 225792, grid=grid(225792), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        buf82 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [x_907, x_908, x_909], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg151_1, buf82, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg151_1
        # Topologically Sorted Source Nodes: [x_907, x_908, x_909], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf83 = extern_kernels.convolution(buf81, buf82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf81
        buf84 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_910, x_911, x_912], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf84, buf83, arg152_1, arg153_1, arg154_1, arg155_1, 225792, grid=grid(225792), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        del buf83
        buf85 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [x_913], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg156_1, buf85, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg156_1
        # Topologically Sorted Source Nodes: [x_913], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf84, buf85, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_914, x_915], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf87, arg157_1, arg158_1, arg159_1, arg160_1, 225792, grid=grid(225792), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        buf88 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_914, x_915, x_916], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg161_1, buf88, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg161_1
        # Topologically Sorted Source Nodes: [x_914, x_915, x_916], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf89 = extern_kernels.convolution(buf87, buf88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf87
        buf90 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_917, x_918, x_919], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf90, buf89, arg162_1, arg163_1, arg164_1, arg165_1, 225792, grid=grid(225792), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        del buf89
        # Topologically Sorted Source Nodes: [input_246], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 18, 28, 28), (14112, 1, 504, 18))
        del arg166_1
        buf92 = reinterpret_tensor(buf60, (8, 18, 56, 56), (56448, 3136, 56, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_247, input_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_15.run(buf91, arg167_1, arg168_1, arg169_1, arg170_1, buf92, 451584, grid=grid(451584), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        del buf91
        buf63 = buf57; del buf57  # reuse
        buf93 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_889, x_890, x_891, y_65, shortcut_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf63, buf62, arg122_1, arg123_1, arg124_1, arg125_1, buf92, buf93, 25088, 18, grid=grid(25088, 18), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        del buf62
        del buf92
        buf94 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_920], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg181_1, buf94, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg181_1
        # Topologically Sorted Source Nodes: [x_920], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf93, buf94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf96 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [x_921, x_922], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf96, arg182_1, arg183_1, arg184_1, arg185_1, 451584, grid=grid(451584), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        buf97 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_921, x_922, x_923], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg186_1, buf97, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg186_1
        # Topologically Sorted Source Nodes: [x_921, x_922, x_923], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf98 = extern_kernels.convolution(buf96, buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf96
        buf99 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_924, x_925, x_926], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf99, buf98, arg187_1, arg188_1, arg189_1, arg190_1, 451584, grid=grid(451584), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        del buf98
        buf100 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_927], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg191_1, buf100, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg191_1
        # Topologically Sorted Source Nodes: [x_927], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf99, buf100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf102 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_928, x_929], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf102, arg192_1, arg193_1, arg194_1, arg195_1, 451584, grid=grid(451584), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        buf103 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_928, x_929, x_930], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg196_1, buf103, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg196_1
        # Topologically Sorted Source Nodes: [x_928, x_929, x_930], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf104 = extern_kernels.convolution(buf102, buf103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf102
        buf105 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_931, x_932, x_933], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf105, arg197_1, arg198_1, arg199_1, arg200_1, buf99, 451584, grid=grid(451584), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        del buf99
        buf106 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [x_934], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg201_1, buf106, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg201_1
        # Topologically Sorted Source Nodes: [x_934], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf105, buf106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf108 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [x_935, x_936], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf108, arg202_1, arg203_1, arg204_1, arg205_1, 451584, grid=grid(451584), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        buf109 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_935, x_936, x_937], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg206_1, buf109, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg206_1
        # Topologically Sorted Source Nodes: [x_935, x_936, x_937], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf110 = extern_kernels.convolution(buf108, buf109, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf108
        buf111 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_938, x_939, x_940], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf111, buf110, arg207_1, arg208_1, arg209_1, arg210_1, 451584, grid=grid(451584), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        del buf110
        buf112 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_941], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg211_1, buf112, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg211_1
        # Topologically Sorted Source Nodes: [x_941], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf111, buf112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf114 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [x_942, x_943], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf114, arg212_1, arg213_1, arg214_1, arg215_1, 451584, grid=grid(451584), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        buf115 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_942, x_943, x_944], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg216_1, buf115, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg216_1
        # Topologically Sorted Source Nodes: [x_942, x_943, x_944], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf116 = extern_kernels.convolution(buf114, buf115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf114
        buf117 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [x_945, x_946, x_947], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf117, buf116, arg217_1, arg218_1, arg219_1, arg220_1, 451584, grid=grid(451584), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        del buf116
        buf118 = empty_strided_cuda((36, 18, 3, 3), (162, 1, 54, 18), torch.float32)
        # Topologically Sorted Source Nodes: [input_249], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(arg171_1, buf118, 648, 9, grid=grid(648, 9), stream=stream0)
        del arg171_1
        # Topologically Sorted Source Nodes: [input_249], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf63, buf118, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf120 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [input_250, y_66, shortcut_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf120, arg172_1, arg173_1, arg174_1, arg175_1, buf90, 225792, grid=grid(225792), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        del buf90
        buf121 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_948], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg221_1, buf121, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg221_1
        # Topologically Sorted Source Nodes: [x_948], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf120, buf121, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf123 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_949, x_950], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf123, arg222_1, arg223_1, arg224_1, arg225_1, 225792, grid=grid(225792), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        buf124 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [x_949, x_950, x_951], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg226_1, buf124, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg226_1
        # Topologically Sorted Source Nodes: [x_949, x_950, x_951], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf125 = extern_kernels.convolution(buf123, buf124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf123
        buf126 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [x_952, x_953, x_954], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf126, arg227_1, arg228_1, arg229_1, arg230_1, buf120, 225792, grid=grid(225792), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        buf127 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_955], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg231_1, buf127, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg231_1
        # Topologically Sorted Source Nodes: [x_955], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf126, buf127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf129 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_956, x_957], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf129, arg232_1, arg233_1, arg234_1, arg235_1, 225792, grid=grid(225792), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        buf130 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_956, x_957, x_958], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg236_1, buf130, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg236_1
        # Topologically Sorted Source Nodes: [x_956, x_957, x_958], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf131 = extern_kernels.convolution(buf129, buf130, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf129
        buf132 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_959, x_960, x_961], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf132, buf131, arg237_1, arg238_1, arg239_1, arg240_1, 225792, grid=grid(225792), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        del buf131
        buf133 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_962], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg241_1, buf133, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg241_1
        # Topologically Sorted Source Nodes: [x_962], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf132, buf133, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf135 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [x_963, x_964], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf135, arg242_1, arg243_1, arg244_1, arg245_1, 225792, grid=grid(225792), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        buf136 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [x_963, x_964, x_965], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg246_1, buf136, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg246_1
        # Topologically Sorted Source Nodes: [x_963, x_964, x_965], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf137 = extern_kernels.convolution(buf135, buf136, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf135
        buf138 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [x_966, x_967, x_968], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf138, buf137, arg247_1, arg248_1, arg249_1, arg250_1, 225792, grid=grid(225792), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        del buf137
        buf139 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_969], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg251_1, buf139, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg251_1
        # Topologically Sorted Source Nodes: [x_969], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf138, buf139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_970, x_971], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf141, arg252_1, arg253_1, arg254_1, arg255_1, 225792, grid=grid(225792), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        buf142 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_970, x_971, x_972], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg256_1, buf142, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg256_1
        # Topologically Sorted Source Nodes: [x_970, x_971, x_972], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf143 = extern_kernels.convolution(buf141, buf142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf141
        buf201 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [input_260], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(arg311_1, buf201, 648, 9, grid=grid(648, 9), stream=stream0)
        del arg311_1
        # Topologically Sorted Source Nodes: [input_260], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf117, buf201, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf147 = empty_strided_cuda((72, 36, 3, 3), (324, 1, 108, 36), torch.float32)
        # Topologically Sorted Source Nodes: [input_251], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg176_1, buf147, 2592, 9, grid=grid(2592, 9), stream=stream0)
        del arg176_1
        # Topologically Sorted Source Nodes: [input_251], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf120, buf147, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf149 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [input_252, input_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf149, arg177_1, arg178_1, arg179_1, arg180_1, 112896, grid=grid(112896), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        buf150 = empty_strided_cuda((72, 72, 3, 3), (648, 1, 216, 72), torch.float32)
        # Topologically Sorted Source Nodes: [x_976], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg261_1, buf150, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg261_1
        # Topologically Sorted Source Nodes: [x_976], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf149, buf150, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf152 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [x_977, x_978], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf152, arg262_1, arg263_1, arg264_1, arg265_1, 112896, grid=grid(112896), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        buf153 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [x_977, x_978, x_979], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg266_1, buf153, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg266_1
        # Topologically Sorted Source Nodes: [x_977, x_978, x_979], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf154 = extern_kernels.convolution(buf152, buf153, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf152
        buf155 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [x_980, x_981, x_982], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf155, buf154, arg267_1, arg268_1, arg269_1, arg270_1, 112896, grid=grid(112896), stream=stream0)
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        del buf154
        buf156 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [x_983], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg271_1, buf156, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg271_1
        # Topologically Sorted Source Nodes: [x_983], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf155, buf156, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf158 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_984, x_985], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf158, arg272_1, arg273_1, arg274_1, arg275_1, 112896, grid=grid(112896), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        buf159 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [x_984, x_985, x_986], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg276_1, buf159, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg276_1
        # Topologically Sorted Source Nodes: [x_984, x_985, x_986], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf160 = extern_kernels.convolution(buf158, buf159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf158
        buf161 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_987, x_988, x_989], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf161, buf160, arg277_1, arg278_1, arg279_1, arg280_1, 112896, grid=grid(112896), stream=stream0)
        del arg277_1
        del arg278_1
        del arg279_1
        del arg280_1
        del buf160
        buf162 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_990], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg281_1, buf162, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg281_1
        # Topologically Sorted Source Nodes: [x_990], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf161, buf162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf164 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_991, x_992], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf164, arg282_1, arg283_1, arg284_1, arg285_1, 112896, grid=grid(112896), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        buf165 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_991, x_992, x_993], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg286_1, buf165, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg286_1
        # Topologically Sorted Source Nodes: [x_991, x_992, x_993], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf166 = extern_kernels.convolution(buf164, buf165, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf164
        buf167 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_994, x_995, x_996], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf167, buf166, arg287_1, arg288_1, arg289_1, arg290_1, 112896, grid=grid(112896), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        del buf166
        buf168 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_997], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg291_1, buf168, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg291_1
        # Topologically Sorted Source Nodes: [x_997], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf167, buf168, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf170 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_998, x_999], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf170, arg292_1, arg293_1, arg294_1, arg295_1, 112896, grid=grid(112896), stream=stream0)
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        buf171 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_998, x_999, x_1000], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg296_1, buf171, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg296_1
        # Topologically Sorted Source Nodes: [x_998, x_999, x_1000], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf172 = extern_kernels.convolution(buf170, buf171, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf170
        buf173 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_1001, x_1002, x_1003], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf173, buf172, arg297_1, arg298_1, arg299_1, arg300_1, 112896, grid=grid(112896), stream=stream0)
        del arg297_1
        del arg298_1
        del arg299_1
        del arg300_1
        del buf172
        # Topologically Sorted Source Nodes: [input_262], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf173, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 36, 14, 14), (7056, 1, 504, 36))
        del arg316_1
        buf204 = reinterpret_tensor(buf120, (8, 36, 28, 28), (28224, 784, 28, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [input_263, input_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_24.run(buf203, arg317_1, arg318_1, arg319_1, arg320_1, buf204, 225792, grid=grid(225792), stream=stream0)
        del arg317_1
        del arg318_1
        del arg319_1
        del arg320_1
        del buf203
        buf144 = buf138; del buf138  # reuse
        buf205 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [x_973, x_974, x_975, input_261, y_69, y_70, shortcut_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf144, buf205, buf143, arg257_1, arg258_1, arg259_1, arg260_1, arg312_1, arg313_1, arg314_1, arg315_1, buf204, 6272, 36, grid=grid(6272, 36), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        del arg312_1
        del arg313_1
        del arg314_1
        del arg315_1
        del buf143
        del buf204
        # Topologically Sorted Source Nodes: [input_254], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 18, 28, 28), (14112, 1, 504, 18))
        del arg301_1
        # Topologically Sorted Source Nodes: [input_257], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (8, 18, 14, 14), (3528, 1, 252, 18))
        del arg306_1
        buf176 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [input_255, input_256, y_67, input_258, input_259, y_68, shortcut_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26.run(buf145, arg302_1, arg303_1, arg304_1, arg305_1, buf174, arg307_1, arg308_1, arg309_1, arg310_1, buf117, buf176, 144, 3136, grid=grid(144, 3136), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        del arg307_1
        del arg308_1
        del arg309_1
        del arg310_1
        del buf145
        del buf174
        buf177 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_1004], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg336_1, buf177, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg336_1
        # Topologically Sorted Source Nodes: [x_1004], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf179 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_1005, x_1006], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf179, arg337_1, arg338_1, arg339_1, arg340_1, 451584, grid=grid(451584), stream=stream0)
        del arg337_1
        del arg338_1
        del arg339_1
        del arg340_1
        buf180 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_1005, x_1006, x_1007], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg341_1, buf180, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg341_1
        # Topologically Sorted Source Nodes: [x_1005, x_1006, x_1007], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf181 = extern_kernels.convolution(buf179, buf180, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf179
        buf182 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [x_1008, x_1009, x_1010], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf182, buf181, arg342_1, arg343_1, arg344_1, arg345_1, 451584, grid=grid(451584), stream=stream0)
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        del buf181
        buf183 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [x_1011], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg346_1, buf183, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg346_1
        # Topologically Sorted Source Nodes: [x_1011], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf182, buf183, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf185 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [x_1012, x_1013], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf185, arg347_1, arg348_1, arg349_1, arg350_1, 451584, grid=grid(451584), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        del arg350_1
        buf186 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [x_1012, x_1013, x_1014], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg351_1, buf186, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg351_1
        # Topologically Sorted Source Nodes: [x_1012, x_1013, x_1014], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf187 = extern_kernels.convolution(buf185, buf186, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf185
        buf188 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [x_1015, x_1016, x_1017], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf188, buf187, arg352_1, arg353_1, arg354_1, arg355_1, 451584, grid=grid(451584), stream=stream0)
        del arg352_1
        del arg353_1
        del arg354_1
        del arg355_1
        del buf187
        buf189 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_1018], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg356_1, buf189, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg356_1
        # Topologically Sorted Source Nodes: [x_1018], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf188, buf189, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf191 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_1019, x_1020], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf191, arg357_1, arg358_1, arg359_1, arg360_1, 451584, grid=grid(451584), stream=stream0)
        del arg357_1
        del arg358_1
        del arg359_1
        del arg360_1
        buf192 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_1019, x_1020, x_1021], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg361_1, buf192, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg361_1
        # Topologically Sorted Source Nodes: [x_1019, x_1020, x_1021], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf193 = extern_kernels.convolution(buf191, buf192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf191
        buf194 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_1022, x_1023, x_1024], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf194, buf193, arg362_1, arg363_1, arg364_1, arg365_1, 451584, grid=grid(451584), stream=stream0)
        del arg362_1
        del arg363_1
        del arg364_1
        del arg365_1
        del buf193
        buf195 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [x_1025], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg366_1, buf195, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg366_1
        # Topologically Sorted Source Nodes: [x_1025], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf194, buf195, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf197 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [x_1026, x_1027], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf197, arg367_1, arg368_1, arg369_1, arg370_1, 451584, grid=grid(451584), stream=stream0)
        del arg367_1
        del arg368_1
        del arg369_1
        del arg370_1
        buf198 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_1026, x_1027, x_1028], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg371_1, buf198, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg371_1
        # Topologically Sorted Source Nodes: [x_1026, x_1027, x_1028], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf199 = extern_kernels.convolution(buf197, buf198, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf197
        buf200 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [x_1029, x_1030, x_1031], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf200, buf199, arg372_1, arg373_1, arg374_1, arg375_1, 451584, grid=grid(451584), stream=stream0)
        del arg372_1
        del arg373_1
        del arg374_1
        del arg375_1
        del buf199
        buf206 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [x_1032], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg376_1, buf206, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg376_1
        # Topologically Sorted Source Nodes: [x_1032], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf205, buf206, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf208 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_1033, x_1034], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf208, arg377_1, arg378_1, arg379_1, arg380_1, 225792, grid=grid(225792), stream=stream0)
        del arg377_1
        del arg378_1
        del arg379_1
        del arg380_1
        buf209 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [x_1033, x_1034, x_1035], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg381_1, buf209, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg381_1
        # Topologically Sorted Source Nodes: [x_1033, x_1034, x_1035], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf210 = extern_kernels.convolution(buf208, buf209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf208
        buf211 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [x_1036, x_1037, x_1038], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf211, buf210, arg382_1, arg383_1, arg384_1, arg385_1, 225792, grid=grid(225792), stream=stream0)
        del arg382_1
        del arg383_1
        del arg384_1
        del arg385_1
        del buf210
        buf212 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [x_1039], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg386_1, buf212, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg386_1
        # Topologically Sorted Source Nodes: [x_1039], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf211, buf212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf214 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [x_1040, x_1041], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf214, arg387_1, arg388_1, arg389_1, arg390_1, 225792, grid=grid(225792), stream=stream0)
        del arg387_1
        del arg388_1
        del arg389_1
        del arg390_1
        buf215 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [x_1040, x_1041, x_1042], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg391_1, buf215, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg391_1
        # Topologically Sorted Source Nodes: [x_1040, x_1041, x_1042], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf216 = extern_kernels.convolution(buf214, buf215, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf214
        buf217 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [x_1043, x_1044, x_1045], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf217, buf216, arg392_1, arg393_1, arg394_1, arg395_1, 225792, grid=grid(225792), stream=stream0)
        del arg392_1
        del arg393_1
        del arg394_1
        del arg395_1
        del buf216
        buf218 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [x_1046], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg396_1, buf218, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg396_1
        # Topologically Sorted Source Nodes: [x_1046], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf217, buf218, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf220 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_1047, x_1048], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf220, arg397_1, arg398_1, arg399_1, arg400_1, 225792, grid=grid(225792), stream=stream0)
        del arg397_1
        del arg398_1
        del arg399_1
        del arg400_1
        buf221 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [x_1047, x_1048, x_1049], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg401_1, buf221, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg401_1
        # Topologically Sorted Source Nodes: [x_1047, x_1048, x_1049], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf222 = extern_kernels.convolution(buf220, buf221, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf220
        buf223 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_1050, x_1051, x_1052], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf223, buf222, arg402_1, arg403_1, arg404_1, arg405_1, 225792, grid=grid(225792), stream=stream0)
        del arg402_1
        del arg403_1
        del arg404_1
        del arg405_1
        del buf222
        buf224 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [x_1053], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg406_1, buf224, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg406_1
        # Topologically Sorted Source Nodes: [x_1053], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf223, buf224, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf226 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [x_1054, x_1055], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf226, arg407_1, arg408_1, arg409_1, arg410_1, 225792, grid=grid(225792), stream=stream0)
        del arg407_1
        del arg408_1
        del arg409_1
        del arg410_1
        buf227 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [x_1054, x_1055, x_1056], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg411_1, buf227, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg411_1
        # Topologically Sorted Source Nodes: [x_1054, x_1055, x_1056], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf228 = extern_kernels.convolution(buf226, buf227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf226
        buf292 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [input_278], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(arg466_1, buf292, 648, 9, grid=grid(648, 9), stream=stream0)
        del arg466_1
        # Topologically Sorted Source Nodes: [input_278], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf200, buf292, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf232 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [input_265], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg321_1, buf232, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg321_1
        # Topologically Sorted Source Nodes: [input_265], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf117, buf232, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (8, 18, 28, 28), (14112, 1, 504, 18))
        buf234 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [input_266, input_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf234, arg322_1, arg323_1, arg324_1, arg325_1, 112896, grid=grid(112896), stream=stream0)
        del arg322_1
        del arg323_1
        del arg324_1
        del arg325_1
        buf235 = reinterpret_tensor(buf227, (72, 18, 3, 3), (162, 1, 54, 18), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [input_266, input_267, input_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg326_1, buf235, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg326_1
        # Topologically Sorted Source Nodes: [input_266, input_267, input_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf236 = extern_kernels.convolution(buf234, buf235, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf234
        buf237 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [input_270], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg331_1, buf237, 2592, 9, grid=grid(2592, 9), stream=stream0)
        del arg331_1
        # Topologically Sorted Source Nodes: [input_270], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf144, buf237, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf239 = buf236; del buf236  # reuse
        buf240 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [input_269, input_271, y_71, y_72, shortcut_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29.run(buf239, buf240, arg327_1, arg328_1, arg329_1, arg330_1, buf238, arg332_1, arg333_1, arg334_1, arg335_1, 112896, grid=grid(112896), stream=stream0)
        del arg327_1
        del arg328_1
        del arg329_1
        del arg330_1
        del arg332_1
        del arg333_1
        del arg334_1
        del arg335_1
        del buf238
        del buf239
        buf241 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_1060], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg416_1, buf241, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg416_1
        # Topologically Sorted Source Nodes: [x_1060], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf240, buf241, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf243 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [x_1061, x_1062], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf243, arg417_1, arg418_1, arg419_1, arg420_1, 112896, grid=grid(112896), stream=stream0)
        del arg417_1
        del arg418_1
        del arg419_1
        del arg420_1
        buf244 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [x_1061, x_1062, x_1063], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg421_1, buf244, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg421_1
        # Topologically Sorted Source Nodes: [x_1061, x_1062, x_1063], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf245 = extern_kernels.convolution(buf243, buf244, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf243
        buf246 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [x_1064, x_1065, x_1066], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf246, buf245, arg422_1, arg423_1, arg424_1, arg425_1, 112896, grid=grid(112896), stream=stream0)
        del arg422_1
        del arg423_1
        del arg424_1
        del arg425_1
        del buf245
        buf247 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [x_1067], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg426_1, buf247, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg426_1
        # Topologically Sorted Source Nodes: [x_1067], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf246, buf247, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf249 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [x_1068, x_1069], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf249, arg427_1, arg428_1, arg429_1, arg430_1, 112896, grid=grid(112896), stream=stream0)
        del arg427_1
        del arg428_1
        del arg429_1
        del arg430_1
        buf250 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [x_1068, x_1069, x_1070], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg431_1, buf250, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg431_1
        # Topologically Sorted Source Nodes: [x_1068, x_1069, x_1070], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf251 = extern_kernels.convolution(buf249, buf250, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf249
        buf252 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [x_1071, x_1072, x_1073], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf252, buf251, arg432_1, arg433_1, arg434_1, arg435_1, 112896, grid=grid(112896), stream=stream0)
        del arg432_1
        del arg433_1
        del arg434_1
        del arg435_1
        del buf251
        buf253 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [x_1074], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg436_1, buf253, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg436_1
        # Topologically Sorted Source Nodes: [x_1074], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf252, buf253, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf255 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [x_1075, x_1076], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf255, arg437_1, arg438_1, arg439_1, arg440_1, 112896, grid=grid(112896), stream=stream0)
        del arg437_1
        del arg438_1
        del arg439_1
        del arg440_1
        buf256 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [x_1075, x_1076, x_1077], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg441_1, buf256, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg441_1
        # Topologically Sorted Source Nodes: [x_1075, x_1076, x_1077], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf257 = extern_kernels.convolution(buf255, buf256, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf255
        buf258 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [x_1078, x_1079, x_1080], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf258, buf257, arg442_1, arg443_1, arg444_1, arg445_1, 112896, grid=grid(112896), stream=stream0)
        del arg442_1
        del arg443_1
        del arg444_1
        del arg445_1
        del buf257
        buf259 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [x_1081], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg446_1, buf259, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg446_1
        # Topologically Sorted Source Nodes: [x_1081], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf258, buf259, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf261 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [x_1082, x_1083], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf261, arg447_1, arg448_1, arg449_1, arg450_1, 112896, grid=grid(112896), stream=stream0)
        del arg447_1
        del arg448_1
        del arg449_1
        del arg450_1
        buf262 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [x_1082, x_1083, x_1084], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg451_1, buf262, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg451_1
        # Topologically Sorted Source Nodes: [x_1082, x_1083, x_1084], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf263 = extern_kernels.convolution(buf261, buf262, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf261
        buf264 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [x_1085, x_1086, x_1087], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf264, buf263, arg452_1, arg453_1, arg454_1, arg455_1, 112896, grid=grid(112896), stream=stream0)
        del arg452_1
        del arg453_1
        del arg454_1
        del arg455_1
        del buf263
        # Topologically Sorted Source Nodes: [input_280], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf264, arg471_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (8, 36, 14, 14), (7056, 1, 504, 36))
        del arg471_1
        buf295 = reinterpret_tensor(buf144, (8, 36, 28, 28), (28224, 784, 28, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [input_281, input_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_24.run(buf294, arg472_1, arg473_1, arg474_1, arg475_1, buf295, 225792, grid=grid(225792), stream=stream0)
        del arg472_1
        del arg473_1
        del arg474_1
        del arg475_1
        del buf294
        buf229 = buf223; del buf223  # reuse
        buf296 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [x_1057, x_1058, x_1059, input_279, y_75, y_76, shortcut_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf229, buf296, buf228, arg412_1, arg413_1, arg414_1, arg415_1, arg467_1, arg468_1, arg469_1, arg470_1, buf295, 6272, 36, grid=grid(6272, 36), stream=stream0)
        del arg412_1
        del arg413_1
        del arg414_1
        del arg415_1
        del arg467_1
        del arg468_1
        del arg469_1
        del arg470_1
        del buf228
        del buf295
        # Topologically Sorted Source Nodes: [input_272], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, arg456_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (8, 18, 28, 28), (14112, 1, 504, 18))
        del arg456_1
        # Topologically Sorted Source Nodes: [input_275], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, arg461_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (8, 18, 14, 14), (3528, 1, 252, 18))
        del arg461_1
        buf267 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [input_273, input_274, y_73, input_276, input_277, y_74, shortcut_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26.run(buf230, arg457_1, arg458_1, arg459_1, arg460_1, buf265, arg462_1, arg463_1, arg464_1, arg465_1, buf200, buf267, 144, 3136, grid=grid(144, 3136), stream=stream0)
        del arg457_1
        del arg458_1
        del arg459_1
        del arg460_1
        del arg462_1
        del arg463_1
        del arg464_1
        del arg465_1
        del buf230
        del buf265
        buf268 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_1088], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg491_1, buf268, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg491_1
        # Topologically Sorted Source Nodes: [x_1088], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf267, buf268, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf270 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [x_1089, x_1090], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf270, arg492_1, arg493_1, arg494_1, arg495_1, 451584, grid=grid(451584), stream=stream0)
        del arg492_1
        del arg493_1
        del arg494_1
        del arg495_1
        buf271 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [x_1089, x_1090, x_1091], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg496_1, buf271, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg496_1
        # Topologically Sorted Source Nodes: [x_1089, x_1090, x_1091], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf272 = extern_kernels.convolution(buf270, buf271, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf270
        buf273 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [x_1092, x_1093, x_1094], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf273, buf272, arg497_1, arg498_1, arg499_1, arg500_1, 451584, grid=grid(451584), stream=stream0)
        del arg497_1
        del arg498_1
        del arg499_1
        del arg500_1
        del buf272
        buf274 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [x_1095], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg501_1, buf274, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg501_1
        # Topologically Sorted Source Nodes: [x_1095], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf273, buf274, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf276 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [x_1096, x_1097], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf276, arg502_1, arg503_1, arg504_1, arg505_1, 451584, grid=grid(451584), stream=stream0)
        del arg502_1
        del arg503_1
        del arg504_1
        del arg505_1
        buf277 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [x_1096, x_1097, x_1098], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg506_1, buf277, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg506_1
        # Topologically Sorted Source Nodes: [x_1096, x_1097, x_1098], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf278 = extern_kernels.convolution(buf276, buf277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf276
        buf279 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [x_1099, x_1100, x_1101], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf279, buf278, arg507_1, arg508_1, arg509_1, arg510_1, 451584, grid=grid(451584), stream=stream0)
        del arg507_1
        del arg508_1
        del arg509_1
        del arg510_1
        del buf278
        buf280 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [x_1102], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg511_1, buf280, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg511_1
        # Topologically Sorted Source Nodes: [x_1102], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf279, buf280, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf282 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [x_1103, x_1104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf282, arg512_1, arg513_1, arg514_1, arg515_1, 451584, grid=grid(451584), stream=stream0)
        del arg512_1
        del arg513_1
        del arg514_1
        del arg515_1
        buf283 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [x_1103, x_1104, x_1105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg516_1, buf283, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg516_1
        # Topologically Sorted Source Nodes: [x_1103, x_1104, x_1105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf284 = extern_kernels.convolution(buf282, buf283, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf282
        buf285 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [x_1106, x_1107, x_1108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf285, buf284, arg517_1, arg518_1, arg519_1, arg520_1, 451584, grid=grid(451584), stream=stream0)
        del arg517_1
        del arg518_1
        del arg519_1
        del arg520_1
        del buf284
        buf286 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [x_1109], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg521_1, buf286, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg521_1
        # Topologically Sorted Source Nodes: [x_1109], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf285, buf286, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf288 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [x_1110, x_1111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf288, arg522_1, arg523_1, arg524_1, arg525_1, 451584, grid=grid(451584), stream=stream0)
        del arg522_1
        del arg523_1
        del arg524_1
        del arg525_1
        buf289 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [x_1110, x_1111, x_1112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg526_1, buf289, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg526_1
        # Topologically Sorted Source Nodes: [x_1110, x_1111, x_1112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf290 = extern_kernels.convolution(buf288, buf289, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf288
        buf291 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [x_1113, x_1114, x_1115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf291, buf290, arg527_1, arg528_1, arg529_1, arg530_1, 451584, grid=grid(451584), stream=stream0)
        del arg527_1
        del arg528_1
        del arg529_1
        del arg530_1
        del buf290
        buf297 = reinterpret_tensor(buf235, (36, 36, 3, 3), (324, 1, 108, 36), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [x_1116], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg531_1, buf297, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg531_1
        # Topologically Sorted Source Nodes: [x_1116], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf296, buf297, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf299 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [x_1117, x_1118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf299, arg532_1, arg533_1, arg534_1, arg535_1, 225792, grid=grid(225792), stream=stream0)
        del arg532_1
        del arg533_1
        del arg534_1
        del arg535_1
        buf300 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [x_1117, x_1118, x_1119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg536_1, buf300, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg536_1
        # Topologically Sorted Source Nodes: [x_1117, x_1118, x_1119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf301 = extern_kernels.convolution(buf299, buf300, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf299
        buf302 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [x_1120, x_1121, x_1122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf302, buf301, arg537_1, arg538_1, arg539_1, arg540_1, 225792, grid=grid(225792), stream=stream0)
        del arg537_1
        del arg538_1
        del arg539_1
        del arg540_1
        del buf301
        buf303 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [x_1123], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg541_1, buf303, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg541_1
        # Topologically Sorted Source Nodes: [x_1123], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf302, buf303, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf305 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [x_1124, x_1125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf305, arg542_1, arg543_1, arg544_1, arg545_1, 225792, grid=grid(225792), stream=stream0)
        del arg542_1
        del arg543_1
        del arg544_1
        del arg545_1
        buf306 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [x_1124, x_1125, x_1126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg546_1, buf306, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg546_1
        # Topologically Sorted Source Nodes: [x_1124, x_1125, x_1126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf307 = extern_kernels.convolution(buf305, buf306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf305
        buf308 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [x_1127, x_1128, x_1129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf308, buf307, arg547_1, arg548_1, arg549_1, arg550_1, 225792, grid=grid(225792), stream=stream0)
        del arg547_1
        del arg548_1
        del arg549_1
        del arg550_1
        del buf307
        buf309 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [x_1130], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg551_1, buf309, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg551_1
        # Topologically Sorted Source Nodes: [x_1130], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf308, buf309, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf311 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [x_1131, x_1132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf311, arg552_1, arg553_1, arg554_1, arg555_1, 225792, grid=grid(225792), stream=stream0)
        del arg552_1
        del arg553_1
        del arg554_1
        del arg555_1
        buf312 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [x_1131, x_1132, x_1133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg556_1, buf312, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg556_1
        # Topologically Sorted Source Nodes: [x_1131, x_1132, x_1133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf313 = extern_kernels.convolution(buf311, buf312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf311
        buf314 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [x_1134, x_1135, x_1136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf314, buf313, arg557_1, arg558_1, arg559_1, arg560_1, 225792, grid=grid(225792), stream=stream0)
        del arg557_1
        del arg558_1
        del arg559_1
        del arg560_1
        del buf313
        buf315 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [x_1137], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg561_1, buf315, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg561_1
        # Topologically Sorted Source Nodes: [x_1137], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf314, buf315, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf317 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [x_1138, x_1139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf317, arg562_1, arg563_1, arg564_1, arg565_1, 225792, grid=grid(225792), stream=stream0)
        del arg562_1
        del arg563_1
        del arg564_1
        del arg565_1
        buf318 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [x_1138, x_1139, x_1140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg566_1, buf318, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg566_1
        # Topologically Sorted Source Nodes: [x_1138, x_1139, x_1140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf319 = extern_kernels.convolution(buf317, buf318, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf317
        buf383 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [input_296], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(arg621_1, buf383, 648, 9, grid=grid(648, 9), stream=stream0)
        del arg621_1
        # Topologically Sorted Source Nodes: [input_296], Original ATen: [aten.convolution]
        buf384 = extern_kernels.convolution(buf291, buf383, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf384, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf323 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [input_283], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg476_1, buf323, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg476_1
        # Topologically Sorted Source Nodes: [input_283], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf200, buf323, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (8, 18, 28, 28), (14112, 1, 504, 18))
        buf325 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [input_284, input_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf325, arg477_1, arg478_1, arg479_1, arg480_1, 112896, grid=grid(112896), stream=stream0)
        del arg477_1
        del arg478_1
        del arg479_1
        del arg480_1
        buf326 = reinterpret_tensor(buf318, (72, 18, 3, 3), (162, 1, 54, 18), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [input_284, input_285, input_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg481_1, buf326, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg481_1
        # Topologically Sorted Source Nodes: [input_284, input_285, input_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf327 = extern_kernels.convolution(buf325, buf326, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf325
        buf328 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [input_288], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg486_1, buf328, 2592, 9, grid=grid(2592, 9), stream=stream0)
        del arg486_1
        # Topologically Sorted Source Nodes: [input_288], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf229, buf328, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf330 = buf327; del buf327  # reuse
        buf331 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [input_287, input_289, y_77, y_78, shortcut_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29.run(buf330, buf331, arg482_1, arg483_1, arg484_1, arg485_1, buf329, arg487_1, arg488_1, arg489_1, arg490_1, 112896, grid=grid(112896), stream=stream0)
        del arg482_1
        del arg483_1
        del arg484_1
        del arg485_1
        del arg487_1
        del arg488_1
        del arg489_1
        del arg490_1
        del buf329
        del buf330
        buf332 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [x_1144], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg571_1, buf332, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg571_1
        # Topologically Sorted Source Nodes: [x_1144], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf331, buf332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf334 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [x_1145, x_1146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf334, arg572_1, arg573_1, arg574_1, arg575_1, 112896, grid=grid(112896), stream=stream0)
        del arg572_1
        del arg573_1
        del arg574_1
        del arg575_1
        buf335 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [x_1145, x_1146, x_1147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg576_1, buf335, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg576_1
        # Topologically Sorted Source Nodes: [x_1145, x_1146, x_1147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf336 = extern_kernels.convolution(buf334, buf335, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf334
        buf337 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [x_1148, x_1149, x_1150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf337, buf336, arg577_1, arg578_1, arg579_1, arg580_1, 112896, grid=grid(112896), stream=stream0)
        del arg577_1
        del arg578_1
        del arg579_1
        del arg580_1
        del buf336
        buf338 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [x_1151], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg581_1, buf338, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg581_1
        # Topologically Sorted Source Nodes: [x_1151], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf337, buf338, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf339, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf340 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [x_1152, x_1153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf340, arg582_1, arg583_1, arg584_1, arg585_1, 112896, grid=grid(112896), stream=stream0)
        del arg582_1
        del arg583_1
        del arg584_1
        del arg585_1
        buf341 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [x_1152, x_1153, x_1154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg586_1, buf341, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg586_1
        # Topologically Sorted Source Nodes: [x_1152, x_1153, x_1154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf342 = extern_kernels.convolution(buf340, buf341, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf340
        buf343 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [x_1155, x_1156, x_1157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf343, buf342, arg587_1, arg588_1, arg589_1, arg590_1, 112896, grid=grid(112896), stream=stream0)
        del arg587_1
        del arg588_1
        del arg589_1
        del arg590_1
        del buf342
        buf344 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [x_1158], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg591_1, buf344, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg591_1
        # Topologically Sorted Source Nodes: [x_1158], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf343, buf344, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf346 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [x_1159, x_1160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf346, arg592_1, arg593_1, arg594_1, arg595_1, 112896, grid=grid(112896), stream=stream0)
        del arg592_1
        del arg593_1
        del arg594_1
        del arg595_1
        buf347 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [x_1159, x_1160, x_1161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg596_1, buf347, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg596_1
        # Topologically Sorted Source Nodes: [x_1159, x_1160, x_1161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf348 = extern_kernels.convolution(buf346, buf347, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf346
        buf349 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [x_1162, x_1163, x_1164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf349, buf348, arg597_1, arg598_1, arg599_1, arg600_1, 112896, grid=grid(112896), stream=stream0)
        del arg597_1
        del arg598_1
        del arg599_1
        del arg600_1
        del buf348
        buf350 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [x_1165], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg601_1, buf350, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg601_1
        # Topologically Sorted Source Nodes: [x_1165], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf349, buf350, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf352 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [x_1166, x_1167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf352, arg602_1, arg603_1, arg604_1, arg605_1, 112896, grid=grid(112896), stream=stream0)
        del arg602_1
        del arg603_1
        del arg604_1
        del arg605_1
        buf353 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [x_1166, x_1167, x_1168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg606_1, buf353, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg606_1
        # Topologically Sorted Source Nodes: [x_1166, x_1167, x_1168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf354 = extern_kernels.convolution(buf352, buf353, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf352
        buf355 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [x_1169, x_1170, x_1171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf355, buf354, arg607_1, arg608_1, arg609_1, arg610_1, 112896, grid=grid(112896), stream=stream0)
        del arg607_1
        del arg608_1
        del arg609_1
        del arg610_1
        del buf354
        # Topologically Sorted Source Nodes: [input_298], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf355, arg626_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (8, 36, 14, 14), (7056, 1, 504, 36))
        del arg626_1
        buf386 = reinterpret_tensor(buf229, (8, 36, 28, 28), (28224, 784, 28, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [input_299, input_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_24.run(buf385, arg627_1, arg628_1, arg629_1, arg630_1, buf386, 225792, grid=grid(225792), stream=stream0)
        del arg627_1
        del arg628_1
        del arg629_1
        del arg630_1
        del buf385
        buf320 = buf314; del buf314  # reuse
        buf387 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [x_1141, x_1142, x_1143, input_297, y_81, y_82, shortcut_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf320, buf387, buf319, arg567_1, arg568_1, arg569_1, arg570_1, arg622_1, arg623_1, arg624_1, arg625_1, buf386, 6272, 36, grid=grid(6272, 36), stream=stream0)
        del arg567_1
        del arg568_1
        del arg569_1
        del arg570_1
        del arg622_1
        del arg623_1
        del arg624_1
        del arg625_1
        del buf319
        del buf386
        # Topologically Sorted Source Nodes: [input_290], Original ATen: [aten.convolution]
        buf321 = extern_kernels.convolution(buf320, arg611_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf321, (8, 18, 28, 28), (14112, 1, 504, 18))
        del arg611_1
        # Topologically Sorted Source Nodes: [input_293], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, arg616_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (8, 18, 14, 14), (3528, 1, 252, 18))
        del arg616_1
        buf358 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [input_291, input_292, y_79, input_294, input_295, y_80, shortcut_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26.run(buf321, arg612_1, arg613_1, arg614_1, arg615_1, buf356, arg617_1, arg618_1, arg619_1, arg620_1, buf291, buf358, 144, 3136, grid=grid(144, 3136), stream=stream0)
        del arg612_1
        del arg613_1
        del arg614_1
        del arg615_1
        del arg617_1
        del arg618_1
        del arg619_1
        del arg620_1
        del buf321
        del buf356
        buf359 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [x_1172], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg646_1, buf359, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg646_1
        # Topologically Sorted Source Nodes: [x_1172], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(buf358, buf359, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf361 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [x_1173, x_1174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf361, arg647_1, arg648_1, arg649_1, arg650_1, 451584, grid=grid(451584), stream=stream0)
        del arg647_1
        del arg648_1
        del arg649_1
        del arg650_1
        buf362 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [x_1173, x_1174, x_1175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg651_1, buf362, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg651_1
        # Topologically Sorted Source Nodes: [x_1173, x_1174, x_1175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf363 = extern_kernels.convolution(buf361, buf362, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf361
        buf364 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [x_1176, x_1177, x_1178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf364, buf363, arg652_1, arg653_1, arg654_1, arg655_1, 451584, grid=grid(451584), stream=stream0)
        del arg652_1
        del arg653_1
        del arg654_1
        del arg655_1
        del buf363
        buf365 = buf362; del buf362  # reuse
        # Topologically Sorted Source Nodes: [x_1179], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg656_1, buf365, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg656_1
        # Topologically Sorted Source Nodes: [x_1179], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf364, buf365, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf367 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [x_1180, x_1181], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf367, arg657_1, arg658_1, arg659_1, arg660_1, 451584, grid=grid(451584), stream=stream0)
        del arg657_1
        del arg658_1
        del arg659_1
        del arg660_1
        buf368 = buf365; del buf365  # reuse
        # Topologically Sorted Source Nodes: [x_1180, x_1181, x_1182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg661_1, buf368, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg661_1
        # Topologically Sorted Source Nodes: [x_1180, x_1181, x_1182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf369 = extern_kernels.convolution(buf367, buf368, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf367
        buf370 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [x_1183, x_1184, x_1185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf370, buf369, arg662_1, arg663_1, arg664_1, arg665_1, 451584, grid=grid(451584), stream=stream0)
        del arg662_1
        del arg663_1
        del arg664_1
        del arg665_1
        del buf369
        buf371 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [x_1186], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg666_1, buf371, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg666_1
        # Topologically Sorted Source Nodes: [x_1186], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf370, buf371, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf373 = buf372; del buf372  # reuse
        # Topologically Sorted Source Nodes: [x_1187, x_1188], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf373, arg667_1, arg668_1, arg669_1, arg670_1, 451584, grid=grid(451584), stream=stream0)
        del arg667_1
        del arg668_1
        del arg669_1
        del arg670_1
        buf374 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [x_1187, x_1188, x_1189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg671_1, buf374, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg671_1
        # Topologically Sorted Source Nodes: [x_1187, x_1188, x_1189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf375 = extern_kernels.convolution(buf373, buf374, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf373
        buf376 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [x_1190, x_1191, x_1192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf376, buf375, arg672_1, arg673_1, arg674_1, arg675_1, 451584, grid=grid(451584), stream=stream0)
        del arg672_1
        del arg673_1
        del arg674_1
        del arg675_1
        del buf375
        buf377 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [x_1193], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg676_1, buf377, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg676_1
        # Topologically Sorted Source Nodes: [x_1193], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf376, buf377, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf378, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf379 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [x_1194, x_1195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf379, arg677_1, arg678_1, arg679_1, arg680_1, 451584, grid=grid(451584), stream=stream0)
        del arg677_1
        del arg678_1
        del arg679_1
        del arg680_1
        buf380 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [x_1194, x_1195, x_1196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg681_1, buf380, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg681_1
        # Topologically Sorted Source Nodes: [x_1194, x_1195, x_1196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf381 = extern_kernels.convolution(buf379, buf380, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf379
        buf382 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [x_1197, x_1198, x_1199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf382, buf381, arg682_1, arg683_1, arg684_1, arg685_1, 451584, grid=grid(451584), stream=stream0)
        del arg682_1
        del arg683_1
        del arg684_1
        del arg685_1
        del buf381
        buf388 = reinterpret_tensor(buf326, (36, 36, 3, 3), (324, 1, 108, 36), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [x_1200], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg686_1, buf388, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg686_1
        # Topologically Sorted Source Nodes: [x_1200], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf387, buf388, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf390 = buf389; del buf389  # reuse
        # Topologically Sorted Source Nodes: [x_1201, x_1202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf390, arg687_1, arg688_1, arg689_1, arg690_1, 225792, grid=grid(225792), stream=stream0)
        del arg687_1
        del arg688_1
        del arg689_1
        del arg690_1
        buf391 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [x_1201, x_1202, x_1203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg691_1, buf391, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg691_1
        # Topologically Sorted Source Nodes: [x_1201, x_1202, x_1203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf392 = extern_kernels.convolution(buf390, buf391, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf390
        buf393 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [x_1204, x_1205, x_1206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf393, buf392, arg692_1, arg693_1, arg694_1, arg695_1, 225792, grid=grid(225792), stream=stream0)
        del arg692_1
        del arg693_1
        del arg694_1
        del arg695_1
        del buf392
        buf394 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [x_1207], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg696_1, buf394, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg696_1
        # Topologically Sorted Source Nodes: [x_1207], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf393, buf394, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf396 = buf395; del buf395  # reuse
        # Topologically Sorted Source Nodes: [x_1208, x_1209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf396, arg697_1, arg698_1, arg699_1, arg700_1, 225792, grid=grid(225792), stream=stream0)
        del arg697_1
        del arg698_1
        del arg699_1
        del arg700_1
        buf397 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [x_1208, x_1209, x_1210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg701_1, buf397, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg701_1
        # Topologically Sorted Source Nodes: [x_1208, x_1209, x_1210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf398 = extern_kernels.convolution(buf396, buf397, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf398, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf396
        buf399 = buf393; del buf393  # reuse
        # Topologically Sorted Source Nodes: [x_1211, x_1212, x_1213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf399, buf398, arg702_1, arg703_1, arg704_1, arg705_1, 225792, grid=grid(225792), stream=stream0)
        del arg702_1
        del arg703_1
        del arg704_1
        del arg705_1
        del buf398
        buf400 = buf397; del buf397  # reuse
        # Topologically Sorted Source Nodes: [x_1214], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg706_1, buf400, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg706_1
        # Topologically Sorted Source Nodes: [x_1214], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf399, buf400, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf402 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [x_1215, x_1216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf402, arg707_1, arg708_1, arg709_1, arg710_1, 225792, grid=grid(225792), stream=stream0)
        del arg707_1
        del arg708_1
        del arg709_1
        del arg710_1
        buf403 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [x_1215, x_1216, x_1217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg711_1, buf403, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg711_1
        # Topologically Sorted Source Nodes: [x_1215, x_1216, x_1217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf404 = extern_kernels.convolution(buf402, buf403, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf402
        buf405 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [x_1218, x_1219, x_1220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf405, buf404, arg712_1, arg713_1, arg714_1, arg715_1, 225792, grid=grid(225792), stream=stream0)
        del arg712_1
        del arg713_1
        del arg714_1
        del arg715_1
        del buf404
        buf406 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [x_1221], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg716_1, buf406, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg716_1
        # Topologically Sorted Source Nodes: [x_1221], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf405, buf406, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf408 = buf407; del buf407  # reuse
        # Topologically Sorted Source Nodes: [x_1222, x_1223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf408, arg717_1, arg718_1, arg719_1, arg720_1, 225792, grid=grid(225792), stream=stream0)
        del arg717_1
        del arg718_1
        del arg719_1
        del arg720_1
        buf409 = buf406; del buf406  # reuse
        # Topologically Sorted Source Nodes: [x_1222, x_1223, x_1224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg721_1, buf409, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg721_1
        # Topologically Sorted Source Nodes: [x_1222, x_1223, x_1224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf410 = extern_kernels.convolution(buf408, buf409, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf408
        buf474 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [input_314], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(arg776_1, buf474, 648, 9, grid=grid(648, 9), stream=stream0)
        del arg776_1
        # Topologically Sorted Source Nodes: [input_314], Original ATen: [aten.convolution]
        buf475 = extern_kernels.convolution(buf382, buf474, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf414 = buf380; del buf380  # reuse
        # Topologically Sorted Source Nodes: [input_301], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg631_1, buf414, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg631_1
        # Topologically Sorted Source Nodes: [input_301], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(buf291, buf414, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (8, 18, 28, 28), (14112, 1, 504, 18))
        buf416 = buf415; del buf415  # reuse
        # Topologically Sorted Source Nodes: [input_302, input_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf416, arg632_1, arg633_1, arg634_1, arg635_1, 112896, grid=grid(112896), stream=stream0)
        del arg632_1
        del arg633_1
        del arg634_1
        del arg635_1
        buf417 = reinterpret_tensor(buf409, (72, 18, 3, 3), (162, 1, 54, 18), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [input_302, input_303, input_304], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg636_1, buf417, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg636_1
        # Topologically Sorted Source Nodes: [input_302, input_303, input_304], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf418 = extern_kernels.convolution(buf416, buf417, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf416
        buf419 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [input_306], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg641_1, buf419, 2592, 9, grid=grid(2592, 9), stream=stream0)
        del arg641_1
        # Topologically Sorted Source Nodes: [input_306], Original ATen: [aten.convolution]
        buf420 = extern_kernels.convolution(buf320, buf419, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf420, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf421 = buf418; del buf418  # reuse
        buf422 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [input_305, input_307, y_83, y_84, shortcut_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29.run(buf421, buf422, arg637_1, arg638_1, arg639_1, arg640_1, buf420, arg642_1, arg643_1, arg644_1, arg645_1, 112896, grid=grid(112896), stream=stream0)
        del arg637_1
        del arg638_1
        del arg639_1
        del arg640_1
        del arg642_1
        del arg643_1
        del arg644_1
        del arg645_1
        del buf420
        del buf421
        buf423 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [x_1228], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg726_1, buf423, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg726_1
        # Topologically Sorted Source Nodes: [x_1228], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf422, buf423, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf425 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [x_1229, x_1230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf425, arg727_1, arg728_1, arg729_1, arg730_1, 112896, grid=grid(112896), stream=stream0)
        del arg727_1
        del arg728_1
        del arg729_1
        del arg730_1
        buf426 = buf423; del buf423  # reuse
        # Topologically Sorted Source Nodes: [x_1229, x_1230, x_1231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg731_1, buf426, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg731_1
        # Topologically Sorted Source Nodes: [x_1229, x_1230, x_1231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf427 = extern_kernels.convolution(buf425, buf426, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf425
        buf428 = buf422; del buf422  # reuse
        # Topologically Sorted Source Nodes: [x_1232, x_1233, x_1234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf428, buf427, arg732_1, arg733_1, arg734_1, arg735_1, 112896, grid=grid(112896), stream=stream0)
        del arg732_1
        del arg733_1
        del arg734_1
        del arg735_1
        del buf427
        buf429 = buf426; del buf426  # reuse
        # Topologically Sorted Source Nodes: [x_1235], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg736_1, buf429, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg736_1
        # Topologically Sorted Source Nodes: [x_1235], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf428, buf429, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf431 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [x_1236, x_1237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf431, arg737_1, arg738_1, arg739_1, arg740_1, 112896, grid=grid(112896), stream=stream0)
        del arg737_1
        del arg738_1
        del arg739_1
        del arg740_1
        buf432 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [x_1236, x_1237, x_1238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg741_1, buf432, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg741_1
        # Topologically Sorted Source Nodes: [x_1236, x_1237, x_1238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf433 = extern_kernels.convolution(buf431, buf432, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf433, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf431
        buf434 = buf428; del buf428  # reuse
        # Topologically Sorted Source Nodes: [x_1239, x_1240, x_1241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf434, buf433, arg742_1, arg743_1, arg744_1, arg745_1, 112896, grid=grid(112896), stream=stream0)
        del arg742_1
        del arg743_1
        del arg744_1
        del arg745_1
        del buf433
        buf435 = buf432; del buf432  # reuse
        # Topologically Sorted Source Nodes: [x_1242], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg746_1, buf435, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg746_1
        # Topologically Sorted Source Nodes: [x_1242], Original ATen: [aten.convolution]
        buf436 = extern_kernels.convolution(buf434, buf435, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf436, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf437 = buf436; del buf436  # reuse
        # Topologically Sorted Source Nodes: [x_1243, x_1244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf437, arg747_1, arg748_1, arg749_1, arg750_1, 112896, grid=grid(112896), stream=stream0)
        del arg747_1
        del arg748_1
        del arg749_1
        del arg750_1
        buf438 = buf435; del buf435  # reuse
        # Topologically Sorted Source Nodes: [x_1243, x_1244, x_1245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg751_1, buf438, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg751_1
        # Topologically Sorted Source Nodes: [x_1243, x_1244, x_1245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf439 = extern_kernels.convolution(buf437, buf438, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf437
        buf440 = buf434; del buf434  # reuse
        # Topologically Sorted Source Nodes: [x_1246, x_1247, x_1248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf440, buf439, arg752_1, arg753_1, arg754_1, arg755_1, 112896, grid=grid(112896), stream=stream0)
        del arg752_1
        del arg753_1
        del arg754_1
        del arg755_1
        del buf439
        buf441 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [x_1249], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg756_1, buf441, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg756_1
        # Topologically Sorted Source Nodes: [x_1249], Original ATen: [aten.convolution]
        buf442 = extern_kernels.convolution(buf440, buf441, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf443 = buf442; del buf442  # reuse
        # Topologically Sorted Source Nodes: [x_1250, x_1251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf443, arg757_1, arg758_1, arg759_1, arg760_1, 112896, grid=grid(112896), stream=stream0)
        del arg757_1
        del arg758_1
        del arg759_1
        del arg760_1
        buf444 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [x_1250, x_1251, x_1252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg761_1, buf444, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg761_1
        # Topologically Sorted Source Nodes: [x_1250, x_1251, x_1252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf445 = extern_kernels.convolution(buf443, buf444, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf443
        buf446 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [x_1253, x_1254, x_1255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf446, buf445, arg762_1, arg763_1, arg764_1, arg765_1, 112896, grid=grid(112896), stream=stream0)
        del arg762_1
        del arg763_1
        del arg764_1
        del arg765_1
        del buf445
        # Topologically Sorted Source Nodes: [input_316], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(buf446, arg781_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (8, 36, 14, 14), (7056, 1, 504, 36))
        del arg781_1
        buf477 = reinterpret_tensor(buf320, (8, 36, 28, 28), (28224, 784, 28, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [input_317, input_318], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_24.run(buf476, arg782_1, arg783_1, arg784_1, arg785_1, buf477, 225792, grid=grid(225792), stream=stream0)
        del arg782_1
        del arg783_1
        del arg784_1
        del arg785_1
        del buf476
        buf411 = buf405; del buf405  # reuse
        buf478 = buf475; del buf475  # reuse
        # Topologically Sorted Source Nodes: [x_1225, x_1226, x_1227, input_315, y_87, y_88, shortcut_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf411, buf478, buf410, arg722_1, arg723_1, arg724_1, arg725_1, arg777_1, arg778_1, arg779_1, arg780_1, buf477, 6272, 36, grid=grid(6272, 36), stream=stream0)
        del arg722_1
        del arg723_1
        del arg724_1
        del arg725_1
        del arg777_1
        del arg778_1
        del arg779_1
        del arg780_1
        del buf410
        del buf477
        # Topologically Sorted Source Nodes: [input_308], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, arg766_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (8, 18, 28, 28), (14112, 1, 504, 18))
        del arg766_1
        # Topologically Sorted Source Nodes: [input_311], Original ATen: [aten.convolution]
        buf447 = extern_kernels.convolution(buf446, arg771_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf447, (8, 18, 14, 14), (3528, 1, 252, 18))
        del arg771_1
        buf449 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [input_309, input_310, y_85, input_312, input_313, y_86, shortcut_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26.run(buf412, arg767_1, arg768_1, arg769_1, arg770_1, buf447, arg772_1, arg773_1, arg774_1, arg775_1, buf382, buf449, 144, 3136, grid=grid(144, 3136), stream=stream0)
        del arg767_1
        del arg768_1
        del arg769_1
        del arg770_1
        del arg772_1
        del arg773_1
        del arg774_1
        del arg775_1
        del buf412
        del buf447
        buf450 = buf414; del buf414  # reuse
        # Topologically Sorted Source Nodes: [x_1256], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg806_1, buf450, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg806_1
        # Topologically Sorted Source Nodes: [x_1256], Original ATen: [aten.convolution]
        buf451 = extern_kernels.convolution(buf449, buf450, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf452 = buf451; del buf451  # reuse
        # Topologically Sorted Source Nodes: [x_1257, x_1258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf452, arg807_1, arg808_1, arg809_1, arg810_1, 451584, grid=grid(451584), stream=stream0)
        del arg807_1
        del arg808_1
        del arg809_1
        del arg810_1
        buf453 = buf450; del buf450  # reuse
        # Topologically Sorted Source Nodes: [x_1257, x_1258, x_1259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg811_1, buf453, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg811_1
        # Topologically Sorted Source Nodes: [x_1257, x_1258, x_1259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf454 = extern_kernels.convolution(buf452, buf453, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf454, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf452
        buf455 = buf449; del buf449  # reuse
        # Topologically Sorted Source Nodes: [x_1260, x_1261, x_1262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf455, buf454, arg812_1, arg813_1, arg814_1, arg815_1, 451584, grid=grid(451584), stream=stream0)
        del arg812_1
        del arg813_1
        del arg814_1
        del arg815_1
        del buf454
        buf456 = buf453; del buf453  # reuse
        # Topologically Sorted Source Nodes: [x_1263], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg816_1, buf456, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg816_1
        # Topologically Sorted Source Nodes: [x_1263], Original ATen: [aten.convolution]
        buf457 = extern_kernels.convolution(buf455, buf456, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf457, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf458 = buf457; del buf457  # reuse
        # Topologically Sorted Source Nodes: [x_1264, x_1265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf458, arg817_1, arg818_1, arg819_1, arg820_1, 451584, grid=grid(451584), stream=stream0)
        del arg817_1
        del arg818_1
        del arg819_1
        del arg820_1
        buf459 = buf456; del buf456  # reuse
        # Topologically Sorted Source Nodes: [x_1264, x_1265, x_1266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg821_1, buf459, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg821_1
        # Topologically Sorted Source Nodes: [x_1264, x_1265, x_1266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf460 = extern_kernels.convolution(buf458, buf459, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf460, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf458
        buf461 = buf455; del buf455  # reuse
        # Topologically Sorted Source Nodes: [x_1267, x_1268, x_1269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf461, buf460, arg822_1, arg823_1, arg824_1, arg825_1, 451584, grid=grid(451584), stream=stream0)
        del arg822_1
        del arg823_1
        del arg824_1
        del arg825_1
        del buf460
        buf462 = buf459; del buf459  # reuse
        # Topologically Sorted Source Nodes: [x_1270], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg826_1, buf462, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg826_1
        # Topologically Sorted Source Nodes: [x_1270], Original ATen: [aten.convolution]
        buf463 = extern_kernels.convolution(buf461, buf462, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf464 = buf463; del buf463  # reuse
        # Topologically Sorted Source Nodes: [x_1271, x_1272], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf464, arg827_1, arg828_1, arg829_1, arg830_1, 451584, grid=grid(451584), stream=stream0)
        del arg827_1
        del arg828_1
        del arg829_1
        del arg830_1
        buf465 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [x_1271, x_1272, x_1273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg831_1, buf465, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg831_1
        # Topologically Sorted Source Nodes: [x_1271, x_1272, x_1273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf466 = extern_kernels.convolution(buf464, buf465, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf466, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf464
        buf467 = buf461; del buf461  # reuse
        # Topologically Sorted Source Nodes: [x_1274, x_1275, x_1276], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf467, buf466, arg832_1, arg833_1, arg834_1, arg835_1, 451584, grid=grid(451584), stream=stream0)
        del arg832_1
        del arg833_1
        del arg834_1
        del arg835_1
        del buf466
        buf468 = buf465; del buf465  # reuse
        # Topologically Sorted Source Nodes: [x_1277], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg836_1, buf468, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg836_1
        # Topologically Sorted Source Nodes: [x_1277], Original ATen: [aten.convolution]
        buf469 = extern_kernels.convolution(buf467, buf468, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf469, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf470 = buf469; del buf469  # reuse
        # Topologically Sorted Source Nodes: [x_1278, x_1279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf470, arg837_1, arg838_1, arg839_1, arg840_1, 451584, grid=grid(451584), stream=stream0)
        del arg837_1
        del arg838_1
        del arg839_1
        del arg840_1
        buf471 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [x_1278, x_1279, x_1280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg841_1, buf471, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg841_1
        # Topologically Sorted Source Nodes: [x_1278, x_1279, x_1280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf472 = extern_kernels.convolution(buf470, buf471, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf472, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf470
        buf473 = buf467; del buf467  # reuse
        # Topologically Sorted Source Nodes: [x_1281, x_1282, x_1283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf473, buf472, arg842_1, arg843_1, arg844_1, arg845_1, 451584, grid=grid(451584), stream=stream0)
        del arg842_1
        del arg843_1
        del arg844_1
        del arg845_1
        del buf472
        buf479 = reinterpret_tensor(buf417, (36, 36, 3, 3), (324, 1, 108, 36), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [x_1284], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg846_1, buf479, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg846_1
        # Topologically Sorted Source Nodes: [x_1284], Original ATen: [aten.convolution]
        buf480 = extern_kernels.convolution(buf478, buf479, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf480, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf481 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [x_1285, x_1286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf481, arg847_1, arg848_1, arg849_1, arg850_1, 225792, grid=grid(225792), stream=stream0)
        del arg847_1
        del arg848_1
        del arg849_1
        del arg850_1
        buf482 = buf479; del buf479  # reuse
        # Topologically Sorted Source Nodes: [x_1285, x_1286, x_1287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg851_1, buf482, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg851_1
        # Topologically Sorted Source Nodes: [x_1285, x_1286, x_1287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf483 = extern_kernels.convolution(buf481, buf482, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf483, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf481
        buf484 = buf478; del buf478  # reuse
        # Topologically Sorted Source Nodes: [x_1288, x_1289, x_1290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf484, buf483, arg852_1, arg853_1, arg854_1, arg855_1, 225792, grid=grid(225792), stream=stream0)
        del arg852_1
        del arg853_1
        del arg854_1
        del arg855_1
        del buf483
        buf485 = buf482; del buf482  # reuse
        # Topologically Sorted Source Nodes: [x_1291], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg856_1, buf485, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg856_1
        # Topologically Sorted Source Nodes: [x_1291], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf484, buf485, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf487 = buf486; del buf486  # reuse
        # Topologically Sorted Source Nodes: [x_1292, x_1293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf487, arg857_1, arg858_1, arg859_1, arg860_1, 225792, grid=grid(225792), stream=stream0)
        del arg857_1
        del arg858_1
        del arg859_1
        del arg860_1
        buf488 = buf485; del buf485  # reuse
        # Topologically Sorted Source Nodes: [x_1292, x_1293, x_1294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg861_1, buf488, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg861_1
        # Topologically Sorted Source Nodes: [x_1292, x_1293, x_1294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf489 = extern_kernels.convolution(buf487, buf488, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf489, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf487
        buf490 = buf484; del buf484  # reuse
        # Topologically Sorted Source Nodes: [x_1295, x_1296, x_1297], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf490, buf489, arg862_1, arg863_1, arg864_1, arg865_1, 225792, grid=grid(225792), stream=stream0)
        del arg862_1
        del arg863_1
        del arg864_1
        del arg865_1
        del buf489
        buf491 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [x_1298], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg866_1, buf491, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg866_1
        # Topologically Sorted Source Nodes: [x_1298], Original ATen: [aten.convolution]
        buf492 = extern_kernels.convolution(buf490, buf491, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf492, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf493 = buf492; del buf492  # reuse
        # Topologically Sorted Source Nodes: [x_1299, x_1300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf493, arg867_1, arg868_1, arg869_1, arg870_1, 225792, grid=grid(225792), stream=stream0)
        del arg867_1
        del arg868_1
        del arg869_1
        del arg870_1
        buf494 = buf491; del buf491  # reuse
        # Topologically Sorted Source Nodes: [x_1299, x_1300, x_1301], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg871_1, buf494, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg871_1
        # Topologically Sorted Source Nodes: [x_1299, x_1300, x_1301], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf495 = extern_kernels.convolution(buf493, buf494, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf495, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf493
        buf496 = buf490; del buf490  # reuse
        # Topologically Sorted Source Nodes: [x_1302, x_1303, x_1304], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf496, buf495, arg872_1, arg873_1, arg874_1, arg875_1, 225792, grid=grid(225792), stream=stream0)
        del arg872_1
        del arg873_1
        del arg874_1
        del arg875_1
        del buf495
        buf497 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [x_1305], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg876_1, buf497, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg876_1
        # Topologically Sorted Source Nodes: [x_1305], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf496, buf497, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf499 = buf498; del buf498  # reuse
        # Topologically Sorted Source Nodes: [x_1306, x_1307], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf499, arg877_1, arg878_1, arg879_1, arg880_1, 225792, grid=grid(225792), stream=stream0)
        del arg877_1
        del arg878_1
        del arg879_1
        del arg880_1
        buf500 = buf497; del buf497  # reuse
        # Topologically Sorted Source Nodes: [x_1306, x_1307, x_1308], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg881_1, buf500, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg881_1
        # Topologically Sorted Source Nodes: [x_1306, x_1307, x_1308], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf501 = extern_kernels.convolution(buf499, buf500, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf501, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf594 = buf474; del buf474  # reuse
        # Topologically Sorted Source Nodes: [input_338], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(arg981_1, buf594, 648, 9, grid=grid(648, 9), stream=stream0)
        del arg981_1
        # Topologically Sorted Source Nodes: [input_338], Original ATen: [aten.convolution]
        buf595 = extern_kernels.convolution(buf473, buf594, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf595, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf505 = buf471; del buf471  # reuse
        # Topologically Sorted Source Nodes: [input_319], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg786_1, buf505, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg786_1
        # Topologically Sorted Source Nodes: [input_319], Original ATen: [aten.convolution]
        buf506 = extern_kernels.convolution(buf382, buf505, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf506, (8, 18, 28, 28), (14112, 1, 504, 18))
        buf507 = buf506; del buf506  # reuse
        # Topologically Sorted Source Nodes: [input_320, input_321], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf507, arg787_1, arg788_1, arg789_1, arg790_1, 112896, grid=grid(112896), stream=stream0)
        del arg787_1
        del arg788_1
        del arg789_1
        del arg790_1
        buf508 = reinterpret_tensor(buf500, (72, 18, 3, 3), (162, 1, 54, 18), 0); del buf500  # reuse
        # Topologically Sorted Source Nodes: [input_320, input_321, input_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg791_1, buf508, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg791_1
        # Topologically Sorted Source Nodes: [input_320, input_321, input_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf509 = extern_kernels.convolution(buf507, buf508, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf509, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf507
        buf510 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [input_324], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg796_1, buf510, 2592, 9, grid=grid(2592, 9), stream=stream0)
        del arg796_1
        # Topologically Sorted Source Nodes: [input_324], Original ATen: [aten.convolution]
        buf511 = extern_kernels.convolution(buf411, buf510, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf511, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf512 = buf509; del buf509  # reuse
        buf513 = buf446; del buf446  # reuse
        # Topologically Sorted Source Nodes: [input_323, input_325, y_89, y_90, shortcut_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29.run(buf512, buf513, arg792_1, arg793_1, arg794_1, arg795_1, buf511, arg797_1, arg798_1, arg799_1, arg800_1, 112896, grid=grid(112896), stream=stream0)
        del arg792_1
        del arg793_1
        del arg794_1
        del arg795_1
        del arg797_1
        del arg798_1
        del arg799_1
        del arg800_1
        del buf511
        del buf512
        buf514 = buf444; del buf444  # reuse
        # Topologically Sorted Source Nodes: [x_1312], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg886_1, buf514, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg886_1
        # Topologically Sorted Source Nodes: [x_1312], Original ATen: [aten.convolution]
        buf515 = extern_kernels.convolution(buf513, buf514, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf515, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf516 = buf515; del buf515  # reuse
        # Topologically Sorted Source Nodes: [x_1313, x_1314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf516, arg887_1, arg888_1, arg889_1, arg890_1, 112896, grid=grid(112896), stream=stream0)
        del arg887_1
        del arg888_1
        del arg889_1
        del arg890_1
        buf517 = buf514; del buf514  # reuse
        # Topologically Sorted Source Nodes: [x_1313, x_1314, x_1315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg891_1, buf517, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg891_1
        # Topologically Sorted Source Nodes: [x_1313, x_1314, x_1315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf518 = extern_kernels.convolution(buf516, buf517, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf518, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf516
        buf519 = buf518; del buf518  # reuse
        # Topologically Sorted Source Nodes: [x_1316, x_1317, x_1318], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30.run(buf519, arg892_1, arg893_1, arg894_1, arg895_1, buf513, 112896, grid=grid(112896), stream=stream0)
        del arg892_1
        del arg893_1
        del arg894_1
        del arg895_1
        buf520 = buf517; del buf517  # reuse
        # Topologically Sorted Source Nodes: [x_1319], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg896_1, buf520, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg896_1
        # Topologically Sorted Source Nodes: [x_1319], Original ATen: [aten.convolution]
        buf521 = extern_kernels.convolution(buf519, buf520, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf521, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf522 = buf521; del buf521  # reuse
        # Topologically Sorted Source Nodes: [x_1320, x_1321], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf522, arg897_1, arg898_1, arg899_1, arg900_1, 112896, grid=grid(112896), stream=stream0)
        del arg897_1
        del arg898_1
        del arg899_1
        del arg900_1
        buf523 = buf520; del buf520  # reuse
        # Topologically Sorted Source Nodes: [x_1320, x_1321, x_1322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg901_1, buf523, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg901_1
        # Topologically Sorted Source Nodes: [x_1320, x_1321, x_1322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf524 = extern_kernels.convolution(buf522, buf523, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf524, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf522
        buf525 = buf519; del buf519  # reuse
        # Topologically Sorted Source Nodes: [x_1323, x_1324, x_1325], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf525, buf524, arg902_1, arg903_1, arg904_1, arg905_1, 112896, grid=grid(112896), stream=stream0)
        del arg902_1
        del arg903_1
        del arg904_1
        del arg905_1
        del buf524
        buf526 = buf523; del buf523  # reuse
        # Topologically Sorted Source Nodes: [x_1326], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg906_1, buf526, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg906_1
        # Topologically Sorted Source Nodes: [x_1326], Original ATen: [aten.convolution]
        buf527 = extern_kernels.convolution(buf525, buf526, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf527, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf528 = buf527; del buf527  # reuse
        # Topologically Sorted Source Nodes: [x_1327, x_1328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf528, arg907_1, arg908_1, arg909_1, arg910_1, 112896, grid=grid(112896), stream=stream0)
        del arg907_1
        del arg908_1
        del arg909_1
        del arg910_1
        buf529 = buf526; del buf526  # reuse
        # Topologically Sorted Source Nodes: [x_1327, x_1328, x_1329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg911_1, buf529, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg911_1
        # Topologically Sorted Source Nodes: [x_1327, x_1328, x_1329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf530 = extern_kernels.convolution(buf528, buf529, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf530, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf528
        buf531 = buf525; del buf525  # reuse
        # Topologically Sorted Source Nodes: [x_1330, x_1331, x_1332], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf531, buf530, arg912_1, arg913_1, arg914_1, arg915_1, 112896, grid=grid(112896), stream=stream0)
        del arg912_1
        del arg913_1
        del arg914_1
        del arg915_1
        del buf530
        buf532 = buf529; del buf529  # reuse
        # Topologically Sorted Source Nodes: [x_1333], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg916_1, buf532, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg916_1
        # Topologically Sorted Source Nodes: [x_1333], Original ATen: [aten.convolution]
        buf533 = extern_kernels.convolution(buf531, buf532, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf533, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf534 = buf533; del buf533  # reuse
        # Topologically Sorted Source Nodes: [x_1334, x_1335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf534, arg917_1, arg918_1, arg919_1, arg920_1, 112896, grid=grid(112896), stream=stream0)
        del arg917_1
        del arg918_1
        del arg919_1
        del arg920_1
        buf535 = buf532; del buf532  # reuse
        # Topologically Sorted Source Nodes: [x_1334, x_1335, x_1336], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg921_1, buf535, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg921_1
        # Topologically Sorted Source Nodes: [x_1334, x_1335, x_1336], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf536 = extern_kernels.convolution(buf534, buf535, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf534
        buf537 = buf531; del buf531  # reuse
        # Topologically Sorted Source Nodes: [x_1337, x_1338, x_1339], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf537, buf536, arg922_1, arg923_1, arg924_1, arg925_1, 112896, grid=grid(112896), stream=stream0)
        del arg922_1
        del arg923_1
        del arg924_1
        del arg925_1
        del buf536
        # Topologically Sorted Source Nodes: [input_340], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf537, arg986_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf596, (8, 36, 14, 14), (7056, 1, 504, 36))
        del arg986_1
        buf597 = reinterpret_tensor(buf411, (8, 36, 28, 28), (28224, 784, 28, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [input_341, input_342], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_24.run(buf596, arg987_1, arg988_1, arg989_1, arg990_1, buf597, 225792, grid=grid(225792), stream=stream0)
        del arg987_1
        del arg988_1
        del arg989_1
        del arg990_1
        del buf596
        buf540 = empty_strided_cuda((144, 72, 3, 3), (648, 1, 216, 72), torch.float32)
        # Topologically Sorted Source Nodes: [input_326], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(arg801_1, buf540, 10368, 9, grid=grid(10368, 9), stream=stream0)
        del arg801_1
        # Topologically Sorted Source Nodes: [input_326], Original ATen: [aten.convolution]
        buf541 = extern_kernels.convolution(buf513, buf540, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf513
        buf542 = buf541; del buf541  # reuse
        # Topologically Sorted Source Nodes: [input_327, input_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf542, arg802_1, arg803_1, arg804_1, arg805_1, 56448, grid=grid(56448), stream=stream0)
        del arg802_1
        del arg803_1
        del arg804_1
        del arg805_1
        buf543 = empty_strided_cuda((144, 144, 3, 3), (1296, 1, 432, 144), torch.float32)
        # Topologically Sorted Source Nodes: [x_1340], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(arg926_1, buf543, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg926_1
        # Topologically Sorted Source Nodes: [x_1340], Original ATen: [aten.convolution]
        buf544 = extern_kernels.convolution(buf542, buf543, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf544, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf545 = buf544; del buf544  # reuse
        # Topologically Sorted Source Nodes: [x_1341, x_1342], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf545, arg927_1, arg928_1, arg929_1, arg930_1, 56448, grid=grid(56448), stream=stream0)
        del arg927_1
        del arg928_1
        del arg929_1
        del arg930_1
        buf546 = buf543; del buf543  # reuse
        # Topologically Sorted Source Nodes: [x_1341, x_1342, x_1343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_33.run(arg931_1, buf546, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg931_1
        # Topologically Sorted Source Nodes: [x_1341, x_1342, x_1343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf547 = extern_kernels.convolution(buf545, buf546, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf547, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf545
        buf548 = buf542; del buf542  # reuse
        # Topologically Sorted Source Nodes: [x_1344, x_1345, x_1346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf548, buf547, arg932_1, arg933_1, arg934_1, arg935_1, 56448, grid=grid(56448), stream=stream0)
        del arg932_1
        del arg933_1
        del arg934_1
        del arg935_1
        del buf547
        buf549 = buf546; del buf546  # reuse
        # Topologically Sorted Source Nodes: [x_1347], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(arg936_1, buf549, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg936_1
        # Topologically Sorted Source Nodes: [x_1347], Original ATen: [aten.convolution]
        buf550 = extern_kernels.convolution(buf548, buf549, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf551 = buf550; del buf550  # reuse
        # Topologically Sorted Source Nodes: [x_1348, x_1349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf551, arg937_1, arg938_1, arg939_1, arg940_1, 56448, grid=grid(56448), stream=stream0)
        del arg937_1
        del arg938_1
        del arg939_1
        del arg940_1
        buf552 = buf549; del buf549  # reuse
        # Topologically Sorted Source Nodes: [x_1348, x_1349, x_1350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_33.run(arg941_1, buf552, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg941_1
        # Topologically Sorted Source Nodes: [x_1348, x_1349, x_1350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf553 = extern_kernels.convolution(buf551, buf552, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf553, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf551
        buf554 = buf548; del buf548  # reuse
        # Topologically Sorted Source Nodes: [x_1351, x_1352, x_1353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf554, buf553, arg942_1, arg943_1, arg944_1, arg945_1, 56448, grid=grid(56448), stream=stream0)
        del arg942_1
        del arg943_1
        del arg944_1
        del arg945_1
        del buf553
        buf555 = buf552; del buf552  # reuse
        # Topologically Sorted Source Nodes: [x_1354], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(arg946_1, buf555, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg946_1
        # Topologically Sorted Source Nodes: [x_1354], Original ATen: [aten.convolution]
        buf556 = extern_kernels.convolution(buf554, buf555, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf556, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf557 = buf556; del buf556  # reuse
        # Topologically Sorted Source Nodes: [x_1355, x_1356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf557, arg947_1, arg948_1, arg949_1, arg950_1, 56448, grid=grid(56448), stream=stream0)
        del arg947_1
        del arg948_1
        del arg949_1
        del arg950_1
        buf558 = buf555; del buf555  # reuse
        # Topologically Sorted Source Nodes: [x_1355, x_1356, x_1357], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_33.run(arg951_1, buf558, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg951_1
        # Topologically Sorted Source Nodes: [x_1355, x_1356, x_1357], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf559 = extern_kernels.convolution(buf557, buf558, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf559, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf557
        buf560 = buf554; del buf554  # reuse
        # Topologically Sorted Source Nodes: [x_1358, x_1359, x_1360], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf560, buf559, arg952_1, arg953_1, arg954_1, arg955_1, 56448, grid=grid(56448), stream=stream0)
        del arg952_1
        del arg953_1
        del arg954_1
        del arg955_1
        del buf559
        buf561 = buf558; del buf558  # reuse
        # Topologically Sorted Source Nodes: [x_1361], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(arg956_1, buf561, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg956_1
        # Topologically Sorted Source Nodes: [x_1361], Original ATen: [aten.convolution]
        buf562 = extern_kernels.convolution(buf560, buf561, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf562, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf563 = buf562; del buf562  # reuse
        # Topologically Sorted Source Nodes: [x_1362, x_1363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf563, arg957_1, arg958_1, arg959_1, arg960_1, 56448, grid=grid(56448), stream=stream0)
        del arg957_1
        del arg958_1
        del arg959_1
        del arg960_1
        buf564 = buf561; del buf561  # reuse
        # Topologically Sorted Source Nodes: [x_1362, x_1363, x_1364], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_33.run(arg961_1, buf564, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg961_1
        # Topologically Sorted Source Nodes: [x_1362, x_1363, x_1364], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf565 = extern_kernels.convolution(buf563, buf564, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf565, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf563
        buf566 = buf560; del buf560  # reuse
        # Topologically Sorted Source Nodes: [x_1365, x_1366, x_1367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf566, buf565, arg962_1, arg963_1, arg964_1, arg965_1, 56448, grid=grid(56448), stream=stream0)
        del arg962_1
        del arg963_1
        del arg964_1
        del arg965_1
        del buf565
        # Topologically Sorted Source Nodes: [input_343], Original ATen: [aten.convolution]
        buf598 = extern_kernels.convolution(buf566, arg991_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf598, (8, 36, 7, 7), (1764, 1, 252, 36))
        del arg991_1
        buf599 = reinterpret_tensor(buf499, (8, 36, 28, 28), (28224, 784, 28, 1), 0); del buf499  # reuse
        # Topologically Sorted Source Nodes: [input_344, input_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_35.run(buf598, arg992_1, arg993_1, arg994_1, arg995_1, buf599, 225792, grid=grid(225792), stream=stream0)
        del arg992_1
        del arg993_1
        del arg994_1
        del arg995_1
        del buf598
        buf502 = buf496; del buf496  # reuse
        buf600 = buf595; del buf595  # reuse
        # Topologically Sorted Source Nodes: [x_1309, x_1310, x_1311, input_339, y_94, y_95, y_96, shortcut_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf502, buf600, buf501, arg882_1, arg883_1, arg884_1, arg885_1, arg982_1, arg983_1, arg984_1, arg985_1, buf597, buf599, 6272, 36, grid=grid(6272, 36), stream=stream0)
        del arg882_1
        del arg883_1
        del arg884_1
        del arg885_1
        del arg982_1
        del arg983_1
        del arg984_1
        del arg985_1
        del buf501
        del buf597
        del buf599
        # Topologically Sorted Source Nodes: [input_329], Original ATen: [aten.convolution]
        buf503 = extern_kernels.convolution(buf502, arg966_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf503, (8, 18, 28, 28), (14112, 1, 504, 18))
        del arg966_1
        # Topologically Sorted Source Nodes: [input_332], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, arg971_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (8, 18, 14, 14), (3528, 1, 252, 18))
        del arg971_1
        # Topologically Sorted Source Nodes: [input_335], Original ATen: [aten.convolution]
        buf567 = extern_kernels.convolution(buf566, arg976_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf567, (8, 18, 7, 7), (882, 1, 126, 18))
        del arg976_1
        buf569 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [input_330, input_331, y_91, input_333, input_334, y_92, input_336, input_337, y_93, shortcut_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_37.run(buf503, arg967_1, arg968_1, arg969_1, arg970_1, buf538, arg972_1, arg973_1, arg974_1, arg975_1, buf567, arg977_1, arg978_1, arg979_1, arg980_1, buf473, buf569, 144, 3136, grid=grid(144, 3136), stream=stream0)
        del arg967_1
        del arg968_1
        del arg969_1
        del arg970_1
        del arg972_1
        del arg973_1
        del arg974_1
        del arg975_1
        del arg977_1
        del arg978_1
        del arg979_1
        del arg980_1
        del buf538
        del buf567
        buf570 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [x_1368], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1046_1, buf570, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1046_1
        # Topologically Sorted Source Nodes: [x_1368], Original ATen: [aten.convolution]
        buf571 = extern_kernels.convolution(buf569, buf570, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf571, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf572 = buf571; del buf571  # reuse
        # Topologically Sorted Source Nodes: [x_1369, x_1370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf572, arg1047_1, arg1048_1, arg1049_1, arg1050_1, 451584, grid=grid(451584), stream=stream0)
        del arg1047_1
        del arg1048_1
        del arg1049_1
        del arg1050_1
        buf573 = buf570; del buf570  # reuse
        # Topologically Sorted Source Nodes: [x_1369, x_1370, x_1371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg1051_1, buf573, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1051_1
        # Topologically Sorted Source Nodes: [x_1369, x_1370, x_1371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf574 = extern_kernels.convolution(buf572, buf573, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf572
        buf575 = buf569; del buf569  # reuse
        # Topologically Sorted Source Nodes: [x_1372, x_1373, x_1374], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf575, buf574, arg1052_1, arg1053_1, arg1054_1, arg1055_1, 451584, grid=grid(451584), stream=stream0)
        del arg1052_1
        del arg1053_1
        del arg1054_1
        del arg1055_1
        del buf574
        buf576 = buf573; del buf573  # reuse
        # Topologically Sorted Source Nodes: [x_1375], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1056_1, buf576, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1056_1
        # Topologically Sorted Source Nodes: [x_1375], Original ATen: [aten.convolution]
        buf577 = extern_kernels.convolution(buf575, buf576, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf577, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf578 = buf577; del buf577  # reuse
        # Topologically Sorted Source Nodes: [x_1376, x_1377], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf578, arg1057_1, arg1058_1, arg1059_1, arg1060_1, 451584, grid=grid(451584), stream=stream0)
        del arg1057_1
        del arg1058_1
        del arg1059_1
        del arg1060_1
        buf579 = buf576; del buf576  # reuse
        # Topologically Sorted Source Nodes: [x_1376, x_1377, x_1378], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg1061_1, buf579, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1061_1
        # Topologically Sorted Source Nodes: [x_1376, x_1377, x_1378], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf580 = extern_kernels.convolution(buf578, buf579, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf580, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf578
        buf581 = buf575; del buf575  # reuse
        # Topologically Sorted Source Nodes: [x_1379, x_1380, x_1381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf581, buf580, arg1062_1, arg1063_1, arg1064_1, arg1065_1, 451584, grid=grid(451584), stream=stream0)
        del arg1062_1
        del arg1063_1
        del arg1064_1
        del arg1065_1
        del buf580
        buf582 = buf579; del buf579  # reuse
        # Topologically Sorted Source Nodes: [x_1382], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1066_1, buf582, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1066_1
        # Topologically Sorted Source Nodes: [x_1382], Original ATen: [aten.convolution]
        buf583 = extern_kernels.convolution(buf581, buf582, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf583, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf584 = buf583; del buf583  # reuse
        # Topologically Sorted Source Nodes: [x_1383, x_1384], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf584, arg1067_1, arg1068_1, arg1069_1, arg1070_1, 451584, grid=grid(451584), stream=stream0)
        del arg1067_1
        del arg1068_1
        del arg1069_1
        del arg1070_1
        buf585 = buf582; del buf582  # reuse
        # Topologically Sorted Source Nodes: [x_1383, x_1384, x_1385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg1071_1, buf585, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1071_1
        # Topologically Sorted Source Nodes: [x_1383, x_1384, x_1385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf586 = extern_kernels.convolution(buf584, buf585, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf584
        buf587 = buf581; del buf581  # reuse
        # Topologically Sorted Source Nodes: [x_1386, x_1387, x_1388], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf587, buf586, arg1072_1, arg1073_1, arg1074_1, arg1075_1, 451584, grid=grid(451584), stream=stream0)
        del arg1072_1
        del arg1073_1
        del arg1074_1
        del arg1075_1
        del buf586
        buf588 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [x_1389], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1076_1, buf588, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1076_1
        # Topologically Sorted Source Nodes: [x_1389], Original ATen: [aten.convolution]
        buf589 = extern_kernels.convolution(buf587, buf588, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf589, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf590 = buf589; del buf589  # reuse
        # Topologically Sorted Source Nodes: [x_1390, x_1391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf590, arg1077_1, arg1078_1, arg1079_1, arg1080_1, 451584, grid=grid(451584), stream=stream0)
        del arg1077_1
        del arg1078_1
        del arg1079_1
        del arg1080_1
        buf591 = buf588; del buf588  # reuse
        # Topologically Sorted Source Nodes: [x_1390, x_1391, x_1392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg1081_1, buf591, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1081_1
        # Topologically Sorted Source Nodes: [x_1390, x_1391, x_1392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf592 = extern_kernels.convolution(buf590, buf591, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf592, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf590
        buf593 = buf587; del buf587  # reuse
        # Topologically Sorted Source Nodes: [x_1393, x_1394, x_1395], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf593, buf592, arg1082_1, arg1083_1, arg1084_1, arg1085_1, 451584, grid=grid(451584), stream=stream0)
        del arg1082_1
        del arg1083_1
        del arg1084_1
        del arg1085_1
        del buf592
        buf601 = reinterpret_tensor(buf508, (36, 36, 3, 3), (324, 1, 108, 36), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [x_1396], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg1086_1, buf601, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1086_1
        # Topologically Sorted Source Nodes: [x_1396], Original ATen: [aten.convolution]
        buf602 = extern_kernels.convolution(buf600, buf601, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf602, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf603 = buf602; del buf602  # reuse
        # Topologically Sorted Source Nodes: [x_1397, x_1398], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf603, arg1087_1, arg1088_1, arg1089_1, arg1090_1, 225792, grid=grid(225792), stream=stream0)
        del arg1087_1
        del arg1088_1
        del arg1089_1
        del arg1090_1
        buf604 = buf601; del buf601  # reuse
        # Topologically Sorted Source Nodes: [x_1397, x_1398, x_1399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg1091_1, buf604, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1091_1
        # Topologically Sorted Source Nodes: [x_1397, x_1398, x_1399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf605 = extern_kernels.convolution(buf603, buf604, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf605, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf603
        buf606 = buf600; del buf600  # reuse
        # Topologically Sorted Source Nodes: [x_1400, x_1401, x_1402], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf606, buf605, arg1092_1, arg1093_1, arg1094_1, arg1095_1, 225792, grid=grid(225792), stream=stream0)
        del arg1092_1
        del arg1093_1
        del arg1094_1
        del arg1095_1
        del buf605
        buf607 = buf604; del buf604  # reuse
        # Topologically Sorted Source Nodes: [x_1403], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg1096_1, buf607, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1096_1
        # Topologically Sorted Source Nodes: [x_1403], Original ATen: [aten.convolution]
        buf608 = extern_kernels.convolution(buf606, buf607, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf608, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf609 = buf608; del buf608  # reuse
        # Topologically Sorted Source Nodes: [x_1404, x_1405], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf609, arg1097_1, arg1098_1, arg1099_1, arg1100_1, 225792, grid=grid(225792), stream=stream0)
        del arg1097_1
        del arg1098_1
        del arg1099_1
        del arg1100_1
        buf610 = buf607; del buf607  # reuse
        # Topologically Sorted Source Nodes: [x_1404, x_1405, x_1406], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg1101_1, buf610, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1101_1
        # Topologically Sorted Source Nodes: [x_1404, x_1405, x_1406], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf611 = extern_kernels.convolution(buf609, buf610, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf611, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf609
        buf612 = buf606; del buf606  # reuse
        # Topologically Sorted Source Nodes: [x_1407, x_1408, x_1409], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf612, buf611, arg1102_1, arg1103_1, arg1104_1, arg1105_1, 225792, grid=grid(225792), stream=stream0)
        del arg1102_1
        del arg1103_1
        del arg1104_1
        del arg1105_1
        del buf611
        buf613 = buf610; del buf610  # reuse
        # Topologically Sorted Source Nodes: [x_1410], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg1106_1, buf613, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1106_1
        # Topologically Sorted Source Nodes: [x_1410], Original ATen: [aten.convolution]
        buf614 = extern_kernels.convolution(buf612, buf613, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf614, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf615 = buf614; del buf614  # reuse
        # Topologically Sorted Source Nodes: [x_1411, x_1412], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf615, arg1107_1, arg1108_1, arg1109_1, arg1110_1, 225792, grid=grid(225792), stream=stream0)
        del arg1107_1
        del arg1108_1
        del arg1109_1
        del arg1110_1
        buf616 = buf613; del buf613  # reuse
        # Topologically Sorted Source Nodes: [x_1411, x_1412, x_1413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg1111_1, buf616, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1111_1
        # Topologically Sorted Source Nodes: [x_1411, x_1412, x_1413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf617 = extern_kernels.convolution(buf615, buf616, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf617, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf615
        buf618 = buf612; del buf612  # reuse
        # Topologically Sorted Source Nodes: [x_1414, x_1415, x_1416], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf618, buf617, arg1112_1, arg1113_1, arg1114_1, arg1115_1, 225792, grid=grid(225792), stream=stream0)
        del arg1112_1
        del arg1113_1
        del arg1114_1
        del arg1115_1
        del buf617
        buf619 = buf616; del buf616  # reuse
        # Topologically Sorted Source Nodes: [x_1417], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg1116_1, buf619, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1116_1
        # Topologically Sorted Source Nodes: [x_1417], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf618, buf619, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf621 = buf620; del buf620  # reuse
        # Topologically Sorted Source Nodes: [x_1418, x_1419], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf621, arg1117_1, arg1118_1, arg1119_1, arg1120_1, 225792, grid=grid(225792), stream=stream0)
        del arg1117_1
        del arg1118_1
        del arg1119_1
        del arg1120_1
        buf622 = buf619; del buf619  # reuse
        # Topologically Sorted Source Nodes: [x_1418, x_1419, x_1420], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg1121_1, buf622, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1121_1
        # Topologically Sorted Source Nodes: [x_1418, x_1419, x_1420], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf623 = extern_kernels.convolution(buf621, buf622, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf623, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf740 = buf594; del buf594  # reuse
        # Topologically Sorted Source Nodes: [input_380], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(arg1221_1, buf740, 648, 9, grid=grid(648, 9), stream=stream0)
        del arg1221_1
        # Topologically Sorted Source Nodes: [input_380], Original ATen: [aten.convolution]
        buf741 = extern_kernels.convolution(buf593, buf740, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf741, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf627 = buf591; del buf591  # reuse
        # Topologically Sorted Source Nodes: [input_346], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg996_1, buf627, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg996_1
        # Topologically Sorted Source Nodes: [input_346], Original ATen: [aten.convolution]
        buf628 = extern_kernels.convolution(buf473, buf627, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf628, (8, 18, 28, 28), (14112, 1, 504, 18))
        buf629 = buf628; del buf628  # reuse
        # Topologically Sorted Source Nodes: [input_347, input_348], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf629, arg997_1, arg998_1, arg999_1, arg1000_1, 112896, grid=grid(112896), stream=stream0)
        del arg1000_1
        del arg997_1
        del arg998_1
        del arg999_1
        buf630 = reinterpret_tensor(buf622, (72, 18, 3, 3), (162, 1, 54, 18), 0); del buf622  # reuse
        # Topologically Sorted Source Nodes: [input_347, input_348, input_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg1001_1, buf630, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1001_1
        # Topologically Sorted Source Nodes: [input_347, input_348, input_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf631 = extern_kernels.convolution(buf629, buf630, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf631, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf632 = buf510; del buf510  # reuse
        # Topologically Sorted Source Nodes: [input_351], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg1006_1, buf632, 2592, 9, grid=grid(2592, 9), stream=stream0)
        del arg1006_1
        # Topologically Sorted Source Nodes: [input_351], Original ATen: [aten.convolution]
        buf633 = extern_kernels.convolution(buf502, buf632, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (8, 72, 14, 14), (14112, 1, 1008, 72))
        # Topologically Sorted Source Nodes: [input_353], Original ATen: [aten.convolution]
        buf635 = extern_kernels.convolution(buf566, arg1011_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf635, (8, 72, 7, 7), (3528, 1, 504, 72))
        del arg1011_1
        buf636 = reinterpret_tensor(buf629, (8, 72, 14, 14), (14112, 196, 14, 1), 0); del buf629  # reuse
        # Topologically Sorted Source Nodes: [input_354, input_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_38.run(buf635, arg1012_1, arg1013_1, arg1014_1, arg1015_1, buf636, 112896, grid=grid(112896), stream=stream0)
        del arg1012_1
        del arg1013_1
        del arg1014_1
        del arg1015_1
        del buf635
        buf634 = buf631; del buf631  # reuse
        buf637 = reinterpret_tensor(buf503, (8, 72, 14, 14), (14112, 1, 1008, 72), 0); del buf503  # reuse
        # Topologically Sorted Source Nodes: [input_350, input_352, y_97, y_98, y_99, shortcut_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39.run(buf634, arg1002_1, arg1003_1, arg1004_1, arg1005_1, buf633, arg1007_1, arg1008_1, arg1009_1, arg1010_1, buf537, buf636, buf637, 1568, 72, grid=grid(1568, 72), stream=stream0)
        del arg1002_1
        del arg1003_1
        del arg1004_1
        del arg1005_1
        del arg1007_1
        del arg1008_1
        del arg1009_1
        del arg1010_1
        del buf633
        del buf634
        del buf636
        buf638 = buf535; del buf535  # reuse
        # Topologically Sorted Source Nodes: [x_1424], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg1126_1, buf638, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1126_1
        # Topologically Sorted Source Nodes: [x_1424], Original ATen: [aten.convolution]
        buf639 = extern_kernels.convolution(buf637, buf638, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf639, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf640 = buf639; del buf639  # reuse
        # Topologically Sorted Source Nodes: [x_1425, x_1426], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf640, arg1127_1, arg1128_1, arg1129_1, arg1130_1, 112896, grid=grid(112896), stream=stream0)
        del arg1127_1
        del arg1128_1
        del arg1129_1
        del arg1130_1
        buf641 = buf638; del buf638  # reuse
        # Topologically Sorted Source Nodes: [x_1425, x_1426, x_1427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg1131_1, buf641, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1131_1
        # Topologically Sorted Source Nodes: [x_1425, x_1426, x_1427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf642 = extern_kernels.convolution(buf640, buf641, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf642, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf640
        buf643 = buf637; del buf637  # reuse
        # Topologically Sorted Source Nodes: [x_1428, x_1429, x_1430], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf643, buf642, arg1132_1, arg1133_1, arg1134_1, arg1135_1, 112896, grid=grid(112896), stream=stream0)
        del arg1132_1
        del arg1133_1
        del arg1134_1
        del arg1135_1
        del buf642
        buf644 = buf641; del buf641  # reuse
        # Topologically Sorted Source Nodes: [x_1431], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg1136_1, buf644, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1136_1
        # Topologically Sorted Source Nodes: [x_1431], Original ATen: [aten.convolution]
        buf645 = extern_kernels.convolution(buf643, buf644, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf645, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf646 = buf645; del buf645  # reuse
        # Topologically Sorted Source Nodes: [x_1432, x_1433], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf646, arg1137_1, arg1138_1, arg1139_1, arg1140_1, 112896, grid=grid(112896), stream=stream0)
        del arg1137_1
        del arg1138_1
        del arg1139_1
        del arg1140_1
        buf647 = buf644; del buf644  # reuse
        # Topologically Sorted Source Nodes: [x_1432, x_1433, x_1434], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg1141_1, buf647, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1141_1
        # Topologically Sorted Source Nodes: [x_1432, x_1433, x_1434], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf648 = extern_kernels.convolution(buf646, buf647, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf648, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf646
        buf649 = buf643; del buf643  # reuse
        # Topologically Sorted Source Nodes: [x_1435, x_1436, x_1437], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf649, buf648, arg1142_1, arg1143_1, arg1144_1, arg1145_1, 112896, grid=grid(112896), stream=stream0)
        del arg1142_1
        del arg1143_1
        del arg1144_1
        del arg1145_1
        del buf648
        buf650 = buf647; del buf647  # reuse
        # Topologically Sorted Source Nodes: [x_1438], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg1146_1, buf650, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1146_1
        # Topologically Sorted Source Nodes: [x_1438], Original ATen: [aten.convolution]
        buf651 = extern_kernels.convolution(buf649, buf650, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf651, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf652 = buf651; del buf651  # reuse
        # Topologically Sorted Source Nodes: [x_1439, x_1440], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf652, arg1147_1, arg1148_1, arg1149_1, arg1150_1, 112896, grid=grid(112896), stream=stream0)
        del arg1147_1
        del arg1148_1
        del arg1149_1
        del arg1150_1
        buf653 = buf650; del buf650  # reuse
        # Topologically Sorted Source Nodes: [x_1439, x_1440, x_1441], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg1151_1, buf653, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1151_1
        # Topologically Sorted Source Nodes: [x_1439, x_1440, x_1441], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf654 = extern_kernels.convolution(buf652, buf653, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf654, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf652
        buf655 = buf649; del buf649  # reuse
        # Topologically Sorted Source Nodes: [x_1442, x_1443, x_1444], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf655, buf654, arg1152_1, arg1153_1, arg1154_1, arg1155_1, 112896, grid=grid(112896), stream=stream0)
        del arg1152_1
        del arg1153_1
        del arg1154_1
        del arg1155_1
        del buf654
        buf656 = buf653; del buf653  # reuse
        # Topologically Sorted Source Nodes: [x_1445], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg1156_1, buf656, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1156_1
        # Topologically Sorted Source Nodes: [x_1445], Original ATen: [aten.convolution]
        buf657 = extern_kernels.convolution(buf655, buf656, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf657, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf658 = buf657; del buf657  # reuse
        # Topologically Sorted Source Nodes: [x_1446, x_1447], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf658, arg1157_1, arg1158_1, arg1159_1, arg1160_1, 112896, grid=grid(112896), stream=stream0)
        del arg1157_1
        del arg1158_1
        del arg1159_1
        del arg1160_1
        buf659 = buf656; del buf656  # reuse
        # Topologically Sorted Source Nodes: [x_1446, x_1447, x_1448], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg1161_1, buf659, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1161_1
        # Topologically Sorted Source Nodes: [x_1446, x_1447, x_1448], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf660 = extern_kernels.convolution(buf658, buf659, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf660, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf658
        buf661 = buf655; del buf655  # reuse
        # Topologically Sorted Source Nodes: [x_1449, x_1450, x_1451], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf661, buf660, arg1162_1, arg1163_1, arg1164_1, arg1165_1, 112896, grid=grid(112896), stream=stream0)
        del arg1162_1
        del arg1163_1
        del arg1164_1
        del arg1165_1
        del buf660
        # Topologically Sorted Source Nodes: [input_382], Original ATen: [aten.convolution]
        buf742 = extern_kernels.convolution(buf661, arg1226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf742, (8, 36, 14, 14), (7056, 1, 504, 36))
        del arg1226_1
        buf743 = reinterpret_tensor(buf621, (8, 36, 28, 28), (28224, 784, 28, 1), 0); del buf621  # reuse
        # Topologically Sorted Source Nodes: [input_383, input_384], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_24.run(buf742, arg1227_1, arg1228_1, arg1229_1, arg1230_1, buf743, 225792, grid=grid(225792), stream=stream0)
        del arg1227_1
        del arg1228_1
        del arg1229_1
        del arg1230_1
        del buf742
        buf664 = buf627; del buf627  # reuse
        # Topologically Sorted Source Nodes: [input_356], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1016_1, buf664, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1016_1
        # Topologically Sorted Source Nodes: [input_356], Original ATen: [aten.convolution]
        buf665 = extern_kernels.convolution(buf473, buf664, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf665, (8, 18, 28, 28), (14112, 1, 504, 18))
        buf666 = buf665; del buf665  # reuse
        # Topologically Sorted Source Nodes: [input_357, input_358], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf666, arg1017_1, arg1018_1, arg1019_1, arg1020_1, 112896, grid=grid(112896), stream=stream0)
        del arg1017_1
        del arg1018_1
        del arg1019_1
        del arg1020_1
        buf667 = buf664; del buf664  # reuse
        # Topologically Sorted Source Nodes: [input_357, input_358, input_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg1021_1, buf667, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1021_1
        # Topologically Sorted Source Nodes: [input_357, input_358, input_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf668 = extern_kernels.convolution(buf666, buf667, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf668, (8, 18, 14, 14), (3528, 1, 252, 18))
        del buf666
        buf669 = buf668; del buf668  # reuse
        # Topologically Sorted Source Nodes: [input_360, input_361], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf669, arg1022_1, arg1023_1, arg1024_1, arg1025_1, 28224, grid=grid(28224), stream=stream0)
        del arg1022_1
        del arg1023_1
        del arg1024_1
        del arg1025_1
        buf670 = reinterpret_tensor(buf632, (144, 18, 3, 3), (162, 1, 54, 18), 0); del buf632  # reuse
        # Topologically Sorted Source Nodes: [input_360, input_361, input_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41.run(arg1026_1, buf670, 2592, 9, grid=grid(2592, 9), stream=stream0)
        del arg1026_1
        # Topologically Sorted Source Nodes: [input_360, input_361, input_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf671 = extern_kernels.convolution(buf669, buf670, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf671, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf669
        buf672 = reinterpret_tensor(buf630, (36, 36, 3, 3), (324, 1, 108, 36), 0); del buf630  # reuse
        # Topologically Sorted Source Nodes: [input_364], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg1031_1, buf672, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1031_1
        # Topologically Sorted Source Nodes: [input_364], Original ATen: [aten.convolution]
        buf673 = extern_kernels.convolution(buf502, buf672, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf673, (8, 36, 14, 14), (7056, 1, 504, 36))
        buf674 = buf673; del buf673  # reuse
        # Topologically Sorted Source Nodes: [input_365, input_366], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_42.run(buf674, arg1032_1, arg1033_1, arg1034_1, arg1035_1, 56448, grid=grid(56448), stream=stream0)
        del arg1032_1
        del arg1033_1
        del arg1034_1
        del arg1035_1
        buf675 = reinterpret_tensor(buf659, (144, 36, 3, 3), (324, 1, 108, 36), 0); del buf659  # reuse
        # Topologically Sorted Source Nodes: [input_365, input_366, input_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_43.run(arg1036_1, buf675, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1036_1
        # Topologically Sorted Source Nodes: [input_365, input_366, input_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf676 = extern_kernels.convolution(buf674, buf675, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf676, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf674
        buf678 = buf540; del buf540  # reuse
        # Topologically Sorted Source Nodes: [input_369], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(arg1041_1, buf678, 10368, 9, grid=grid(10368, 9), stream=stream0)
        del arg1041_1
        # Topologically Sorted Source Nodes: [input_369], Original ATen: [aten.convolution]
        buf679 = extern_kernels.convolution(buf537, buf678, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf679, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf537
        buf677 = buf671; del buf671  # reuse
        buf680 = buf566; del buf566  # reuse
        # Topologically Sorted Source Nodes: [input_363, input_368, y_100, input_370, y_101, y_102, shortcut_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf677, buf680, arg1027_1, arg1028_1, arg1029_1, arg1030_1, buf676, arg1037_1, arg1038_1, arg1039_1, arg1040_1, buf679, arg1042_1, arg1043_1, arg1044_1, arg1045_1, 56448, grid=grid(56448), stream=stream0)
        del arg1027_1
        del arg1028_1
        del arg1029_1
        del arg1030_1
        del arg1037_1
        del arg1038_1
        del arg1039_1
        del arg1040_1
        del arg1042_1
        del arg1043_1
        del arg1044_1
        del arg1045_1
        del buf676
        del buf677
        del buf679
        buf681 = buf564; del buf564  # reuse
        # Topologically Sorted Source Nodes: [x_1452], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(arg1166_1, buf681, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1166_1
        # Topologically Sorted Source Nodes: [x_1452], Original ATen: [aten.convolution]
        buf682 = extern_kernels.convolution(buf680, buf681, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf682, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf683 = buf682; del buf682  # reuse
        # Topologically Sorted Source Nodes: [x_1453, x_1454], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf683, arg1167_1, arg1168_1, arg1169_1, arg1170_1, 56448, grid=grid(56448), stream=stream0)
        del arg1167_1
        del arg1168_1
        del arg1169_1
        del arg1170_1
        buf684 = buf681; del buf681  # reuse
        # Topologically Sorted Source Nodes: [x_1453, x_1454, x_1455], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_33.run(arg1171_1, buf684, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1171_1
        # Topologically Sorted Source Nodes: [x_1453, x_1454, x_1455], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf685 = extern_kernels.convolution(buf683, buf684, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf685, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf683
        buf686 = buf680; del buf680  # reuse
        # Topologically Sorted Source Nodes: [x_1456, x_1457, x_1458], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf686, buf685, arg1172_1, arg1173_1, arg1174_1, arg1175_1, 56448, grid=grid(56448), stream=stream0)
        del arg1172_1
        del arg1173_1
        del arg1174_1
        del arg1175_1
        del buf685
        buf687 = buf684; del buf684  # reuse
        # Topologically Sorted Source Nodes: [x_1459], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(arg1176_1, buf687, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1176_1
        # Topologically Sorted Source Nodes: [x_1459], Original ATen: [aten.convolution]
        buf688 = extern_kernels.convolution(buf686, buf687, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf688, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf689 = buf688; del buf688  # reuse
        # Topologically Sorted Source Nodes: [x_1460, x_1461], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf689, arg1177_1, arg1178_1, arg1179_1, arg1180_1, 56448, grid=grid(56448), stream=stream0)
        del arg1177_1
        del arg1178_1
        del arg1179_1
        del arg1180_1
        buf690 = buf687; del buf687  # reuse
        # Topologically Sorted Source Nodes: [x_1460, x_1461, x_1462], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_33.run(arg1181_1, buf690, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1181_1
        # Topologically Sorted Source Nodes: [x_1460, x_1461, x_1462], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf691 = extern_kernels.convolution(buf689, buf690, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf691, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf689
        buf692 = buf686; del buf686  # reuse
        # Topologically Sorted Source Nodes: [x_1463, x_1464, x_1465], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf692, buf691, arg1182_1, arg1183_1, arg1184_1, arg1185_1, 56448, grid=grid(56448), stream=stream0)
        del arg1182_1
        del arg1183_1
        del arg1184_1
        del arg1185_1
        del buf691
        buf693 = buf690; del buf690  # reuse
        # Topologically Sorted Source Nodes: [x_1466], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(arg1186_1, buf693, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1186_1
        # Topologically Sorted Source Nodes: [x_1466], Original ATen: [aten.convolution]
        buf694 = extern_kernels.convolution(buf692, buf693, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf694, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf695 = buf694; del buf694  # reuse
        # Topologically Sorted Source Nodes: [x_1467, x_1468], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf695, arg1187_1, arg1188_1, arg1189_1, arg1190_1, 56448, grid=grid(56448), stream=stream0)
        del arg1187_1
        del arg1188_1
        del arg1189_1
        del arg1190_1
        buf696 = buf693; del buf693  # reuse
        # Topologically Sorted Source Nodes: [x_1467, x_1468, x_1469], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_33.run(arg1191_1, buf696, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1191_1
        # Topologically Sorted Source Nodes: [x_1467, x_1468, x_1469], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf697 = extern_kernels.convolution(buf695, buf696, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf697, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf695
        buf698 = buf692; del buf692  # reuse
        # Topologically Sorted Source Nodes: [x_1470, x_1471, x_1472], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf698, buf697, arg1192_1, arg1193_1, arg1194_1, arg1195_1, 56448, grid=grid(56448), stream=stream0)
        del arg1192_1
        del arg1193_1
        del arg1194_1
        del arg1195_1
        del buf697
        buf699 = buf696; del buf696  # reuse
        # Topologically Sorted Source Nodes: [x_1473], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(arg1196_1, buf699, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1196_1
        # Topologically Sorted Source Nodes: [x_1473], Original ATen: [aten.convolution]
        buf700 = extern_kernels.convolution(buf698, buf699, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf700, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf701 = buf700; del buf700  # reuse
        # Topologically Sorted Source Nodes: [x_1474, x_1475], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf701, arg1197_1, arg1198_1, arg1199_1, arg1200_1, 56448, grid=grid(56448), stream=stream0)
        del arg1197_1
        del arg1198_1
        del arg1199_1
        del arg1200_1
        buf702 = buf699; del buf699  # reuse
        # Topologically Sorted Source Nodes: [x_1474, x_1475, x_1476], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_33.run(arg1201_1, buf702, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1201_1
        # Topologically Sorted Source Nodes: [x_1474, x_1475, x_1476], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf703 = extern_kernels.convolution(buf701, buf702, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf703, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf701
        buf704 = buf698; del buf698  # reuse
        # Topologically Sorted Source Nodes: [x_1477, x_1478, x_1479], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf704, buf703, arg1202_1, arg1203_1, arg1204_1, arg1205_1, 56448, grid=grid(56448), stream=stream0)
        del arg1202_1
        del arg1203_1
        del arg1204_1
        del arg1205_1
        del buf703
        # Topologically Sorted Source Nodes: [input_385], Original ATen: [aten.convolution]
        buf744 = extern_kernels.convolution(buf704, arg1231_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf744, (8, 36, 7, 7), (1764, 1, 252, 36))
        del arg1231_1
        buf745 = reinterpret_tensor(buf502, (8, 36, 28, 28), (28224, 784, 28, 1), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [input_386, input_387], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_35.run(buf744, arg1232_1, arg1233_1, arg1234_1, arg1235_1, buf745, 225792, grid=grid(225792), stream=stream0)
        del arg1232_1
        del arg1233_1
        del arg1234_1
        del arg1235_1
        del buf744
        buf624 = buf618; del buf618  # reuse
        buf746 = buf741; del buf741  # reuse
        # Topologically Sorted Source Nodes: [x_1421, x_1422, x_1423, input_381, y_106, y_107, y_108, shortcut_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf624, buf746, buf623, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, buf743, buf745, 6272, 36, grid=grid(6272, 36), stream=stream0)
        del arg1122_1
        del arg1123_1
        del arg1124_1
        del arg1125_1
        del arg1222_1
        del arg1223_1
        del arg1224_1
        del arg1225_1
        del buf623
        del buf743
        del buf745
        # Topologically Sorted Source Nodes: [input_371], Original ATen: [aten.convolution]
        buf625 = extern_kernels.convolution(buf624, arg1206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf625, (8, 18, 28, 28), (14112, 1, 504, 18))
        del arg1206_1
        # Topologically Sorted Source Nodes: [input_374], Original ATen: [aten.convolution]
        buf662 = extern_kernels.convolution(buf661, arg1211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf662, (8, 18, 14, 14), (3528, 1, 252, 18))
        del arg1211_1
        # Topologically Sorted Source Nodes: [input_377], Original ATen: [aten.convolution]
        buf705 = extern_kernels.convolution(buf704, arg1216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf705, (8, 18, 7, 7), (882, 1, 126, 18))
        del arg1216_1
        buf707 = buf473; del buf473  # reuse
        # Topologically Sorted Source Nodes: [input_372, input_373, y_103, input_375, input_376, y_104, input_378, input_379, y_105, shortcut_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_37.run(buf625, arg1207_1, arg1208_1, arg1209_1, arg1210_1, buf662, arg1212_1, arg1213_1, arg1214_1, arg1215_1, buf705, arg1217_1, arg1218_1, arg1219_1, arg1220_1, buf593, buf707, 144, 3136, grid=grid(144, 3136), stream=stream0)
        del arg1207_1
        del arg1208_1
        del arg1209_1
        del arg1210_1
        del arg1212_1
        del arg1213_1
        del arg1214_1
        del arg1215_1
        del arg1217_1
        del arg1218_1
        del arg1219_1
        del arg1220_1
        del buf625
        del buf662
        del buf705
        buf708 = buf667; del buf667  # reuse
        # Topologically Sorted Source Nodes: [x_1480], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1286_1, buf708, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1286_1
        # Topologically Sorted Source Nodes: [x_1480], Original ATen: [aten.convolution]
        buf709 = extern_kernels.convolution(buf707, buf708, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf709, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf710 = buf709; del buf709  # reuse
        # Topologically Sorted Source Nodes: [x_1481, x_1482], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf710, arg1287_1, arg1288_1, arg1289_1, arg1290_1, 451584, grid=grid(451584), stream=stream0)
        del arg1287_1
        del arg1288_1
        del arg1289_1
        del arg1290_1
        buf711 = buf708; del buf708  # reuse
        # Topologically Sorted Source Nodes: [x_1481, x_1482, x_1483], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg1291_1, buf711, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1291_1
        # Topologically Sorted Source Nodes: [x_1481, x_1482, x_1483], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf712 = extern_kernels.convolution(buf710, buf711, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf712, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf710
        buf713 = buf707; del buf707  # reuse
        # Topologically Sorted Source Nodes: [x_1484, x_1485, x_1486], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf713, buf712, arg1292_1, arg1293_1, arg1294_1, arg1295_1, 451584, grid=grid(451584), stream=stream0)
        del arg1292_1
        del arg1293_1
        del arg1294_1
        del arg1295_1
        del buf712
        buf714 = buf711; del buf711  # reuse
        # Topologically Sorted Source Nodes: [x_1487], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1296_1, buf714, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1296_1
        # Topologically Sorted Source Nodes: [x_1487], Original ATen: [aten.convolution]
        buf715 = extern_kernels.convolution(buf713, buf714, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf715, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf716 = buf715; del buf715  # reuse
        # Topologically Sorted Source Nodes: [x_1488, x_1489], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf716, arg1297_1, arg1298_1, arg1299_1, arg1300_1, 451584, grid=grid(451584), stream=stream0)
        del arg1297_1
        del arg1298_1
        del arg1299_1
        del arg1300_1
        buf717 = buf714; del buf714  # reuse
        # Topologically Sorted Source Nodes: [x_1488, x_1489, x_1490], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg1301_1, buf717, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1301_1
        # Topologically Sorted Source Nodes: [x_1488, x_1489, x_1490], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf718 = extern_kernels.convolution(buf716, buf717, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf718, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf716
        buf719 = buf713; del buf713  # reuse
        # Topologically Sorted Source Nodes: [x_1491, x_1492, x_1493], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf719, buf718, arg1302_1, arg1303_1, arg1304_1, arg1305_1, 451584, grid=grid(451584), stream=stream0)
        del arg1302_1
        del arg1303_1
        del arg1304_1
        del arg1305_1
        del buf718
        buf720 = buf717; del buf717  # reuse
        # Topologically Sorted Source Nodes: [x_1494], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1306_1, buf720, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1306_1
        # Topologically Sorted Source Nodes: [x_1494], Original ATen: [aten.convolution]
        buf721 = extern_kernels.convolution(buf719, buf720, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf721, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf722 = buf721; del buf721  # reuse
        # Topologically Sorted Source Nodes: [x_1495, x_1496], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf722, arg1307_1, arg1308_1, arg1309_1, arg1310_1, 451584, grid=grid(451584), stream=stream0)
        del arg1307_1
        del arg1308_1
        del arg1309_1
        del arg1310_1
        buf723 = buf720; del buf720  # reuse
        # Topologically Sorted Source Nodes: [x_1495, x_1496, x_1497], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg1311_1, buf723, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1311_1
        # Topologically Sorted Source Nodes: [x_1495, x_1496, x_1497], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf724 = extern_kernels.convolution(buf722, buf723, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf724, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf722
        buf725 = buf719; del buf719  # reuse
        # Topologically Sorted Source Nodes: [x_1498, x_1499, x_1500], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf725, buf724, arg1312_1, arg1313_1, arg1314_1, arg1315_1, 451584, grid=grid(451584), stream=stream0)
        del arg1312_1
        del arg1313_1
        del arg1314_1
        del arg1315_1
        del buf724
        buf726 = buf723; del buf723  # reuse
        # Topologically Sorted Source Nodes: [x_1501], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1316_1, buf726, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1316_1
        # Topologically Sorted Source Nodes: [x_1501], Original ATen: [aten.convolution]
        buf727 = extern_kernels.convolution(buf725, buf726, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf727, (8, 18, 56, 56), (56448, 1, 1008, 18))
        buf728 = buf727; del buf727  # reuse
        # Topologically Sorted Source Nodes: [x_1502, x_1503], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf728, arg1317_1, arg1318_1, arg1319_1, arg1320_1, 451584, grid=grid(451584), stream=stream0)
        del arg1317_1
        del arg1318_1
        del arg1319_1
        del arg1320_1
        buf729 = buf726; del buf726  # reuse
        # Topologically Sorted Source Nodes: [x_1502, x_1503, x_1504], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg1321_1, buf729, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1321_1
        # Topologically Sorted Source Nodes: [x_1502, x_1503, x_1504], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf730 = extern_kernels.convolution(buf728, buf729, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf730, (8, 18, 56, 56), (56448, 1, 1008, 18))
        del buf728
        buf731 = buf725; del buf725  # reuse
        # Topologically Sorted Source Nodes: [x_1505, x_1506, x_1507], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf731, buf730, arg1322_1, arg1323_1, arg1324_1, arg1325_1, 451584, grid=grid(451584), stream=stream0)
        del arg1322_1
        del arg1323_1
        del arg1324_1
        del arg1325_1
        del buf730
        buf732 = buf729; del buf729  # reuse
        # Topologically Sorted Source Nodes: [input_440], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1496_1, buf732, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1496_1
        # Topologically Sorted Source Nodes: [input_440], Original ATen: [aten.convolution]
        buf733 = extern_kernels.convolution(buf731, buf732, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf733, (8, 18, 28, 28), (14112, 1, 504, 18))
        buf734 = buf733; del buf733  # reuse
        # Topologically Sorted Source Nodes: [input_441, input_442], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf734, arg1497_1, arg1498_1, arg1499_1, arg1500_1, 112896, grid=grid(112896), stream=stream0)
        del arg1497_1
        del arg1498_1
        del arg1499_1
        del arg1500_1
        buf735 = buf732; del buf732  # reuse
        # Topologically Sorted Source Nodes: [input_441, input_442, input_443], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg1501_1, buf735, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1501_1
        # Topologically Sorted Source Nodes: [input_441, input_442, input_443], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf736 = extern_kernels.convolution(buf734, buf735, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf736, (8, 18, 14, 14), (3528, 1, 252, 18))
        buf737 = buf736; del buf736  # reuse
        # Topologically Sorted Source Nodes: [input_444, input_445], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf737, arg1502_1, arg1503_1, arg1504_1, arg1505_1, 28224, grid=grid(28224), stream=stream0)
        del arg1502_1
        del arg1503_1
        del arg1504_1
        del arg1505_1
        buf738 = buf670; del buf670  # reuse
        # Topologically Sorted Source Nodes: [input_444, input_445, input_446], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41.run(arg1506_1, buf738, 2592, 9, grid=grid(2592, 9), stream=stream0)
        del arg1506_1
        # Topologically Sorted Source Nodes: [input_444, input_445, input_446], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf739 = extern_kernels.convolution(buf737, buf738, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf739, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf737
        buf747 = buf672; del buf672  # reuse
        # Topologically Sorted Source Nodes: [x_1508], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg1326_1, buf747, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1326_1
        # Topologically Sorted Source Nodes: [x_1508], Original ATen: [aten.convolution]
        buf748 = extern_kernels.convolution(buf746, buf747, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf748, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf749 = buf748; del buf748  # reuse
        # Topologically Sorted Source Nodes: [x_1509, x_1510], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf749, arg1327_1, arg1328_1, arg1329_1, arg1330_1, 225792, grid=grid(225792), stream=stream0)
        del arg1327_1
        del arg1328_1
        del arg1329_1
        del arg1330_1
        buf750 = buf747; del buf747  # reuse
        # Topologically Sorted Source Nodes: [x_1509, x_1510, x_1511], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg1331_1, buf750, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1331_1
        # Topologically Sorted Source Nodes: [x_1509, x_1510, x_1511], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf751 = extern_kernels.convolution(buf749, buf750, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf751, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf749
        buf752 = buf746; del buf746  # reuse
        # Topologically Sorted Source Nodes: [x_1512, x_1513, x_1514], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf752, buf751, arg1332_1, arg1333_1, arg1334_1, arg1335_1, 225792, grid=grid(225792), stream=stream0)
        del arg1332_1
        del arg1333_1
        del arg1334_1
        del arg1335_1
        del buf751
        buf753 = buf750; del buf750  # reuse
        # Topologically Sorted Source Nodes: [x_1515], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg1336_1, buf753, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1336_1
        # Topologically Sorted Source Nodes: [x_1515], Original ATen: [aten.convolution]
        buf754 = extern_kernels.convolution(buf752, buf753, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf754, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf755 = buf754; del buf754  # reuse
        # Topologically Sorted Source Nodes: [x_1516, x_1517], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf755, arg1337_1, arg1338_1, arg1339_1, arg1340_1, 225792, grid=grid(225792), stream=stream0)
        del arg1337_1
        del arg1338_1
        del arg1339_1
        del arg1340_1
        buf756 = buf753; del buf753  # reuse
        # Topologically Sorted Source Nodes: [x_1516, x_1517, x_1518], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg1341_1, buf756, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1341_1
        # Topologically Sorted Source Nodes: [x_1516, x_1517, x_1518], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf757 = extern_kernels.convolution(buf755, buf756, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf757, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf755
        buf758 = buf752; del buf752  # reuse
        # Topologically Sorted Source Nodes: [x_1519, x_1520, x_1521], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf758, buf757, arg1342_1, arg1343_1, arg1344_1, arg1345_1, 225792, grid=grid(225792), stream=stream0)
        del arg1342_1
        del arg1343_1
        del arg1344_1
        del arg1345_1
        del buf757
        buf759 = buf756; del buf756  # reuse
        # Topologically Sorted Source Nodes: [x_1522], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg1346_1, buf759, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1346_1
        # Topologically Sorted Source Nodes: [x_1522], Original ATen: [aten.convolution]
        buf760 = extern_kernels.convolution(buf758, buf759, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf760, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf761 = buf760; del buf760  # reuse
        # Topologically Sorted Source Nodes: [x_1523, x_1524], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf761, arg1347_1, arg1348_1, arg1349_1, arg1350_1, 225792, grid=grid(225792), stream=stream0)
        del arg1347_1
        del arg1348_1
        del arg1349_1
        del arg1350_1
        buf762 = buf759; del buf759  # reuse
        # Topologically Sorted Source Nodes: [x_1523, x_1524, x_1525], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg1351_1, buf762, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1351_1
        # Topologically Sorted Source Nodes: [x_1523, x_1524, x_1525], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf763 = extern_kernels.convolution(buf761, buf762, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf763, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf761
        buf764 = buf758; del buf758  # reuse
        # Topologically Sorted Source Nodes: [x_1526, x_1527, x_1528], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf764, buf763, arg1352_1, arg1353_1, arg1354_1, arg1355_1, 225792, grid=grid(225792), stream=stream0)
        del arg1352_1
        del arg1353_1
        del arg1354_1
        del arg1355_1
        del buf763
        buf765 = buf762; del buf762  # reuse
        # Topologically Sorted Source Nodes: [x_1529], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg1356_1, buf765, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1356_1
        # Topologically Sorted Source Nodes: [x_1529], Original ATen: [aten.convolution]
        buf766 = extern_kernels.convolution(buf764, buf765, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf766, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf767 = buf766; del buf766  # reuse
        # Topologically Sorted Source Nodes: [x_1530, x_1531], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf767, arg1357_1, arg1358_1, arg1359_1, arg1360_1, 225792, grid=grid(225792), stream=stream0)
        del arg1357_1
        del arg1358_1
        del arg1359_1
        del arg1360_1
        buf768 = buf765; del buf765  # reuse
        # Topologically Sorted Source Nodes: [x_1530, x_1531, x_1532], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_13.run(arg1361_1, buf768, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1361_1
        # Topologically Sorted Source Nodes: [x_1530, x_1531, x_1532], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf769 = extern_kernels.convolution(buf767, buf768, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf769, (8, 36, 28, 28), (28224, 1, 1008, 36))
        buf883 = buf740; del buf740  # reuse
        # Topologically Sorted Source Nodes: [input_422], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(arg1461_1, buf883, 648, 9, grid=grid(648, 9), stream=stream0)
        del arg1461_1
        # Topologically Sorted Source Nodes: [input_422], Original ATen: [aten.convolution]
        buf884 = extern_kernels.convolution(buf731, buf883, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf884, (8, 36, 28, 28), (28224, 1, 1008, 36))
        del buf883
        buf777 = buf735; del buf735  # reuse
        # Topologically Sorted Source Nodes: [input_388], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1236_1, buf777, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1236_1
        # Topologically Sorted Source Nodes: [input_388], Original ATen: [aten.convolution]
        buf778 = extern_kernels.convolution(buf593, buf777, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf778, (8, 18, 28, 28), (14112, 1, 504, 18))
        buf779 = buf778; del buf778  # reuse
        # Topologically Sorted Source Nodes: [input_389, input_390], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf779, arg1237_1, arg1238_1, arg1239_1, arg1240_1, 112896, grid=grid(112896), stream=stream0)
        del arg1237_1
        del arg1238_1
        del arg1239_1
        del arg1240_1
        buf780 = reinterpret_tensor(buf768, (72, 18, 3, 3), (162, 1, 54, 18), 0); del buf768  # reuse
        # Topologically Sorted Source Nodes: [input_389, input_390, input_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg1241_1, buf780, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1241_1
        # Topologically Sorted Source Nodes: [input_389, input_390, input_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf781 = extern_kernels.convolution(buf779, buf780, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf781, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf782 = reinterpret_tensor(buf738, (72, 36, 3, 3), (324, 1, 108, 36), 0); del buf738  # reuse
        # Topologically Sorted Source Nodes: [input_393], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg1246_1, buf782, 2592, 9, grid=grid(2592, 9), stream=stream0)
        del arg1246_1
        # Topologically Sorted Source Nodes: [input_393], Original ATen: [aten.convolution]
        buf783 = extern_kernels.convolution(buf624, buf782, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf783, (8, 72, 14, 14), (14112, 1, 1008, 72))
        # Topologically Sorted Source Nodes: [input_395], Original ATen: [aten.convolution]
        buf785 = extern_kernels.convolution(buf704, arg1251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf785, (8, 72, 7, 7), (3528, 1, 504, 72))
        del arg1251_1
        buf786 = reinterpret_tensor(buf779, (8, 72, 14, 14), (14112, 196, 14, 1), 0); del buf779  # reuse
        # Topologically Sorted Source Nodes: [input_396, input_397], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_38.run(buf785, arg1252_1, arg1253_1, arg1254_1, arg1255_1, buf786, 112896, grid=grid(112896), stream=stream0)
        del arg1252_1
        del arg1253_1
        del arg1254_1
        del arg1255_1
        del buf785
        buf784 = buf781; del buf781  # reuse
        buf787 = reinterpret_tensor(buf734, (8, 72, 14, 14), (14112, 1, 1008, 72), 0); del buf734  # reuse
        # Topologically Sorted Source Nodes: [input_392, input_394, y_109, y_110, y_111, shortcut_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39.run(buf784, arg1242_1, arg1243_1, arg1244_1, arg1245_1, buf783, arg1247_1, arg1248_1, arg1249_1, arg1250_1, buf661, buf786, buf787, 1568, 72, grid=grid(1568, 72), stream=stream0)
        del arg1242_1
        del arg1243_1
        del arg1244_1
        del arg1245_1
        del arg1247_1
        del arg1248_1
        del arg1249_1
        del arg1250_1
        del buf783
        del buf784
        del buf786
        buf788 = reinterpret_tensor(buf675, (72, 72, 3, 3), (648, 1, 216, 72), 0); del buf675  # reuse
        # Topologically Sorted Source Nodes: [x_1536], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg1366_1, buf788, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1366_1
        # Topologically Sorted Source Nodes: [x_1536], Original ATen: [aten.convolution]
        buf789 = extern_kernels.convolution(buf787, buf788, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf789, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf790 = buf789; del buf789  # reuse
        # Topologically Sorted Source Nodes: [x_1537, x_1538], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf790, arg1367_1, arg1368_1, arg1369_1, arg1370_1, 112896, grid=grid(112896), stream=stream0)
        del arg1367_1
        del arg1368_1
        del arg1369_1
        del arg1370_1
        buf791 = buf788; del buf788  # reuse
        # Topologically Sorted Source Nodes: [x_1537, x_1538, x_1539], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg1371_1, buf791, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1371_1
        # Topologically Sorted Source Nodes: [x_1537, x_1538, x_1539], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf792 = extern_kernels.convolution(buf790, buf791, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf792, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf790
        buf793 = buf787; del buf787  # reuse
        # Topologically Sorted Source Nodes: [x_1540, x_1541, x_1542], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf793, buf792, arg1372_1, arg1373_1, arg1374_1, arg1375_1, 112896, grid=grid(112896), stream=stream0)
        del arg1372_1
        del arg1373_1
        del arg1374_1
        del arg1375_1
        del buf792
        buf794 = buf791; del buf791  # reuse
        # Topologically Sorted Source Nodes: [x_1543], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg1376_1, buf794, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1376_1
        # Topologically Sorted Source Nodes: [x_1543], Original ATen: [aten.convolution]
        buf795 = extern_kernels.convolution(buf793, buf794, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf795, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf796 = buf795; del buf795  # reuse
        # Topologically Sorted Source Nodes: [x_1544, x_1545], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf796, arg1377_1, arg1378_1, arg1379_1, arg1380_1, 112896, grid=grid(112896), stream=stream0)
        del arg1377_1
        del arg1378_1
        del arg1379_1
        del arg1380_1
        buf797 = buf794; del buf794  # reuse
        # Topologically Sorted Source Nodes: [x_1544, x_1545, x_1546], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg1381_1, buf797, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1381_1
        # Topologically Sorted Source Nodes: [x_1544, x_1545, x_1546], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf798 = extern_kernels.convolution(buf796, buf797, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf798, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf796
        buf799 = buf793; del buf793  # reuse
        # Topologically Sorted Source Nodes: [x_1547, x_1548, x_1549], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf799, buf798, arg1382_1, arg1383_1, arg1384_1, arg1385_1, 112896, grid=grid(112896), stream=stream0)
        del arg1382_1
        del arg1383_1
        del arg1384_1
        del arg1385_1
        del buf798
        buf800 = buf797; del buf797  # reuse
        # Topologically Sorted Source Nodes: [x_1550], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg1386_1, buf800, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1386_1
        # Topologically Sorted Source Nodes: [x_1550], Original ATen: [aten.convolution]
        buf801 = extern_kernels.convolution(buf799, buf800, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf801, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf802 = buf801; del buf801  # reuse
        # Topologically Sorted Source Nodes: [x_1551, x_1552], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf802, arg1387_1, arg1388_1, arg1389_1, arg1390_1, 112896, grid=grid(112896), stream=stream0)
        del arg1387_1
        del arg1388_1
        del arg1389_1
        del arg1390_1
        buf803 = buf800; del buf800  # reuse
        # Topologically Sorted Source Nodes: [x_1551, x_1552, x_1553], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg1391_1, buf803, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1391_1
        # Topologically Sorted Source Nodes: [x_1551, x_1552, x_1553], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf804 = extern_kernels.convolution(buf802, buf803, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf804, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf802
        buf805 = buf799; del buf799  # reuse
        # Topologically Sorted Source Nodes: [x_1554, x_1555, x_1556], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf805, buf804, arg1392_1, arg1393_1, arg1394_1, arg1395_1, 112896, grid=grid(112896), stream=stream0)
        del arg1392_1
        del arg1393_1
        del arg1394_1
        del arg1395_1
        del buf804
        buf806 = buf803; del buf803  # reuse
        # Topologically Sorted Source Nodes: [x_1557], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg1396_1, buf806, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1396_1
        # Topologically Sorted Source Nodes: [x_1557], Original ATen: [aten.convolution]
        buf807 = extern_kernels.convolution(buf805, buf806, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf807, (8, 72, 14, 14), (14112, 1, 1008, 72))
        buf808 = buf807; del buf807  # reuse
        # Topologically Sorted Source Nodes: [x_1558, x_1559], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf808, arg1397_1, arg1398_1, arg1399_1, arg1400_1, 112896, grid=grid(112896), stream=stream0)
        del arg1397_1
        del arg1398_1
        del arg1399_1
        del arg1400_1
        buf809 = buf806; del buf806  # reuse
        # Topologically Sorted Source Nodes: [x_1558, x_1559, x_1560], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_22.run(arg1401_1, buf809, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1401_1
        # Topologically Sorted Source Nodes: [x_1558, x_1559, x_1560], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf810 = extern_kernels.convolution(buf808, buf809, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf810, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf808
        buf811 = buf805; del buf805  # reuse
        # Topologically Sorted Source Nodes: [x_1561, x_1562, x_1563], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf811, buf810, arg1402_1, arg1403_1, arg1404_1, arg1405_1, 112896, grid=grid(112896), stream=stream0)
        del arg1402_1
        del arg1403_1
        del arg1404_1
        del arg1405_1
        del buf810
        # Topologically Sorted Source Nodes: [input_424], Original ATen: [aten.convolution]
        buf885 = extern_kernels.convolution(buf811, arg1466_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf885, (8, 36, 14, 14), (7056, 1, 504, 36))
        del arg1466_1
        buf886 = reinterpret_tensor(buf767, (8, 36, 28, 28), (28224, 784, 28, 1), 0); del buf767  # reuse
        # Topologically Sorted Source Nodes: [input_425, input_426], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_24.run(buf885, arg1467_1, arg1468_1, arg1469_1, arg1470_1, buf886, 225792, grid=grid(225792), stream=stream0)
        del arg1467_1
        del arg1468_1
        del arg1469_1
        del arg1470_1
        del buf885
        buf814 = buf777; del buf777  # reuse
        # Topologically Sorted Source Nodes: [input_398], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1256_1, buf814, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1256_1
        # Topologically Sorted Source Nodes: [input_398], Original ATen: [aten.convolution]
        buf815 = extern_kernels.convolution(buf593, buf814, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf815, (8, 18, 28, 28), (14112, 1, 504, 18))
        del buf593
        buf816 = buf815; del buf815  # reuse
        # Topologically Sorted Source Nodes: [input_399, input_400], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf816, arg1257_1, arg1258_1, arg1259_1, arg1260_1, 112896, grid=grid(112896), stream=stream0)
        del arg1257_1
        del arg1258_1
        del arg1259_1
        del arg1260_1
        buf817 = buf814; del buf814  # reuse
        # Topologically Sorted Source Nodes: [input_399, input_400, input_401], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_9.run(arg1261_1, buf817, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1261_1
        # Topologically Sorted Source Nodes: [input_399, input_400, input_401], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf818 = extern_kernels.convolution(buf816, buf817, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf818, (8, 18, 14, 14), (3528, 1, 252, 18))
        del buf816
        buf819 = buf818; del buf818  # reuse
        # Topologically Sorted Source Nodes: [input_402, input_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf819, arg1262_1, arg1263_1, arg1264_1, arg1265_1, 28224, grid=grid(28224), stream=stream0)
        del arg1262_1
        del arg1263_1
        del arg1264_1
        del arg1265_1
        buf820 = reinterpret_tensor(buf782, (144, 18, 3, 3), (162, 1, 54, 18), 0); del buf782  # reuse
        # Topologically Sorted Source Nodes: [input_402, input_403, input_404], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41.run(arg1266_1, buf820, 2592, 9, grid=grid(2592, 9), stream=stream0)
        del arg1266_1
        # Topologically Sorted Source Nodes: [input_402, input_403, input_404], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf821 = extern_kernels.convolution(buf819, buf820, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf821, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf819
        buf822 = reinterpret_tensor(buf780, (36, 36, 3, 3), (324, 1, 108, 36), 0); del buf780  # reuse
        # Topologically Sorted Source Nodes: [input_406], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg1271_1, buf822, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1271_1
        # Topologically Sorted Source Nodes: [input_406], Original ATen: [aten.convolution]
        buf823 = extern_kernels.convolution(buf624, buf822, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf823, (8, 36, 14, 14), (7056, 1, 504, 36))
        buf824 = buf823; del buf823  # reuse
        # Topologically Sorted Source Nodes: [input_407, input_408], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_42.run(buf824, arg1272_1, arg1273_1, arg1274_1, arg1275_1, 56448, grid=grid(56448), stream=stream0)
        del arg1272_1
        del arg1273_1
        del arg1274_1
        del arg1275_1
        buf825 = reinterpret_tensor(buf809, (144, 36, 3, 3), (324, 1, 108, 36), 0); del buf809  # reuse
        # Topologically Sorted Source Nodes: [input_407, input_408, input_409], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_43.run(arg1276_1, buf825, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1276_1
        # Topologically Sorted Source Nodes: [input_407, input_408, input_409], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf826 = extern_kernels.convolution(buf824, buf825, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf826, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf824
        buf828 = buf678; del buf678  # reuse
        # Topologically Sorted Source Nodes: [input_411], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(arg1281_1, buf828, 10368, 9, grid=grid(10368, 9), stream=stream0)
        del arg1281_1
        # Topologically Sorted Source Nodes: [input_411], Original ATen: [aten.convolution]
        buf829 = extern_kernels.convolution(buf661, buf828, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf829, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf827 = buf821; del buf821  # reuse
        buf830 = buf704; del buf704  # reuse
        # Topologically Sorted Source Nodes: [input_405, input_410, y_112, input_412, y_113, y_114, shortcut_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf827, buf830, arg1267_1, arg1268_1, arg1269_1, arg1270_1, buf826, arg1277_1, arg1278_1, arg1279_1, arg1280_1, buf829, arg1282_1, arg1283_1, arg1284_1, arg1285_1, 56448, grid=grid(56448), stream=stream0)
        del arg1267_1
        del arg1268_1
        del arg1269_1
        del arg1270_1
        del arg1277_1
        del arg1278_1
        del arg1279_1
        del arg1280_1
        del arg1282_1
        del arg1283_1
        del arg1284_1
        del arg1285_1
        del buf826
        del buf827
        del buf829
        buf831 = buf702; del buf702  # reuse
        # Topologically Sorted Source Nodes: [x_1564], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(arg1406_1, buf831, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1406_1
        # Topologically Sorted Source Nodes: [x_1564], Original ATen: [aten.convolution]
        buf832 = extern_kernels.convolution(buf830, buf831, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf832, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf833 = buf832; del buf832  # reuse
        # Topologically Sorted Source Nodes: [x_1565, x_1566], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf833, arg1407_1, arg1408_1, arg1409_1, arg1410_1, 56448, grid=grid(56448), stream=stream0)
        del arg1407_1
        del arg1408_1
        del arg1409_1
        del arg1410_1
        buf834 = buf831; del buf831  # reuse
        # Topologically Sorted Source Nodes: [x_1565, x_1566, x_1567], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_33.run(arg1411_1, buf834, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1411_1
        # Topologically Sorted Source Nodes: [x_1565, x_1566, x_1567], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf835 = extern_kernels.convolution(buf833, buf834, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf835, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf833
        buf836 = buf830; del buf830  # reuse
        # Topologically Sorted Source Nodes: [x_1568, x_1569, x_1570], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf836, buf835, arg1412_1, arg1413_1, arg1414_1, arg1415_1, 56448, grid=grid(56448), stream=stream0)
        del arg1412_1
        del arg1413_1
        del arg1414_1
        del arg1415_1
        del buf835
        buf837 = buf834; del buf834  # reuse
        # Topologically Sorted Source Nodes: [x_1571], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(arg1416_1, buf837, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1416_1
        # Topologically Sorted Source Nodes: [x_1571], Original ATen: [aten.convolution]
        buf838 = extern_kernels.convolution(buf836, buf837, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf838, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf839 = buf838; del buf838  # reuse
        # Topologically Sorted Source Nodes: [x_1572, x_1573], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf839, arg1417_1, arg1418_1, arg1419_1, arg1420_1, 56448, grid=grid(56448), stream=stream0)
        del arg1417_1
        del arg1418_1
        del arg1419_1
        del arg1420_1
        buf840 = buf837; del buf837  # reuse
        # Topologically Sorted Source Nodes: [x_1572, x_1573, x_1574], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_33.run(arg1421_1, buf840, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1421_1
        # Topologically Sorted Source Nodes: [x_1572, x_1573, x_1574], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf841 = extern_kernels.convolution(buf839, buf840, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf841, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf839
        buf842 = buf836; del buf836  # reuse
        # Topologically Sorted Source Nodes: [x_1575, x_1576, x_1577], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf842, buf841, arg1422_1, arg1423_1, arg1424_1, arg1425_1, 56448, grid=grid(56448), stream=stream0)
        del arg1422_1
        del arg1423_1
        del arg1424_1
        del arg1425_1
        del buf841
        buf843 = buf840; del buf840  # reuse
        # Topologically Sorted Source Nodes: [x_1578], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(arg1426_1, buf843, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1426_1
        # Topologically Sorted Source Nodes: [x_1578], Original ATen: [aten.convolution]
        buf844 = extern_kernels.convolution(buf842, buf843, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf844, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf845 = buf844; del buf844  # reuse
        # Topologically Sorted Source Nodes: [x_1579, x_1580], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf845, arg1427_1, arg1428_1, arg1429_1, arg1430_1, 56448, grid=grid(56448), stream=stream0)
        del arg1427_1
        del arg1428_1
        del arg1429_1
        del arg1430_1
        buf846 = buf843; del buf843  # reuse
        # Topologically Sorted Source Nodes: [x_1579, x_1580, x_1581], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_33.run(arg1431_1, buf846, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1431_1
        # Topologically Sorted Source Nodes: [x_1579, x_1580, x_1581], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf847 = extern_kernels.convolution(buf845, buf846, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf847, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf845
        buf848 = buf842; del buf842  # reuse
        # Topologically Sorted Source Nodes: [x_1582, x_1583, x_1584], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf848, buf847, arg1432_1, arg1433_1, arg1434_1, arg1435_1, 56448, grid=grid(56448), stream=stream0)
        del arg1432_1
        del arg1433_1
        del arg1434_1
        del arg1435_1
        del buf847
        buf849 = buf846; del buf846  # reuse
        # Topologically Sorted Source Nodes: [x_1585], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(arg1436_1, buf849, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1436_1
        # Topologically Sorted Source Nodes: [x_1585], Original ATen: [aten.convolution]
        buf850 = extern_kernels.convolution(buf848, buf849, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf850, (8, 144, 7, 7), (7056, 1, 1008, 144))
        buf851 = buf850; del buf850  # reuse
        # Topologically Sorted Source Nodes: [x_1586, x_1587], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf851, arg1437_1, arg1438_1, arg1439_1, arg1440_1, 56448, grid=grid(56448), stream=stream0)
        del arg1437_1
        del arg1438_1
        del arg1439_1
        del arg1440_1
        buf852 = buf849; del buf849  # reuse
        # Topologically Sorted Source Nodes: [x_1586, x_1587, x_1588], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused_convolution_33.run(arg1441_1, buf852, 20736, 9, grid=grid(20736, 9), stream=stream0)
        del arg1441_1
        # Topologically Sorted Source Nodes: [x_1586, x_1587, x_1588], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf853 = extern_kernels.convolution(buf851, buf852, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf853, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf851
        del buf852
        buf854 = buf848; del buf848  # reuse
        # Topologically Sorted Source Nodes: [x_1589, x_1590, x_1591], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf854, buf853, arg1442_1, arg1443_1, arg1444_1, arg1445_1, 56448, grid=grid(56448), stream=stream0)
        del arg1442_1
        del arg1443_1
        del arg1444_1
        del arg1445_1
        del buf853
        # Topologically Sorted Source Nodes: [input_427], Original ATen: [aten.convolution]
        buf887 = extern_kernels.convolution(buf854, arg1471_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf887, (8, 36, 7, 7), (1764, 1, 252, 36))
        del arg1471_1
        buf888 = reinterpret_tensor(buf624, (8, 36, 28, 28), (28224, 784, 28, 1), 0); del buf624  # reuse
        # Topologically Sorted Source Nodes: [input_428, input_429], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_35.run(buf887, arg1472_1, arg1473_1, arg1474_1, arg1475_1, buf888, 225792, grid=grid(225792), stream=stream0)
        del arg1472_1
        del arg1473_1
        del arg1474_1
        del arg1475_1
        del buf887
        buf770 = buf764; del buf764  # reuse
        buf889 = buf884; del buf884  # reuse
        # Topologically Sorted Source Nodes: [x_1533, x_1534, x_1535, input_423, y_118, y_119, y_120, shortcut_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf770, buf889, buf769, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1462_1, arg1463_1, arg1464_1, arg1465_1, buf886, buf888, 6272, 36, grid=grid(6272, 36), stream=stream0)
        del arg1362_1
        del arg1363_1
        del arg1364_1
        del arg1365_1
        del arg1462_1
        del arg1463_1
        del arg1464_1
        del arg1465_1
        del buf769
        del buf886
        del buf888
        buf771 = buf822; del buf822  # reuse
        # Topologically Sorted Source Nodes: [input_448], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg1511_1, buf771, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1511_1
        # Topologically Sorted Source Nodes: [input_448], Original ATen: [aten.convolution]
        buf772 = extern_kernels.convolution(buf770, buf771, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf772, (8, 36, 14, 14), (7056, 1, 504, 36))
        buf773 = buf772; del buf772  # reuse
        # Topologically Sorted Source Nodes: [input_449, input_450], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_42.run(buf773, arg1512_1, arg1513_1, arg1514_1, arg1515_1, 56448, grid=grid(56448), stream=stream0)
        del arg1512_1
        del arg1513_1
        del arg1514_1
        del arg1515_1
        buf774 = buf825; del buf825  # reuse
        # Topologically Sorted Source Nodes: [input_449, input_450, input_451], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_43.run(arg1516_1, buf774, 5184, 9, grid=grid(5184, 9), stream=stream0)
        del arg1516_1
        # Topologically Sorted Source Nodes: [input_449, input_450, input_451], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf775 = extern_kernels.convolution(buf773, buf774, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf775, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf773
        del buf774
        buf812 = buf828; del buf828  # reuse
        # Topologically Sorted Source Nodes: [input_453], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(arg1521_1, buf812, 10368, 9, grid=grid(10368, 9), stream=stream0)
        del arg1521_1
        # Topologically Sorted Source Nodes: [input_453], Original ATen: [aten.convolution]
        buf813 = extern_kernels.convolution(buf811, buf812, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf813, (8, 144, 7, 7), (7056, 1, 1008, 144))
        del buf812
        buf776 = buf739; del buf739  # reuse
        buf855 = buf813; del buf813  # reuse
        # Topologically Sorted Source Nodes: [input_447, input_452, y_124, input_454, y_125, y_126, shortcut_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45.run(buf776, buf855, arg1507_1, arg1508_1, arg1509_1, arg1510_1, buf775, arg1517_1, arg1518_1, arg1519_1, arg1520_1, arg1522_1, arg1523_1, arg1524_1, arg1525_1, buf854, 56448, grid=grid(56448), stream=stream0)
        del arg1507_1
        del arg1508_1
        del arg1509_1
        del arg1510_1
        del arg1517_1
        del arg1518_1
        del arg1519_1
        del arg1520_1
        del arg1522_1
        del arg1523_1
        del arg1524_1
        del arg1525_1
        del buf775
        del buf776
        # Topologically Sorted Source Nodes: [x_1622], Original ATen: [aten.convolution]
        buf856 = extern_kernels.convolution(buf855, arg1598_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf856, (8, 256, 7, 7), (12544, 1, 1792, 256))
        del arg1598_1
        buf857 = buf856; del buf856  # reuse
        # Topologically Sorted Source Nodes: [x_1623, x_1624], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf857, arg1599_1, arg1600_1, arg1601_1, arg1602_1, 100352, grid=grid(100352), stream=stream0)
        del arg1599_1
        del arg1600_1
        del arg1601_1
        del arg1602_1
        buf858 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_1623, x_1624, x_1625], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47.run(arg1603_1, buf858, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg1603_1
        # Topologically Sorted Source Nodes: [x_1623, x_1624, x_1625], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf859 = extern_kernels.convolution(buf857, buf858, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf859, (8, 256, 7, 7), (12544, 1, 1792, 256))
        del buf857
        del buf858
        buf860 = buf859; del buf859  # reuse
        # Topologically Sorted Source Nodes: [x_1626, x_1627], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf860, arg1604_1, arg1605_1, arg1606_1, arg1607_1, 100352, grid=grid(100352), stream=stream0)
        del arg1604_1
        del arg1605_1
        del arg1606_1
        del arg1607_1
        # Topologically Sorted Source Nodes: [x_1626, x_1627, x_1628], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf861 = extern_kernels.convolution(buf860, arg1608_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf861, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
        del arg1608_1
        del buf860
        # Topologically Sorted Source Nodes: [input_467], Original ATen: [aten.convolution]
        buf862 = extern_kernels.convolution(buf855, arg1613_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf862, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
        del arg1613_1
        del buf855
        buf864 = buf817; del buf817  # reuse
        # Topologically Sorted Source Nodes: [input_430], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg1476_1, buf864, 324, 9, grid=grid(324, 9), stream=stream0)
        del arg1476_1
        # Topologically Sorted Source Nodes: [input_430], Original ATen: [aten.convolution]
        buf865 = extern_kernels.convolution(buf731, buf864, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf865, (8, 18, 28, 28), (14112, 1, 504, 18))
        del buf864
        buf866 = buf865; del buf865  # reuse
        # Topologically Sorted Source Nodes: [input_431, input_432], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf866, arg1477_1, arg1478_1, arg1479_1, arg1480_1, 112896, grid=grid(112896), stream=stream0)
        del arg1477_1
        del arg1478_1
        del arg1479_1
        del arg1480_1
        buf867 = reinterpret_tensor(buf771, (72, 18, 3, 3), (162, 1, 54, 18), 0); del buf771  # reuse
        # Topologically Sorted Source Nodes: [input_431, input_432, input_433], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg1481_1, buf867, 1296, 9, grid=grid(1296, 9), stream=stream0)
        del arg1481_1
        # Topologically Sorted Source Nodes: [input_431, input_432, input_433], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf868 = extern_kernels.convolution(buf866, buf867, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf868, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf867
        buf869 = reinterpret_tensor(buf820, (72, 36, 3, 3), (324, 1, 108, 36), 0); del buf820  # reuse
        # Topologically Sorted Source Nodes: [input_435], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg1486_1, buf869, 2592, 9, grid=grid(2592, 9), stream=stream0)
        del arg1486_1
        # Topologically Sorted Source Nodes: [input_435], Original ATen: [aten.convolution]
        buf870 = extern_kernels.convolution(buf770, buf869, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf870, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del buf869
        # Topologically Sorted Source Nodes: [input_437], Original ATen: [aten.convolution]
        buf872 = extern_kernels.convolution(buf854, arg1491_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf872, (8, 72, 7, 7), (3528, 1, 504, 72))
        del arg1491_1
        buf873 = reinterpret_tensor(buf866, (8, 72, 14, 14), (14112, 196, 14, 1), 0); del buf866  # reuse
        # Topologically Sorted Source Nodes: [input_438, input_439], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_38.run(buf872, arg1492_1, arg1493_1, arg1494_1, arg1495_1, buf873, 112896, grid=grid(112896), stream=stream0)
        del arg1492_1
        del arg1493_1
        del arg1494_1
        del arg1495_1
        del buf872
        buf871 = buf868; del buf868  # reuse
        buf874 = buf661; del buf661  # reuse
        # Topologically Sorted Source Nodes: [input_434, input_436, y_121, y_122, y_123, shortcut_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39.run(buf871, arg1482_1, arg1483_1, arg1484_1, arg1485_1, buf870, arg1487_1, arg1488_1, arg1489_1, arg1490_1, buf811, buf873, buf874, 1568, 72, grid=grid(1568, 72), stream=stream0)
        del arg1482_1
        del arg1483_1
        del arg1484_1
        del arg1485_1
        del arg1487_1
        del arg1488_1
        del arg1489_1
        del arg1490_1
        del buf870
        del buf871
        del buf873
        # Topologically Sorted Source Nodes: [x_1612], Original ATen: [aten.convolution]
        buf875 = extern_kernels.convolution(buf874, arg1572_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf875, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg1572_1
        buf876 = buf875; del buf875  # reuse
        # Topologically Sorted Source Nodes: [x_1613, x_1614], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf876, arg1573_1, arg1574_1, arg1575_1, arg1576_1, 200704, grid=grid(200704), stream=stream0)
        del arg1573_1
        del arg1574_1
        del arg1575_1
        del arg1576_1
        buf877 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_1613, x_1614, x_1615], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49.run(arg1577_1, buf877, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg1577_1
        # Topologically Sorted Source Nodes: [x_1613, x_1614, x_1615], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf878 = extern_kernels.convolution(buf876, buf877, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf878, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del buf876
        del buf877
        buf879 = buf878; del buf878  # reuse
        # Topologically Sorted Source Nodes: [x_1616, x_1617], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf879, arg1578_1, arg1579_1, arg1580_1, arg1581_1, 200704, grid=grid(200704), stream=stream0)
        del arg1578_1
        del arg1579_1
        del arg1580_1
        del arg1581_1
        # Topologically Sorted Source Nodes: [x_1616, x_1617, x_1618], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf880 = extern_kernels.convolution(buf879, arg1582_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf880, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg1582_1
        del buf879
        # Topologically Sorted Source Nodes: [input_462], Original ATen: [aten.convolution]
        buf881 = extern_kernels.convolution(buf874, arg1587_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf881, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg1587_1
        del buf874
        # Topologically Sorted Source Nodes: [x_1602], Original ATen: [aten.convolution]
        buf890 = extern_kernels.convolution(buf889, arg1546_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf890, (8, 64, 28, 28), (50176, 1, 1792, 64))
        del arg1546_1
        buf891 = buf890; del buf890  # reuse
        # Topologically Sorted Source Nodes: [x_1603, x_1604], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf891, arg1547_1, arg1548_1, arg1549_1, arg1550_1, 401408, grid=grid(401408), stream=stream0)
        del arg1547_1
        del arg1548_1
        del arg1549_1
        del arg1550_1
        buf892 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_1603, x_1604, x_1605], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg1551_1, buf892, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg1551_1
        # Topologically Sorted Source Nodes: [x_1603, x_1604, x_1605], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf893 = extern_kernels.convolution(buf891, buf892, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf893, (8, 64, 28, 28), (50176, 1, 1792, 64))
        del buf891
        del buf892
        buf894 = buf893; del buf893  # reuse
        # Topologically Sorted Source Nodes: [x_1606, x_1607], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf894, arg1552_1, arg1553_1, arg1554_1, arg1555_1, 401408, grid=grid(401408), stream=stream0)
        del arg1552_1
        del arg1553_1
        del arg1554_1
        del arg1555_1
        # Topologically Sorted Source Nodes: [x_1606, x_1607, x_1608], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf895 = extern_kernels.convolution(buf894, arg1556_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf895, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg1556_1
        del buf894
        # Topologically Sorted Source Nodes: [input_457], Original ATen: [aten.convolution]
        buf896 = extern_kernels.convolution(buf889, arg1561_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf896, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg1561_1
        del buf889
        # Topologically Sorted Source Nodes: [input_413], Original ATen: [aten.convolution]
        buf898 = extern_kernels.convolution(buf770, arg1446_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf898, (8, 18, 28, 28), (14112, 1, 504, 18))
        del arg1446_1
        del buf770
        # Topologically Sorted Source Nodes: [input_416], Original ATen: [aten.convolution]
        buf900 = extern_kernels.convolution(buf811, arg1451_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf900, (8, 18, 14, 14), (3528, 1, 252, 18))
        del arg1451_1
        del buf811
        # Topologically Sorted Source Nodes: [input_419], Original ATen: [aten.convolution]
        buf902 = extern_kernels.convolution(buf854, arg1456_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf902, (8, 18, 7, 7), (882, 1, 126, 18))
        del arg1456_1
        del buf854
        buf904 = buf731; del buf731  # reuse
        # Topologically Sorted Source Nodes: [input_414, input_415, y_115, input_417, input_418, y_116, input_420, input_421, y_117, shortcut_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_51.run(buf904, buf898, arg1447_1, arg1448_1, arg1449_1, arg1450_1, buf900, arg1452_1, arg1453_1, arg1454_1, arg1455_1, buf902, arg1457_1, arg1458_1, arg1459_1, arg1460_1, 144, 3136, grid=grid(144, 3136), stream=stream0)
        del arg1447_1
        del arg1448_1
        del arg1449_1
        del arg1450_1
        del arg1452_1
        del arg1453_1
        del arg1454_1
        del arg1455_1
        del arg1457_1
        del arg1458_1
        del arg1459_1
        del arg1460_1
        del buf898
        del buf900
        del buf902
        # Topologically Sorted Source Nodes: [x_1592], Original ATen: [aten.convolution]
        buf905 = extern_kernels.convolution(buf904, arg1526_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf905, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del arg1526_1
        buf906 = buf905; del buf905  # reuse
        # Topologically Sorted Source Nodes: [x_1593, x_1594], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf906, arg1527_1, arg1528_1, arg1529_1, arg1530_1, 802816, grid=grid(802816), stream=stream0)
        del arg1527_1
        del arg1528_1
        del arg1529_1
        del arg1530_1
        buf907 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_1593, x_1594, x_1595], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53.run(arg1531_1, buf907, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del arg1531_1
        # Topologically Sorted Source Nodes: [x_1593, x_1594, x_1595], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf908 = extern_kernels.convolution(buf906, buf907, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf908, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del buf906
        del buf907
        buf909 = buf908; del buf908  # reuse
        # Topologically Sorted Source Nodes: [x_1596, x_1597], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf909, arg1532_1, arg1533_1, arg1534_1, arg1535_1, 802816, grid=grid(802816), stream=stream0)
        del arg1532_1
        del arg1533_1
        del arg1534_1
        del arg1535_1
        # Topologically Sorted Source Nodes: [x_1596, x_1597, x_1598], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf910 = extern_kernels.convolution(buf909, arg1536_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf910, (8, 128, 56, 56), (401408, 1, 7168, 128))
        del arg1536_1
        del buf909
        # Topologically Sorted Source Nodes: [input_455], Original ATen: [aten.convolution]
        buf911 = extern_kernels.convolution(buf904, arg1541_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf911, (8, 128, 56, 56), (401408, 1, 7168, 128))
        del arg1541_1
        del buf904
        buf912 = buf910; del buf910  # reuse
        buf913 = empty_strided_cuda((8, 128, 56, 56), (401408, 1, 7168, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_1599, input_456, x_1600, x_1601], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_54.run(buf912, arg1537_1, arg1538_1, arg1539_1, arg1540_1, buf911, arg1542_1, arg1543_1, arg1544_1, arg1545_1, buf913, 3211264, grid=grid(3211264), stream=stream0)
        del arg1537_1
        del arg1538_1
        del arg1539_1
        del arg1540_1
        del arg1542_1
        del arg1543_1
        del arg1544_1
        del arg1545_1
        del buf911
        del buf912
        buf914 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_1601, input_459], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_55.run(arg1566_1, buf914, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg1566_1
        # Topologically Sorted Source Nodes: [x_1601, input_459], Original ATen: [aten.relu, aten.convolution]
        buf915 = extern_kernels.convolution(buf913, buf914, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf915, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del buf913
        del buf914
        buf897 = buf895; del buf895  # reuse
        buf916 = buf915; del buf915  # reuse
        # Topologically Sorted Source Nodes: [x_1609, input_458, x_1610, x_1611, x_1601, input_459, input_460, input_461, y_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_56.run(buf897, buf916, arg1557_1, arg1558_1, arg1559_1, arg1560_1, buf896, arg1562_1, arg1563_1, arg1564_1, arg1565_1, arg1567_1, arg1568_1, arg1569_1, arg1570_1, arg1571_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg1557_1
        del arg1558_1
        del arg1559_1
        del arg1560_1
        del arg1562_1
        del arg1563_1
        del arg1564_1
        del arg1565_1
        del arg1567_1
        del arg1568_1
        del arg1569_1
        del arg1570_1
        del arg1571_1
        del buf896
        del buf897
        buf917 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_1611, x_1601, input_459, input_460, input_461, y_127, input_464], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_57.run(arg1592_1, buf917, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg1592_1
        # Topologically Sorted Source Nodes: [x_1611, x_1601, input_459, input_460, input_461, y_127, input_464], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        buf918 = extern_kernels.convolution(buf916, buf917, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf918, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del buf916
        del buf917
        buf882 = buf880; del buf880  # reuse
        buf919 = buf918; del buf918  # reuse
        # Topologically Sorted Source Nodes: [x_1619, input_463, x_1620, x_1621, x_1611, x_1601, input_459, input_460, input_461, y_127, input_464, input_465, input_466, y_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_58.run(buf882, buf919, arg1583_1, arg1584_1, arg1585_1, arg1586_1, buf881, arg1588_1, arg1589_1, arg1590_1, arg1591_1, arg1593_1, arg1594_1, arg1595_1, arg1596_1, arg1597_1, 802816, grid=grid(802816), stream=stream0)
        del arg1583_1
        del arg1584_1
        del arg1585_1
        del arg1586_1
        del arg1588_1
        del arg1589_1
        del arg1590_1
        del arg1591_1
        del arg1593_1
        del arg1594_1
        del arg1595_1
        del arg1596_1
        del arg1597_1
        del buf881
        del buf882
        buf920 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_1621, x_1611, x_1601, input_459, input_460, input_461, y_127, input_464, input_465, input_466, y_128, input_469], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_59.run(arg1618_1, buf920, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del arg1618_1
        # Topologically Sorted Source Nodes: [x_1621, x_1611, x_1601, input_459, input_460, input_461, y_127, input_464, input_465, input_466, y_128, input_469], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        buf921 = extern_kernels.convolution(buf919, buf920, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf921, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
        del buf919
        del buf920
        buf863 = buf861; del buf861  # reuse
        buf922 = buf921; del buf921  # reuse
        # Topologically Sorted Source Nodes: [x_1629, input_468, x_1630, x_1631, x_1621, x_1611, x_1601, input_459, input_460, input_461, y_127, input_464, input_465, input_466, y_128, input_469, input_470, input_471, y_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_60.run(buf863, buf922, arg1609_1, arg1610_1, arg1611_1, arg1612_1, buf862, arg1614_1, arg1615_1, arg1616_1, arg1617_1, arg1619_1, arg1620_1, arg1621_1, arg1622_1, arg1623_1, 401408, grid=grid(401408), stream=stream0)
        del arg1609_1
        del arg1610_1
        del arg1611_1
        del arg1612_1
        del arg1614_1
        del arg1615_1
        del arg1616_1
        del arg1617_1
        del arg1619_1
        del arg1620_1
        del arg1621_1
        del arg1622_1
        del arg1623_1
        del buf862
        del buf863
        # Topologically Sorted Source Nodes: [x_1631, x_1621, x_1611, x_1601, input_459, input_460, input_461, y_127, input_464, input_465, input_466, y_128, input_469, input_470, input_471, y_129, input_472], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        buf923 = extern_kernels.convolution(buf922, arg1624_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf923, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg1624_1
        del buf922
        buf925 = empty_strided_cuda((8, 2048, 1, 1), (2048, 1, 16384, 16384), torch.float32)
        # Topologically Sorted Source Nodes: [x_1631, x_1621, x_1611, x_1601, input_459, input_460, input_461, y_127, input_464, input_465, input_466, y_128, input_469, input_470, input_471, y_129, input_472, input_473, input_474, x_1632], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_61.run(buf923, arg1625_1, arg1626_1, arg1627_1, arg1628_1, arg1629_1, buf925, 16384, 49, grid=grid(16384), stream=stream0)
        del arg1625_1
        del arg1626_1
        del arg1627_1
        del arg1628_1
        del arg1629_1
        del buf923
        buf926 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1635], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg1631_1, reinterpret_tensor(buf925, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg1630_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf926)
        del arg1630_1
        del arg1631_1
        del buf925
    return (buf926, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((18, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((36, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg710_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg713_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg716_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg719_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg722_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg725_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg728_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg731_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg734_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg737_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg740_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg743_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg746_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg749_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg752_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg755_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg758_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg761_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg764_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg767_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg770_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg773_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg776_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg779_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg782_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg785_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg788_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg791_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg794_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg797_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg798_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg799_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg800_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg801_1 = rand_strided((144, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg802_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg803_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg804_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg805_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg806_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg807_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg808_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg809_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg810_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg811_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg812_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg813_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg814_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg815_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg816_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg817_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg818_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg819_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg820_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg821_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg822_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg823_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg824_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg825_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg826_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg827_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg828_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg829_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg830_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg831_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg832_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg833_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg834_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg835_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg836_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg837_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg838_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg839_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg840_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg841_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg842_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg843_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg844_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg845_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg846_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg847_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg848_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg849_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg850_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg851_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg852_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg853_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg854_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg855_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg856_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg857_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg858_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg859_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg860_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg861_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg862_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg863_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg864_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg865_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg866_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg867_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg868_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg869_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg870_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg871_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg872_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg873_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg874_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg875_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg876_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg877_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg878_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg879_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg880_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg881_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg882_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg883_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg884_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg885_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg886_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg887_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg888_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg889_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg890_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg891_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg892_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg893_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg894_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg895_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg896_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg897_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg898_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg899_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg900_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg901_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg902_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg903_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg904_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg905_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg906_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg907_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg908_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg909_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg910_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg911_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg912_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg913_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg914_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg915_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg916_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg917_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg918_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg919_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg920_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg921_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg922_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg923_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg924_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg925_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg926_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg927_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg928_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg929_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg930_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg931_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg932_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg933_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg934_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg935_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg936_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg937_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg938_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg939_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg940_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg941_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg942_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg943_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg944_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg945_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg946_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg947_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg948_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg949_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg950_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg951_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg952_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg953_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg954_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg955_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg956_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg957_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg958_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg959_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg960_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg961_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg962_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg963_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg964_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg965_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg966_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg967_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg968_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg969_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg970_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg971_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg972_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg973_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg974_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg975_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg976_1 = rand_strided((18, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg977_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg978_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg979_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg980_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg981_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg982_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg983_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg984_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg985_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg986_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg987_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg988_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg989_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg990_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg991_1 = rand_strided((36, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg992_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg993_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg994_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg995_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg996_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg997_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg998_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg999_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1000_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1001_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1002_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1003_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1004_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1005_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1006_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1007_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1008_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1009_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1010_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1011_1 = rand_strided((72, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1012_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1013_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1014_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1015_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1016_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1017_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1018_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1019_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1020_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1021_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1022_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1023_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1024_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1025_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1026_1 = rand_strided((144, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1027_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1028_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1029_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1030_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1031_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1032_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1033_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1034_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1035_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1036_1 = rand_strided((144, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1037_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1038_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1039_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1040_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1041_1 = rand_strided((144, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1042_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1043_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1044_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1045_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1046_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1047_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1048_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1049_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1050_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1051_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1052_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1053_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1054_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1055_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1056_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1057_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1058_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1059_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1060_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1061_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1062_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1063_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1064_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1065_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1066_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1067_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1068_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1069_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1070_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1071_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1072_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1073_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1074_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1075_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1076_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1077_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1078_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1079_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1080_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1081_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1082_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1083_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1084_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1085_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1086_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1087_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1088_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1089_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1090_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1091_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1092_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1093_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1094_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1095_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1096_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1097_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1098_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1099_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1100_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1101_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1102_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1103_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1104_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1105_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1106_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1107_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1108_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1109_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1110_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1111_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1112_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1113_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1114_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1115_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1116_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1117_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1118_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1119_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1120_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1121_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1122_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1123_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1124_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1125_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1126_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1127_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1128_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1129_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1130_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1131_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1132_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1133_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1134_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1135_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1136_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1137_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1138_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1139_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1140_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1141_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1142_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1143_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1144_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1145_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1146_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1147_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1148_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1149_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1150_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1151_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1152_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1153_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1154_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1155_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1156_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1157_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1158_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1159_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1160_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1161_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1162_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1163_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1164_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1165_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1166_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1167_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1168_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1169_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1170_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1171_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1172_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1173_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1174_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1175_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1176_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1177_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1178_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1179_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1180_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1181_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1182_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1183_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1184_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1185_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1186_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1187_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1188_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1189_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1190_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1191_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1192_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1193_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1194_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1195_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1196_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1197_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1198_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1199_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1200_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1201_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1202_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1203_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1204_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1205_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1206_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1207_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1208_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1209_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1210_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1211_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1212_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1213_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1214_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1215_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1216_1 = rand_strided((18, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1217_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1218_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1219_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1220_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1221_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1222_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1223_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1224_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1225_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1226_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1227_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1228_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1229_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1230_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1231_1 = rand_strided((36, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1232_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1233_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1234_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1235_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1236_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1237_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1238_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1239_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1240_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1241_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1242_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1243_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1244_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1245_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1246_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1247_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1248_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1249_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1250_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1251_1 = rand_strided((72, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1252_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1253_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1254_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1255_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1256_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1257_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1258_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1259_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1260_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1261_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1262_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1263_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1264_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1265_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1266_1 = rand_strided((144, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1267_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1268_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1269_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1270_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1271_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1272_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1273_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1274_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1275_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1276_1 = rand_strided((144, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1277_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1278_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1279_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1280_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1281_1 = rand_strided((144, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1282_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1283_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1284_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1285_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1286_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1287_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1288_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1289_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1290_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1291_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1292_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1293_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1294_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1295_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1296_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1297_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1298_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1299_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1300_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1301_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1302_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1303_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1304_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1305_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1306_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1307_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1308_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1309_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1310_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1311_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1312_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1313_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1314_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1315_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1316_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1317_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1318_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1319_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1320_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1321_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1322_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1323_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1324_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1325_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1326_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1327_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1328_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1329_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1330_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1331_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1332_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1333_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1334_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1335_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1336_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1337_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1338_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1339_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1340_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1341_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1342_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1343_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1344_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1345_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1346_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1347_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1348_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1349_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1350_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1351_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1352_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1353_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1354_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1355_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1356_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1357_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1358_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1359_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1360_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1361_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1362_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1363_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1364_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1365_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1366_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1367_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1368_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1369_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1370_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1371_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1372_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1373_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1374_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1375_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1376_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1377_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1378_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1379_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1380_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1381_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1382_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1383_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1384_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1385_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1386_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1387_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1388_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1389_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1390_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1391_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1392_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1393_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1394_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1395_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1396_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1397_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1398_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1399_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1400_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1401_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1402_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1403_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1404_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1405_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1406_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1407_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1408_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1409_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1410_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1411_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1412_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1413_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1414_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1415_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1416_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1417_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1418_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1419_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1420_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1421_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1422_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1423_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1424_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1425_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1426_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1427_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1428_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1429_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1430_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1431_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1432_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1433_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1434_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1435_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1436_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1437_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1438_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1439_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1440_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1441_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1442_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1443_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1444_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1445_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1446_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1447_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1448_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1449_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1450_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1451_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1452_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1453_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1454_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1455_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1456_1 = rand_strided((18, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1457_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1458_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1459_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1460_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1461_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1462_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1463_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1464_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1465_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1466_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1467_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1468_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1469_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1470_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1471_1 = rand_strided((36, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1472_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1473_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1474_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1475_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1476_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1477_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1478_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1479_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1480_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1481_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1482_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1483_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1484_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1485_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1486_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1487_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1488_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1489_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1490_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1491_1 = rand_strided((72, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1492_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1493_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1494_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1495_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1496_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1497_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1498_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1499_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1500_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1501_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1502_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1503_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1504_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1505_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1506_1 = rand_strided((144, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1507_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1508_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1509_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1510_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1511_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1512_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1513_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1514_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1515_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1516_1 = rand_strided((144, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1517_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1518_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1519_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1520_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1521_1 = rand_strided((144, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1522_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1523_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1524_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1525_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1526_1 = rand_strided((32, 18, 1, 1), (18, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1527_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1528_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1529_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1530_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1531_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1532_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1533_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1534_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1535_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1536_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1537_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1538_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1539_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1540_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1541_1 = rand_strided((128, 18, 1, 1), (18, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1542_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1543_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1544_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1545_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1546_1 = rand_strided((64, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1547_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1548_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1549_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1550_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1551_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1552_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1553_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1554_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1555_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1556_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1557_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1558_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1559_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1560_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1561_1 = rand_strided((256, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1562_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1563_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1564_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1565_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1566_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1567_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1568_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1569_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1570_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1571_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1572_1 = rand_strided((128, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1573_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1574_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1575_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1576_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1577_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1578_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1579_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1580_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1581_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1582_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1583_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1584_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1585_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1586_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1587_1 = rand_strided((512, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1588_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1589_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1590_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1591_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1592_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1593_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1594_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1595_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1596_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1597_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1598_1 = rand_strided((256, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1599_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1600_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1601_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1602_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1603_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1604_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1605_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1606_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1607_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1608_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1609_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1610_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1611_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1612_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1613_1 = rand_strided((1024, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1614_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1615_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1616_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1617_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1618_1 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1619_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1620_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1621_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1622_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1623_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1624_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1625_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1626_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1627_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1628_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1629_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1630_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg1631_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1, arg1180_1, arg1181_1, arg1182_1, arg1183_1, arg1184_1, arg1185_1, arg1186_1, arg1187_1, arg1188_1, arg1189_1, arg1190_1, arg1191_1, arg1192_1, arg1193_1, arg1194_1, arg1195_1, arg1196_1, arg1197_1, arg1198_1, arg1199_1, arg1200_1, arg1201_1, arg1202_1, arg1203_1, arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1234_1, arg1235_1, arg1236_1, arg1237_1, arg1238_1, arg1239_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1276_1, arg1277_1, arg1278_1, arg1279_1, arg1280_1, arg1281_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1318_1, arg1319_1, arg1320_1, arg1321_1, arg1322_1, arg1323_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1360_1, arg1361_1, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1, arg1380_1, arg1381_1, arg1382_1, arg1383_1, arg1384_1, arg1385_1, arg1386_1, arg1387_1, arg1388_1, arg1389_1, arg1390_1, arg1391_1, arg1392_1, arg1393_1, arg1394_1, arg1395_1, arg1396_1, arg1397_1, arg1398_1, arg1399_1, arg1400_1, arg1401_1, arg1402_1, arg1403_1, arg1404_1, arg1405_1, arg1406_1, arg1407_1, arg1408_1, arg1409_1, arg1410_1, arg1411_1, arg1412_1, arg1413_1, arg1414_1, arg1415_1, arg1416_1, arg1417_1, arg1418_1, arg1419_1, arg1420_1, arg1421_1, arg1422_1, arg1423_1, arg1424_1, arg1425_1, arg1426_1, arg1427_1, arg1428_1, arg1429_1, arg1430_1, arg1431_1, arg1432_1, arg1433_1, arg1434_1, arg1435_1, arg1436_1, arg1437_1, arg1438_1, arg1439_1, arg1440_1, arg1441_1, arg1442_1, arg1443_1, arg1444_1, arg1445_1, arg1446_1, arg1447_1, arg1448_1, arg1449_1, arg1450_1, arg1451_1, arg1452_1, arg1453_1, arg1454_1, arg1455_1, arg1456_1, arg1457_1, arg1458_1, arg1459_1, arg1460_1, arg1461_1, arg1462_1, arg1463_1, arg1464_1, arg1465_1, arg1466_1, arg1467_1, arg1468_1, arg1469_1, arg1470_1, arg1471_1, arg1472_1, arg1473_1, arg1474_1, arg1475_1, arg1476_1, arg1477_1, arg1478_1, arg1479_1, arg1480_1, arg1481_1, arg1482_1, arg1483_1, arg1484_1, arg1485_1, arg1486_1, arg1487_1, arg1488_1, arg1489_1, arg1490_1, arg1491_1, arg1492_1, arg1493_1, arg1494_1, arg1495_1, arg1496_1, arg1497_1, arg1498_1, arg1499_1, arg1500_1, arg1501_1, arg1502_1, arg1503_1, arg1504_1, arg1505_1, arg1506_1, arg1507_1, arg1508_1, arg1509_1, arg1510_1, arg1511_1, arg1512_1, arg1513_1, arg1514_1, arg1515_1, arg1516_1, arg1517_1, arg1518_1, arg1519_1, arg1520_1, arg1521_1, arg1522_1, arg1523_1, arg1524_1, arg1525_1, arg1526_1, arg1527_1, arg1528_1, arg1529_1, arg1530_1, arg1531_1, arg1532_1, arg1533_1, arg1534_1, arg1535_1, arg1536_1, arg1537_1, arg1538_1, arg1539_1, arg1540_1, arg1541_1, arg1542_1, arg1543_1, arg1544_1, arg1545_1, arg1546_1, arg1547_1, arg1548_1, arg1549_1, arg1550_1, arg1551_1, arg1552_1, arg1553_1, arg1554_1, arg1555_1, arg1556_1, arg1557_1, arg1558_1, arg1559_1, arg1560_1, arg1561_1, arg1562_1, arg1563_1, arg1564_1, arg1565_1, arg1566_1, arg1567_1, arg1568_1, arg1569_1, arg1570_1, arg1571_1, arg1572_1, arg1573_1, arg1574_1, arg1575_1, arg1576_1, arg1577_1, arg1578_1, arg1579_1, arg1580_1, arg1581_1, arg1582_1, arg1583_1, arg1584_1, arg1585_1, arg1586_1, arg1587_1, arg1588_1, arg1589_1, arg1590_1, arg1591_1, arg1592_1, arg1593_1, arg1594_1, arg1595_1, arg1596_1, arg1597_1, arg1598_1, arg1599_1, arg1600_1, arg1601_1, arg1602_1, arg1603_1, arg1604_1, arg1605_1, arg1606_1, arg1607_1, arg1608_1, arg1609_1, arg1610_1, arg1611_1, arg1612_1, arg1613_1, arg1614_1, arg1615_1, arg1616_1, arg1617_1, arg1618_1, arg1619_1, arg1620_1, arg1621_1, arg1622_1, arg1623_1, arg1624_1, arg1625_1, arg1626_1, arg1627_1, arg1628_1, arg1629_1, arg1630_1, arg1631_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hrnet_w18', benchmark_compiled_module)
