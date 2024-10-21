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
# Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_18 => convolution_106
# Graph fragment:
#   %convolution_106 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/a3/ca3fkscaxwe76vfv26fyvvcg33mhfkdp227jhxoeldy4jjwm4yyt.py
# Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_18 => convolution_106
# Graph fragment:
#   %convolution_106 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[64, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
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


# kernel path: /tmp/torchinductor_sahanp/g2/cg2xfj7xgayguczsdoziufjalihswbizwk62inlhvukgi5i4fbce.py
# Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_19 => add_253, mul_316, mul_317, sub_105
#   input_20 => relu_101
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_106, %unsqueeze_841), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_316, %unsqueeze_845), kwargs = {})
#   %add_253 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_317, %unsqueeze_847), kwargs = {})
#   %relu_101 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_253,), kwargs = {})
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bf/cbfxcw7ntlfgzhk5rq34ywqcf2yqtxrfu7dtfoscmxunm2rbioqm.py
# Topologically Sorted Source Nodes: [input_19, input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_19 => add_253, mul_316, mul_317, sub_105
#   input_20 => relu_101
#   input_21 => convolution_107
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_106, %unsqueeze_841), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_316, %unsqueeze_845), kwargs = {})
#   %add_253 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_317, %unsqueeze_847), kwargs = {})
#   %relu_101 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_253,), kwargs = {})
#   %convolution_107 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_101, %arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
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


# kernel path: /tmp/torchinductor_sahanp/sj/csjlsf6cildosaufpdemqjmfwjtuhshoh4stl4umj6c35zaxm5pl.py
# Topologically Sorted Source Nodes: [input_22, input_23, input_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_22 => add_255, mul_319, mul_320, sub_106
#   input_23 => relu_102
#   input_24 => convolution_108
# Graph fragment:
#   %sub_106 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_107, %unsqueeze_849), kwargs = {})
#   %mul_319 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_106, %unsqueeze_851), kwargs = {})
#   %mul_320 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_319, %unsqueeze_853), kwargs = {})
#   %add_255 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_320, %unsqueeze_855), kwargs = {})
#   %relu_102 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_255,), kwargs = {})
#   %convolution_108 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_102, %arg11_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
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


# kernel path: /tmp/torchinductor_sahanp/mk/cmk5ld5vtpajybmnqphqudnvawpy7fiutr3h6iir6ntlmv6j62dj.py
# Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_25 => add_257, mul_322, mul_323, sub_107
#   input_26 => relu_103
# Graph fragment:
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_108, %unsqueeze_857), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %unsqueeze_859), kwargs = {})
#   %mul_323 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_322, %unsqueeze_861), kwargs = {})
#   %add_257 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_323, %unsqueeze_863), kwargs = {})
#   %relu_103 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_257,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/oz/cozt6cl2mgfa5fegnph6u3eu3ckbdg26ueotlki5mzqixgbwj2ct.py
# Topologically Sorted Source Nodes: [out_281, out_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_281 => add_261, mul_328, mul_329, sub_109
#   out_282 => relu_104
# Graph fragment:
#   %sub_109 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_110, %unsqueeze_873), kwargs = {})
#   %mul_328 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_109, %unsqueeze_875), kwargs = {})
#   %mul_329 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_328, %unsqueeze_877), kwargs = {})
#   %add_261 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_329, %unsqueeze_879), kwargs = {})
#   %relu_104 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_261,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/mz/cmzrwymenmdtyzw3yrvahkpzifh2vo7evtk2k3k3urk5mrjxaypm.py
# Topologically Sorted Source Nodes: [out_281, out_282, out_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_281 => add_261, mul_328, mul_329, sub_109
#   out_282 => relu_104
#   out_283 => convolution_111
# Graph fragment:
#   %sub_109 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_110, %unsqueeze_873), kwargs = {})
#   %mul_328 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_109, %unsqueeze_875), kwargs = {})
#   %mul_329 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_328, %unsqueeze_877), kwargs = {})
#   %add_261 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_329, %unsqueeze_879), kwargs = {})
#   %relu_104 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_261,), kwargs = {})
#   %convolution_111 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_104, %arg26_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ts/ctsgtid5zlvqlgyrmim7dk47mjg3p7z4pibd44js35ejjevxlgqx.py
# Topologically Sorted Source Nodes: [out_284, out_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_284 => add_263, mul_331, mul_332, sub_110
#   out_285 => relu_105
# Graph fragment:
#   %sub_110 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_111, %unsqueeze_881), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_110, %unsqueeze_883), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_331, %unsqueeze_885), kwargs = {})
#   %add_263 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_332, %unsqueeze_887), kwargs = {})
#   %relu_105 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_263,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/xe/cxe2fa5gpjn6mxvxnau3unmsleodpqm6y5nj6pwqdkudjdd76jse.py
# Topologically Sorted Source Nodes: [bottom_9], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   bottom_9 => _low_memory_max_pool2d_with_offsets_9
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_9 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_103, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_9 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 56
    x2 = (xindex // 1792)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x1) + (7168*x2)), None)
    tmp1 = tl.load(in_ptr0 + (32 + x0 + (64*x1) + (7168*x2)), None)
    tmp3 = tl.load(in_ptr0 + (3584 + x0 + (64*x1) + (7168*x2)), None)
    tmp5 = tl.load(in_ptr0 + (3616 + x0 + (64*x1) + (7168*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nz/cnzrhayg6vnwwftrphhz24ry4xyhubfhqa4moebp467xbvyger4o.py
# Topologically Sorted Source Nodes: [out_287, input_28, out_288, out_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_28 => add_259, mul_325, mul_326, sub_108
#   out_287 => add_265, mul_334, mul_335, sub_111
#   out_288 => add_266
#   out_289 => relu_106
# Graph fragment:
#   %sub_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_112, %unsqueeze_889), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_111, %unsqueeze_891), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_334, %unsqueeze_893), kwargs = {})
#   %add_265 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_335, %unsqueeze_895), kwargs = {})
#   %sub_108 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_109, %unsqueeze_865), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_108, %unsqueeze_867), kwargs = {})
#   %mul_326 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_325, %unsqueeze_869), kwargs = {})
#   %add_259 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_326, %unsqueeze_871), kwargs = {})
#   %add_266 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_265, %add_259), kwargs = {})
#   %relu_106 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_266,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/kl/cklacngtjdjvom6kaqheo5ixgq5crbe5lgikn3ra5v7zb25rlozj.py
# Topologically Sorted Source Nodes: [out_297, out_298, out_299, cat_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_14 => cat_14
#   out_297 => add_272, mul_343, mul_344, sub_114
#   out_298 => add_273
#   out_299 => relu_109
# Graph fragment:
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_913), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_915), kwargs = {})
#   %mul_344 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_343, %unsqueeze_917), kwargs = {})
#   %add_272 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_344, %unsqueeze_919), kwargs = {})
#   %add_273 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_272, %relu_106), kwargs = {})
#   %relu_109 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_273,), kwargs = {})
#   %cat_14 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_109, %relu_106], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x0 + (256*x1)), tmp19, None)
    tl.store(out_ptr1 + (x0 + (256*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rq/crqtiw76h2bxtbd55egld5hkeu6ezktenk7u4pbbeht2rlyb6e2m.py
# Topologically Sorted Source Nodes: [x_61, x_62, x_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_61 => add_275, mul_346, mul_347, sub_115
#   x_62 => add_276
#   x_63 => relu_110
# Graph fragment:
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_116, %unsqueeze_921), kwargs = {})
#   %mul_346 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %unsqueeze_923), kwargs = {})
#   %mul_347 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_346, %unsqueeze_925), kwargs = {})
#   %add_275 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_347, %unsqueeze_927), kwargs = {})
#   %add_276 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_275, %relu_109), kwargs = {})
#   %relu_110 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_276,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0 + (256*x1)), None)
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


# kernel path: /tmp/torchinductor_sahanp/hp/chpfobv77lu4mhgchxtopf46gb5a2m3tvnpvnxltdqd7o3bsnmib.py
# Topologically Sorted Source Nodes: [out_301, out_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_301 => add_280, mul_352, mul_353, sub_117
#   out_302 => relu_111
# Graph fragment:
#   %sub_117 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_118, %unsqueeze_937), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_117, %unsqueeze_939), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, %unsqueeze_941), kwargs = {})
#   %add_280 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_353, %unsqueeze_943), kwargs = {})
#   %relu_111 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_280,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/c4/cc4isndmx26auhlas3vbkkcii5ejyprwizeddwdxntxtbk7szhm2.py
# Topologically Sorted Source Nodes: [out_301, out_302, out_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_301 => add_280, mul_352, mul_353, sub_117
#   out_302 => relu_111
#   out_303 => convolution_119
# Graph fragment:
#   %sub_117 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_118, %unsqueeze_937), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_117, %unsqueeze_939), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, %unsqueeze_941), kwargs = {})
#   %add_280 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_353, %unsqueeze_943), kwargs = {})
#   %relu_111 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_280,), kwargs = {})
#   %convolution_119 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_111, %arg66_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/q6/cq6knaa6bx2i3n353ddkjpm26sjz4t7kwpzcamzclwvkd5svgx5g.py
# Topologically Sorted Source Nodes: [out_304, out_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_304 => add_282, mul_355, mul_356, sub_118
#   out_305 => relu_112
# Graph fragment:
#   %sub_118 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_119, %unsqueeze_945), kwargs = {})
#   %mul_355 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_118, %unsqueeze_947), kwargs = {})
#   %mul_356 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_355, %unsqueeze_949), kwargs = {})
#   %add_282 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_356, %unsqueeze_951), kwargs = {})
#   %relu_112 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_282,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
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


# kernel path: /tmp/torchinductor_sahanp/fu/cfumg3yunn2csye56rpwmt6iwiy76o2faibhlmmtal6tgg53sb3f.py
# Topologically Sorted Source Nodes: [bottom_12], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   bottom_12 => _low_memory_max_pool2d_with_offsets_12
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_12 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_110, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_16 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128) % 28
    x2 = (xindex // 3584)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + (256*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (256*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7296 + x0 + (256*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ux/cuxry45qbdkgol43w4cv36q2ltbqliiuilxb3dzzhyngcv5gtpbp.py
# Topologically Sorted Source Nodes: [out_307, input_30, out_308, out_309], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_30 => add_278, mul_349, mul_350, sub_116
#   out_307 => add_284, mul_358, mul_359, sub_119
#   out_308 => add_285
#   out_309 => relu_113
# Graph fragment:
#   %sub_119 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_120, %unsqueeze_953), kwargs = {})
#   %mul_358 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_119, %unsqueeze_955), kwargs = {})
#   %mul_359 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_358, %unsqueeze_957), kwargs = {})
#   %add_284 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_359, %unsqueeze_959), kwargs = {})
#   %sub_116 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_117, %unsqueeze_929), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_116, %unsqueeze_931), kwargs = {})
#   %mul_350 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_349, %unsqueeze_933), kwargs = {})
#   %add_278 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_350, %unsqueeze_935), kwargs = {})
#   %add_285 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_284, %add_278), kwargs = {})
#   %relu_113 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_285,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/54/c547g7tlkgqo543mrfrdd75fwhlf4igavzxkgts5ijuu5av577lt.py
# Topologically Sorted Source Nodes: [out_317, out_318, out_319, cat_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_15 => cat_15
#   out_317 => add_291, mul_367, mul_368, sub_122
#   out_318 => add_292
#   out_319 => relu_116
# Graph fragment:
#   %sub_122 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_123, %unsqueeze_977), kwargs = {})
#   %mul_367 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_122, %unsqueeze_979), kwargs = {})
#   %mul_368 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_367, %unsqueeze_981), kwargs = {})
#   %add_291 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_368, %unsqueeze_983), kwargs = {})
#   %add_292 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_291, %relu_113), kwargs = {})
#   %relu_116 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_292,), kwargs = {})
#   %cat_15 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_116, %relu_113], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x0 + (512*x1)), tmp19, None)
    tl.store(out_ptr1 + (x0 + (512*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/px/cpx6wb5sxfopkkgm3ehf6wbucrydeqwhzwhcezj4fdxpf5thzye5.py
# Topologically Sorted Source Nodes: [x_65, x_66, x_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_65 => add_294, mul_370, mul_371, sub_123
#   x_66 => add_295
#   x_67 => relu_117
# Graph fragment:
#   %sub_123 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_124, %unsqueeze_985), kwargs = {})
#   %mul_370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_123, %unsqueeze_987), kwargs = {})
#   %mul_371 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_370, %unsqueeze_989), kwargs = {})
#   %add_294 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_371, %unsqueeze_991), kwargs = {})
#   %add_295 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_294, %relu_116), kwargs = {})
#   %relu_117 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_295,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0 + (512*x1)), None)
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


# kernel path: /tmp/torchinductor_sahanp/j2/cj2vkb4s3bu5kkahshqxpda3ytmsqbaol3gyoh3doleox6rzgluz.py
# Topologically Sorted Source Nodes: [out_327, out_328, out_329, cat_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_16 => cat_16
#   out_327 => add_301, mul_379, mul_380, sub_126
#   out_328 => add_302
#   out_329 => relu_120
# Graph fragment:
#   %sub_126 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_127, %unsqueeze_1009), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_126, %unsqueeze_1011), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_379, %unsqueeze_1013), kwargs = {})
#   %add_301 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_380, %unsqueeze_1015), kwargs = {})
#   %add_302 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_301, %relu_117), kwargs = {})
#   %relu_120 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_302,), kwargs = {})
#   %cat_16 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_123, %relu_120, %relu_117], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
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
    tl.store(out_ptr0 + (x0 + (768*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ta/ctacjizffktlhxqwzcbpqlr26o2lu6k3jwnli22r2uuambllhlf3.py
# Topologically Sorted Source Nodes: [out_337, out_338, out_339, cat_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_16 => cat_16
#   out_337 => add_308, mul_388, mul_389, sub_129
#   out_338 => add_309
#   out_339 => relu_123
# Graph fragment:
#   %sub_129 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_130, %unsqueeze_1033), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %unsqueeze_1035), kwargs = {})
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_388, %unsqueeze_1037), kwargs = {})
#   %add_308 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_389, %unsqueeze_1039), kwargs = {})
#   %add_309 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_308, %relu_120), kwargs = {})
#   %relu_123 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_309,), kwargs = {})
#   %cat_16 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_123, %relu_120, %relu_117], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x0 + (768*x1)), tmp19, None)
    tl.store(out_ptr1 + (x0 + (768*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pz/cpzcz7vep6z5awqpzpeyhiqlkjvw7dcnp6hxvdne2wiqeoxedau5.py
# Topologically Sorted Source Nodes: [x_69, x_70, x_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_69 => add_311, mul_391, mul_392, sub_130
#   x_70 => add_312
#   x_71 => relu_124
# Graph fragment:
#   %sub_130 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_131, %unsqueeze_1041), kwargs = {})
#   %mul_391 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_130, %unsqueeze_1043), kwargs = {})
#   %mul_392 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_391, %unsqueeze_1045), kwargs = {})
#   %add_311 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_392, %unsqueeze_1047), kwargs = {})
#   %add_312 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_311, %relu_123), kwargs = {})
#   %relu_124 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_312,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0 + (768*x1)), None)
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


# kernel path: /tmp/torchinductor_sahanp/nh/cnhvfa4zzddrfmejdp5z5vbe3uqqdyxbrk66utl3s4kl3g5lu2gc.py
# Topologically Sorted Source Nodes: [out_347, out_348, out_349, cat_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_18 => cat_18
#   out_347 => add_318, mul_400, mul_401, sub_133
#   out_348 => add_319
#   out_349 => relu_127
# Graph fragment:
#   %sub_133 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_134, %unsqueeze_1065), kwargs = {})
#   %mul_400 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_133, %unsqueeze_1067), kwargs = {})
#   %mul_401 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_400, %unsqueeze_1069), kwargs = {})
#   %add_318 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_401, %unsqueeze_1071), kwargs = {})
#   %add_319 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_318, %relu_124), kwargs = {})
#   %relu_127 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_319,), kwargs = {})
#   %cat_18 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_137, %relu_134, %getitem_20, %relu_124, %relu_131], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
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
    tl.store(out_ptr0 + (x0 + (1152*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qq/cqq6l3xm7a6r5efulj7xwvpdxdz6tj3jwutg2r567cqtn37pkhur.py
# Topologically Sorted Source Nodes: [out_377, out_378, out_379, cat_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_18 => cat_18
#   out_377 => add_342, mul_430, mul_431, sub_143
#   out_378 => add_343
#   out_379 => relu_137
# Graph fragment:
#   %sub_143 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_144, %unsqueeze_1145), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_143, %unsqueeze_1147), kwargs = {})
#   %mul_431 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_430, %unsqueeze_1149), kwargs = {})
#   %add_342 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_431, %unsqueeze_1151), kwargs = {})
#   %add_343 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_342, %relu_134), kwargs = {})
#   %relu_137 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_343,), kwargs = {})
#   %cat_18 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_137, %relu_134, %getitem_20, %relu_124, %relu_131], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x0 + (1152*x1)), tmp19, None)
    tl.store(out_ptr1 + (x0 + (1152*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/i7/ci763xesqsa6ra5mx5re52amuhhuufpdlegu2iop7av2yqaxxpfj.py
# Topologically Sorted Source Nodes: [bottom_10], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   bottom_10 => _low_memory_max_pool2d_with_offsets_10
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_10 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_110, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_25 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128) % 28
    x2 = (xindex // 3584)
    x3 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + (256*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (256*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7296 + x0 + (256*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x0 + (1152*x3)), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bf/cbfnpes5f5fkyfcrqn6konyvnjt3crjgleeunwjozg6izupauorl.py
# Topologically Sorted Source Nodes: [x_77, x_78, x_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_77 => add_345, mul_433, mul_434, sub_144
#   x_78 => add_346
#   x_79 => relu_138
# Graph fragment:
#   %sub_144 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_145, %unsqueeze_1153), kwargs = {})
#   %mul_433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_144, %unsqueeze_1155), kwargs = {})
#   %mul_434 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_433, %unsqueeze_1157), kwargs = {})
#   %add_345 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_434, %unsqueeze_1159), kwargs = {})
#   %add_346 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_345, %relu_137), kwargs = {})
#   %relu_138 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_346,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0 + (1152*x1)), None)
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


# kernel path: /tmp/torchinductor_sahanp/wk/cwk4olb6xcm5dcyyxhp4cev67tfelq5upxujvpdnvxdoedwdflyr.py
# Topologically Sorted Source Nodes: [out_381, out_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_381 => add_350, mul_439, mul_440, sub_146
#   out_382 => relu_139
# Graph fragment:
#   %sub_146 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_147, %unsqueeze_1169), kwargs = {})
#   %mul_439 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_146, %unsqueeze_1171), kwargs = {})
#   %mul_440 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_439, %unsqueeze_1173), kwargs = {})
#   %add_350 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_440, %unsqueeze_1175), kwargs = {})
#   %relu_139 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_350,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/6o/c6o4huux2bne3cwtwqosp62bel53tsgltuav57s5zejjj2yoa22z.py
# Topologically Sorted Source Nodes: [out_381, out_382, out_383], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_381 => add_350, mul_439, mul_440, sub_146
#   out_382 => relu_139
#   out_383 => convolution_148
# Graph fragment:
#   %sub_146 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_147, %unsqueeze_1169), kwargs = {})
#   %mul_439 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_146, %unsqueeze_1171), kwargs = {})
#   %mul_440 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_439, %unsqueeze_1173), kwargs = {})
#   %add_350 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_440, %unsqueeze_1175), kwargs = {})
#   %relu_139 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_350,), kwargs = {})
#   %convolution_148 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_139, %arg211_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/vw/cvwx432fjgfoyqijjfz5zokymhuudwe45iegrwh4w5gtq6mt6aoe.py
# Topologically Sorted Source Nodes: [out_384, out_385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_384 => add_352, mul_442, mul_443, sub_147
#   out_385 => relu_140
# Graph fragment:
#   %sub_147 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_148, %unsqueeze_1177), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_147, %unsqueeze_1179), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_442, %unsqueeze_1181), kwargs = {})
#   %add_352 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_443, %unsqueeze_1183), kwargs = {})
#   %relu_140 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_352,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
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


# kernel path: /tmp/torchinductor_sahanp/x3/cx3njdgkobasfosacqqzlo6nqhg5rabszzqdhbi3qkcdwr7em5b5.py
# Topologically Sorted Source Nodes: [bottom_16], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   bottom_16 => _low_memory_max_pool2d_with_offsets_16
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_16 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_138, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_30 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256) % 14
    x2 = (xindex // 3584)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (512*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7424 + x0 + (512*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/u4/cu4n25h3fe7eizfoihdhtg7mtm65exmczlro545eg634lzburtsd.py
# Topologically Sorted Source Nodes: [out_387, input_32, out_388, out_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_32 => add_348, mul_436, mul_437, sub_145
#   out_387 => add_354, mul_445, mul_446, sub_148
#   out_388 => add_355
#   out_389 => relu_141
# Graph fragment:
#   %sub_148 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_149, %unsqueeze_1185), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_148, %unsqueeze_1187), kwargs = {})
#   %mul_446 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_445, %unsqueeze_1189), kwargs = {})
#   %add_354 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_446, %unsqueeze_1191), kwargs = {})
#   %sub_145 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_146, %unsqueeze_1161), kwargs = {})
#   %mul_436 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_145, %unsqueeze_1163), kwargs = {})
#   %mul_437 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_436, %unsqueeze_1165), kwargs = {})
#   %add_348 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_437, %unsqueeze_1167), kwargs = {})
#   %add_355 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_354, %add_348), kwargs = {})
#   %relu_141 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_355,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/r7/cr7meaghi3i7hgex4ldd2v3qpsstj3nkq5v36wm762bxkqv6pnst.py
# Topologically Sorted Source Nodes: [out_397, out_398, out_399, cat_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_19 => cat_19
#   out_397 => add_361, mul_454, mul_455, sub_151
#   out_398 => add_362
#   out_399 => relu_144
# Graph fragment:
#   %sub_151 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_152, %unsqueeze_1209), kwargs = {})
#   %mul_454 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_151, %unsqueeze_1211), kwargs = {})
#   %mul_455 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_454, %unsqueeze_1213), kwargs = {})
#   %add_361 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_455, %unsqueeze_1215), kwargs = {})
#   %add_362 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_361, %relu_141), kwargs = {})
#   %relu_144 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_362,), kwargs = {})
#   %cat_19 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_144, %relu_141], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x0 + (1024*x1)), tmp19, None)
    tl.store(out_ptr1 + (x0 + (1024*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/za/czacimhcrrc3xhzij6q3uxamtybfjs7lggfuawchzrpccynub22i.py
# Topologically Sorted Source Nodes: [x_81, x_82, x_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_81 => add_364, mul_457, mul_458, sub_152
#   x_82 => add_365
#   x_83 => relu_145
# Graph fragment:
#   %sub_152 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_153, %unsqueeze_1217), kwargs = {})
#   %mul_457 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_152, %unsqueeze_1219), kwargs = {})
#   %mul_458 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_457, %unsqueeze_1221), kwargs = {})
#   %add_364 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_458, %unsqueeze_1223), kwargs = {})
#   %add_365 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_364, %relu_144), kwargs = {})
#   %relu_145 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_365,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0 + (1024*x1)), None)
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


# kernel path: /tmp/torchinductor_sahanp/bk/cbkrjfqtkufbim7zv5vaoui2d4hfwa26ctjwpgp23rqywtcxjsj2.py
# Topologically Sorted Source Nodes: [out_407, out_408, out_409, cat_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_20 => cat_20
#   out_407 => add_371, mul_466, mul_467, sub_155
#   out_408 => add_372
#   out_409 => relu_148
# Graph fragment:
#   %sub_155 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_156, %unsqueeze_1241), kwargs = {})
#   %mul_466 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_155, %unsqueeze_1243), kwargs = {})
#   %mul_467 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_466, %unsqueeze_1245), kwargs = {})
#   %add_371 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_467, %unsqueeze_1247), kwargs = {})
#   %add_372 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_371, %relu_145), kwargs = {})
#   %relu_148 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_372,), kwargs = {})
#   %cat_20 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_151, %relu_148, %relu_145], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
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
    tl.store(out_ptr0 + (x0 + (1536*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dp/cdpdw6qqxjbcrasajmy7the3rnsjzfi7m33b5hzjgywpijtief5g.py
# Topologically Sorted Source Nodes: [out_417, out_418, out_419, cat_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_20 => cat_20
#   out_417 => add_378, mul_475, mul_476, sub_158
#   out_418 => add_379
#   out_419 => relu_151
# Graph fragment:
#   %sub_158 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_159, %unsqueeze_1265), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_158, %unsqueeze_1267), kwargs = {})
#   %mul_476 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_475, %unsqueeze_1269), kwargs = {})
#   %add_378 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_476, %unsqueeze_1271), kwargs = {})
#   %add_379 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_378, %relu_148), kwargs = {})
#   %relu_151 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_379,), kwargs = {})
#   %cat_20 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_151, %relu_148, %relu_145], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x0 + (1536*x1)), tmp19, None)
    tl.store(out_ptr1 + (x0 + (1536*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qb/cqb5okrpddqxu5oq2msbfmhl2ka2bkzmwwlnbal4245gpy46uzo2.py
# Topologically Sorted Source Nodes: [x_85, x_86, x_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_85 => add_381, mul_478, mul_479, sub_159
#   x_86 => add_382
#   x_87 => relu_152
# Graph fragment:
#   %sub_159 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_160, %unsqueeze_1273), kwargs = {})
#   %mul_478 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_159, %unsqueeze_1275), kwargs = {})
#   %mul_479 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_478, %unsqueeze_1277), kwargs = {})
#   %add_381 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_479, %unsqueeze_1279), kwargs = {})
#   %add_382 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_381, %relu_151), kwargs = {})
#   %relu_152 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_382,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0 + (1536*x1)), None)
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


# kernel path: /tmp/torchinductor_sahanp/2j/c2jaeeowbcfxpds4ec7kwdp5s7qspm4wpjxfapzfnikp4bzra36b.py
# Topologically Sorted Source Nodes: [out_427, out_428, out_429, cat_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_22 => cat_22
#   out_427 => add_388, mul_487, mul_488, sub_162
#   out_428 => add_389
#   out_429 => relu_155
# Graph fragment:
#   %sub_162 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_163, %unsqueeze_1297), kwargs = {})
#   %mul_487 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_162, %unsqueeze_1299), kwargs = {})
#   %mul_488 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_487, %unsqueeze_1301), kwargs = {})
#   %add_388 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_488, %unsqueeze_1303), kwargs = {})
#   %add_389 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_388, %relu_152), kwargs = {})
#   %relu_155 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_389,), kwargs = {})
#   %cat_22 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_165, %relu_162, %relu_152, %relu_159], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
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
    tl.store(out_ptr0 + (x0 + (2048*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/af/cafxpu3abaawaiwp3ogqoiq6wlrfwrd4s4hsyhxl7vbfjsikg2sq.py
# Topologically Sorted Source Nodes: [out_457, out_458, out_459, cat_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_22 => cat_22
#   out_457 => add_412, mul_517, mul_518, sub_172
#   out_458 => add_413
#   out_459 => relu_165
# Graph fragment:
#   %sub_172 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_173, %unsqueeze_1377), kwargs = {})
#   %mul_517 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_172, %unsqueeze_1379), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_517, %unsqueeze_1381), kwargs = {})
#   %add_412 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_518, %unsqueeze_1383), kwargs = {})
#   %add_413 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_412, %relu_162), kwargs = {})
#   %relu_165 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_413,), kwargs = {})
#   %cat_22 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_165, %relu_162, %relu_152, %relu_159], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x0 + (2048*x1)), tmp19, None)
    tl.store(out_ptr1 + (x0 + (2048*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fs/cfsg34squjjwev2oydfcmwzk2dit2ogveb7bjahejpblheo7sfre.py
# Topologically Sorted Source Nodes: [x_93, x_94, x_95, cat_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_26 => cat_26
#   x_93 => add_415, mul_520, mul_521, sub_173
#   x_94 => add_416
#   x_95 => relu_166
# Graph fragment:
#   %sub_173 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_174, %unsqueeze_1385), kwargs = {})
#   %mul_520 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_173, %unsqueeze_1387), kwargs = {})
#   %mul_521 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_520, %unsqueeze_1389), kwargs = {})
#   %add_415 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_521, %unsqueeze_1391), kwargs = {})
#   %add_416 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_415, %relu_165), kwargs = {})
#   %relu_166 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_416,), kwargs = {})
#   %cat_26 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_193, %relu_190, %getitem_26, %relu_166, %relu_180, %relu_187], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0 + (2048*x1)), None)
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
    tl.store(out_ptr0 + (x0 + (2816*x1)), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ra/crangj2j3hrsu7jm42ju4dowweuczrmlcfjuhzqn3bb627q5hrlx.py
# Topologically Sorted Source Nodes: [out_467, out_468, out_469], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_467 => add_422, mul_529, mul_530, sub_176
#   out_468 => add_423
#   out_469 => relu_169
# Graph fragment:
#   %sub_176 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_177, %unsqueeze_1409), kwargs = {})
#   %mul_529 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_176, %unsqueeze_1411), kwargs = {})
#   %mul_530 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_529, %unsqueeze_1413), kwargs = {})
#   %add_422 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_530, %unsqueeze_1415), kwargs = {})
#   %add_423 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_422, %relu_166), kwargs = {})
#   %relu_169 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_423,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
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


# kernel path: /tmp/torchinductor_sahanp/lt/cltl43aczazt6payjmskqyxrzvi57qgkot2qaqsauw4t5dqalwlx.py
# Topologically Sorted Source Nodes: [out_507, out_508, out_509, cat_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_26 => cat_26
#   out_507 => add_456, mul_571, mul_572, sub_190
#   out_508 => add_457
#   out_509 => relu_183
# Graph fragment:
#   %sub_190 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_191, %unsqueeze_1521), kwargs = {})
#   %mul_571 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_190, %unsqueeze_1523), kwargs = {})
#   %mul_572 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_571, %unsqueeze_1525), kwargs = {})
#   %add_456 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_572, %unsqueeze_1527), kwargs = {})
#   %add_457 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_456, %relu_180), kwargs = {})
#   %relu_183 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_457,), kwargs = {})
#   %cat_26 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_193, %relu_190, %getitem_26, %relu_166, %relu_180, %relu_187], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_41', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
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
    tl.store(out_ptr0 + (x0 + (2816*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/np/cnp6pkcw4hi4toa6l5tapqw4xa23jty57rgzkzavgmolh24enmu3.py
# Topologically Sorted Source Nodes: [out_537, out_538, out_539, cat_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_26 => cat_26
#   out_537 => add_480, mul_601, mul_602, sub_200
#   out_538 => add_481
#   out_539 => relu_193
# Graph fragment:
#   %sub_200 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_201, %unsqueeze_1601), kwargs = {})
#   %mul_601 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_200, %unsqueeze_1603), kwargs = {})
#   %mul_602 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_601, %unsqueeze_1605), kwargs = {})
#   %add_480 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_602, %unsqueeze_1607), kwargs = {})
#   %add_481 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_480, %relu_190), kwargs = {})
#   %relu_193 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_481,), kwargs = {})
#   %cat_26 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_193, %relu_190, %getitem_26, %relu_166, %relu_180, %relu_187], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x0 + (2816*x1)), tmp19, None)
    tl.store(out_ptr1 + (x0 + (2816*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4i/c4ifygddfm6oo4xhqaleu3ejjkjodpri34pokg3d7ctlt4ftmykj.py
# Topologically Sorted Source Nodes: [bottom_13], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   bottom_13 => _low_memory_max_pool2d_with_offsets_13
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_13 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_138, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_43 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_43(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256) % 14
    x2 = (xindex // 3584)
    x3 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (512*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7424 + x0 + (512*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x0 + (2816*x3)), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/b3/cb326pxgr4xfilqhsgdib5gpnkxvqjlawvsya6ynl5ejw4dywylp.py
# Topologically Sorted Source Nodes: [x_109, x_110, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_109 => add_483, mul_604, mul_605, sub_201
#   x_110 => add_484
#   x_111 => relu_194
# Graph fragment:
#   %sub_201 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_202, %unsqueeze_1609), kwargs = {})
#   %mul_604 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_201, %unsqueeze_1611), kwargs = {})
#   %mul_605 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_604, %unsqueeze_1613), kwargs = {})
#   %add_483 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_605, %unsqueeze_1615), kwargs = {})
#   %add_484 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_483, %relu_193), kwargs = {})
#   %relu_194 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_484,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0 + (2816*x1)), None)
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


# kernel path: /tmp/torchinductor_sahanp/lg/clgmnzh62jwqmlrapgc5hx3g42myxho3gll32fahlawp32fsi7dv.py
# Topologically Sorted Source Nodes: [bottom_17, cat_27], Original ATen: [aten.max_pool2d_with_indices, aten.cat]
# Source node to ATen node mapping:
#   bottom_17 => _low_memory_max_pool2d_with_offsets_17
#   cat_27 => cat_27
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_17 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_194, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_200, %relu_197, %getitem_34], 1), kwargs = {})
triton_poi_fused_cat_max_pool2d_with_indices_45 = async_compile.triton('triton_poi_fused_cat_max_pool2d_with_indices_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_max_pool2d_with_indices_45(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584)
    x4 = xindex
    x3 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + (1024*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (1024*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7680 + x0 + (1024*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x4), tmp6, None)
    tl.store(out_ptr1 + (x0 + (2560*x3)), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5z/c5z3q2ekkchjxxl4esbx6id65d54yvrhqo55yy5koouw6wfrf4af.py
# Topologically Sorted Source Nodes: [out_541, out_542], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_541 => add_488, mul_610, mul_611, sub_203
#   out_542 => relu_195
# Graph fragment:
#   %sub_203 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_204, %unsqueeze_1625), kwargs = {})
#   %mul_610 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_203, %unsqueeze_1627), kwargs = {})
#   %mul_611 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_610, %unsqueeze_1629), kwargs = {})
#   %add_488 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_611, %unsqueeze_1631), kwargs = {})
#   %relu_195 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_488,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ov/covvlpqpdrqgiis7gkflvmaya345q5jyeg6n73w4x7adynrtpzxr.py
# Topologically Sorted Source Nodes: [out_541, out_542, out_543], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_541 => add_488, mul_610, mul_611, sub_203
#   out_542 => relu_195
#   out_543 => convolution_205
# Graph fragment:
#   %sub_203 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_204, %unsqueeze_1625), kwargs = {})
#   %mul_610 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_203, %unsqueeze_1627), kwargs = {})
#   %mul_611 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_610, %unsqueeze_1629), kwargs = {})
#   %add_488 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_611, %unsqueeze_1631), kwargs = {})
#   %relu_195 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_488,), kwargs = {})
#   %convolution_205 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_195, %arg496_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
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


# kernel path: /tmp/torchinductor_sahanp/zb/czbbd3hfd37u2pivy6zyihlck3phzuxyifvhn7bwjfpzsgj5n2j7.py
# Topologically Sorted Source Nodes: [out_544, out_545], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_544 => add_490, mul_613, mul_614, sub_204
#   out_545 => relu_196
# Graph fragment:
#   %sub_204 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_205, %unsqueeze_1633), kwargs = {})
#   %mul_613 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_204, %unsqueeze_1635), kwargs = {})
#   %mul_614 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_613, %unsqueeze_1637), kwargs = {})
#   %add_490 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_614, %unsqueeze_1639), kwargs = {})
#   %relu_196 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_490,), kwargs = {})
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
    x0 = xindex % 512
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


# kernel path: /tmp/torchinductor_sahanp/yq/cyqtnqlgqslkve5fz3zo7qaduce4bezqhnvoqbsxuamdaeabhmpw.py
# Topologically Sorted Source Nodes: [out_547, input_34, out_548, out_549], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_34 => add_486, mul_607, mul_608, sub_202
#   out_547 => add_492, mul_616, mul_617, sub_205
#   out_548 => add_493
#   out_549 => relu_197
# Graph fragment:
#   %sub_205 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_206, %unsqueeze_1641), kwargs = {})
#   %mul_616 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_205, %unsqueeze_1643), kwargs = {})
#   %mul_617 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_616, %unsqueeze_1645), kwargs = {})
#   %add_492 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_617, %unsqueeze_1647), kwargs = {})
#   %sub_202 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_203, %unsqueeze_1617), kwargs = {})
#   %mul_607 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_202, %unsqueeze_1619), kwargs = {})
#   %mul_608 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_607, %unsqueeze_1621), kwargs = {})
#   %add_486 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_608, %unsqueeze_1623), kwargs = {})
#   %add_493 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_492, %add_486), kwargs = {})
#   %relu_197 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_493,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/im/cim2uijngmlu5fjg4lffwdmtaqvcjigg6ton4m3g5rw5zwvgvk7g.py
# Topologically Sorted Source Nodes: [out_557, out_558, out_559, cat_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_27 => cat_27
#   out_557 => add_499, mul_625, mul_626, sub_208
#   out_558 => add_500
#   out_559 => relu_200
# Graph fragment:
#   %sub_208 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_209, %unsqueeze_1665), kwargs = {})
#   %mul_625 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_208, %unsqueeze_1667), kwargs = {})
#   %mul_626 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_625, %unsqueeze_1669), kwargs = {})
#   %add_499 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_626, %unsqueeze_1671), kwargs = {})
#   %add_500 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_499, %relu_197), kwargs = {})
#   %relu_200 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_500,), kwargs = {})
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_200, %relu_197, %getitem_34], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_50', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x0 + (2560*x1)), tmp19, None)
    tl.store(out_ptr1 + (x0 + (2560*x1)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rl/crlfsjeqmkhay6mx2nfll3tbwgar4d52h6ia7dxxwervcfd7wspf.py
# Topologically Sorted Source Nodes: [x_113, x_114, x_115, x_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   x_113 => add_502, mul_628, mul_629, sub_209
#   x_114 => add_503
#   x_115 => relu_201
#   x_116 => mean_1
# Graph fragment:
#   %sub_209 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_210, %unsqueeze_1673), kwargs = {})
#   %mul_628 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_209, %unsqueeze_1675), kwargs = {})
#   %mul_629 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_628, %unsqueeze_1677), kwargs = {})
#   %add_502 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_629, %unsqueeze_1679), kwargs = {})
#   %add_503 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_502, %relu_200), kwargs = {})
#   %relu_201 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_503,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_201, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (50176*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0 + (2560*r2) + (125440*x1)), rmask, other=0.0)
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
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = 49.0
    tmp25 = tmp23 / tmp24
    tl.store(out_ptr1 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wn/cwnxrjqejioopkaqsmnpqskruznwnqgafwlu5vohwyggpqb62r4y.py
# Topologically Sorted Source Nodes: [x_113, x_114, x_115, x_116, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_113 => add_502, mul_628, mul_629, sub_209
#   x_114 => add_503
#   x_115 => relu_201
#   x_116 => mean_1
#   x_118 => convolution_211
# Graph fragment:
#   %sub_209 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_210, %unsqueeze_1673), kwargs = {})
#   %mul_628 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_209, %unsqueeze_1675), kwargs = {})
#   %mul_629 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_628, %unsqueeze_1677), kwargs = {})
#   %add_502 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_629, %unsqueeze_1679), kwargs = {})
#   %add_503 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_502, %relu_200), kwargs = {})
#   %relu_201 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_503,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_201, [-1, -2], True), kwargs = {})
#   %convolution_211 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_1, %arg526_1, %arg527_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_52(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (16, ), (1, ))
    assert_size_stride(arg9_1, (16, ), (1, ))
    assert_size_stride(arg10_1, (16, ), (1, ))
    assert_size_stride(arg11_1, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg12_1, (32, ), (1, ))
    assert_size_stride(arg13_1, (32, ), (1, ))
    assert_size_stride(arg14_1, (32, ), (1, ))
    assert_size_stride(arg15_1, (32, ), (1, ))
    assert_size_stride(arg16_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg27_1, (64, ), (1, ))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, ), (1, ))
    assert_size_stride(arg31_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (64, ), (1, ))
    assert_size_stride(arg39_1, (64, ), (1, ))
    assert_size_stride(arg40_1, (64, ), (1, ))
    assert_size_stride(arg41_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (128, ), (1, ))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (128, ), (1, ))
    assert_size_stride(arg51_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (256, ), (1, ))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, ), (1, ))
    assert_size_stride(arg76_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg82_1, (128, ), (1, ))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg102_1, (128, ), (1, ))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, ), (1, ))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg107_1, (256, ), (1, ))
    assert_size_stride(arg108_1, (256, ), (1, ))
    assert_size_stride(arg109_1, (256, ), (1, ))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg112_1, (128, ), (1, ))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg117_1, (128, ), (1, ))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (256, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (256, ), (1, ))
    assert_size_stride(arg131_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (128, ), (1, ))
    assert_size_stride(arg135_1, (128, ), (1, ))
    assert_size_stride(arg136_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg147_1, (128, ), (1, ))
    assert_size_stride(arg148_1, (128, ), (1, ))
    assert_size_stride(arg149_1, (128, ), (1, ))
    assert_size_stride(arg150_1, (128, ), (1, ))
    assert_size_stride(arg151_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg152_1, (128, ), (1, ))
    assert_size_stride(arg153_1, (128, ), (1, ))
    assert_size_stride(arg154_1, (128, ), (1, ))
    assert_size_stride(arg155_1, (128, ), (1, ))
    assert_size_stride(arg156_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg157_1, (256, ), (1, ))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (256, ), (1, ))
    assert_size_stride(arg160_1, (256, ), (1, ))
    assert_size_stride(arg161_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg162_1, (256, ), (1, ))
    assert_size_stride(arg163_1, (256, ), (1, ))
    assert_size_stride(arg164_1, (256, ), (1, ))
    assert_size_stride(arg165_1, (256, ), (1, ))
    assert_size_stride(arg166_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg167_1, (128, ), (1, ))
    assert_size_stride(arg168_1, (128, ), (1, ))
    assert_size_stride(arg169_1, (128, ), (1, ))
    assert_size_stride(arg170_1, (128, ), (1, ))
    assert_size_stride(arg171_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg172_1, (128, ), (1, ))
    assert_size_stride(arg173_1, (128, ), (1, ))
    assert_size_stride(arg174_1, (128, ), (1, ))
    assert_size_stride(arg175_1, (128, ), (1, ))
    assert_size_stride(arg176_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg177_1, (256, ), (1, ))
    assert_size_stride(arg178_1, (256, ), (1, ))
    assert_size_stride(arg179_1, (256, ), (1, ))
    assert_size_stride(arg180_1, (256, ), (1, ))
    assert_size_stride(arg181_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg182_1, (128, ), (1, ))
    assert_size_stride(arg183_1, (128, ), (1, ))
    assert_size_stride(arg184_1, (128, ), (1, ))
    assert_size_stride(arg185_1, (128, ), (1, ))
    assert_size_stride(arg186_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg187_1, (128, ), (1, ))
    assert_size_stride(arg188_1, (128, ), (1, ))
    assert_size_stride(arg189_1, (128, ), (1, ))
    assert_size_stride(arg190_1, (128, ), (1, ))
    assert_size_stride(arg191_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (256, ), (1, ))
    assert_size_stride(arg194_1, (256, ), (1, ))
    assert_size_stride(arg195_1, (256, ), (1, ))
    assert_size_stride(arg196_1, (256, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg197_1, (256, ), (1, ))
    assert_size_stride(arg198_1, (256, ), (1, ))
    assert_size_stride(arg199_1, (256, ), (1, ))
    assert_size_stride(arg200_1, (256, ), (1, ))
    assert_size_stride(arg201_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (512, ), (1, ))
    assert_size_stride(arg204_1, (512, ), (1, ))
    assert_size_stride(arg205_1, (512, ), (1, ))
    assert_size_stride(arg206_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg207_1, (256, ), (1, ))
    assert_size_stride(arg208_1, (256, ), (1, ))
    assert_size_stride(arg209_1, (256, ), (1, ))
    assert_size_stride(arg210_1, (256, ), (1, ))
    assert_size_stride(arg211_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg212_1, (256, ), (1, ))
    assert_size_stride(arg213_1, (256, ), (1, ))
    assert_size_stride(arg214_1, (256, ), (1, ))
    assert_size_stride(arg215_1, (256, ), (1, ))
    assert_size_stride(arg216_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg217_1, (512, ), (1, ))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (512, ), (1, ))
    assert_size_stride(arg220_1, (512, ), (1, ))
    assert_size_stride(arg221_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg222_1, (256, ), (1, ))
    assert_size_stride(arg223_1, (256, ), (1, ))
    assert_size_stride(arg224_1, (256, ), (1, ))
    assert_size_stride(arg225_1, (256, ), (1, ))
    assert_size_stride(arg226_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg227_1, (256, ), (1, ))
    assert_size_stride(arg228_1, (256, ), (1, ))
    assert_size_stride(arg229_1, (256, ), (1, ))
    assert_size_stride(arg230_1, (256, ), (1, ))
    assert_size_stride(arg231_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg232_1, (512, ), (1, ))
    assert_size_stride(arg233_1, (512, ), (1, ))
    assert_size_stride(arg234_1, (512, ), (1, ))
    assert_size_stride(arg235_1, (512, ), (1, ))
    assert_size_stride(arg236_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg237_1, (512, ), (1, ))
    assert_size_stride(arg238_1, (512, ), (1, ))
    assert_size_stride(arg239_1, (512, ), (1, ))
    assert_size_stride(arg240_1, (512, ), (1, ))
    assert_size_stride(arg241_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg242_1, (256, ), (1, ))
    assert_size_stride(arg243_1, (256, ), (1, ))
    assert_size_stride(arg244_1, (256, ), (1, ))
    assert_size_stride(arg245_1, (256, ), (1, ))
    assert_size_stride(arg246_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg247_1, (256, ), (1, ))
    assert_size_stride(arg248_1, (256, ), (1, ))
    assert_size_stride(arg249_1, (256, ), (1, ))
    assert_size_stride(arg250_1, (256, ), (1, ))
    assert_size_stride(arg251_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg252_1, (512, ), (1, ))
    assert_size_stride(arg253_1, (512, ), (1, ))
    assert_size_stride(arg254_1, (512, ), (1, ))
    assert_size_stride(arg255_1, (512, ), (1, ))
    assert_size_stride(arg256_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg257_1, (256, ), (1, ))
    assert_size_stride(arg258_1, (256, ), (1, ))
    assert_size_stride(arg259_1, (256, ), (1, ))
    assert_size_stride(arg260_1, (256, ), (1, ))
    assert_size_stride(arg261_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg262_1, (256, ), (1, ))
    assert_size_stride(arg263_1, (256, ), (1, ))
    assert_size_stride(arg264_1, (256, ), (1, ))
    assert_size_stride(arg265_1, (256, ), (1, ))
    assert_size_stride(arg266_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg267_1, (512, ), (1, ))
    assert_size_stride(arg268_1, (512, ), (1, ))
    assert_size_stride(arg269_1, (512, ), (1, ))
    assert_size_stride(arg270_1, (512, ), (1, ))
    assert_size_stride(arg271_1, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg272_1, (512, ), (1, ))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (512, ), (1, ))
    assert_size_stride(arg275_1, (512, ), (1, ))
    assert_size_stride(arg276_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg277_1, (256, ), (1, ))
    assert_size_stride(arg278_1, (256, ), (1, ))
    assert_size_stride(arg279_1, (256, ), (1, ))
    assert_size_stride(arg280_1, (256, ), (1, ))
    assert_size_stride(arg281_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg282_1, (256, ), (1, ))
    assert_size_stride(arg283_1, (256, ), (1, ))
    assert_size_stride(arg284_1, (256, ), (1, ))
    assert_size_stride(arg285_1, (256, ), (1, ))
    assert_size_stride(arg286_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg287_1, (512, ), (1, ))
    assert_size_stride(arg288_1, (512, ), (1, ))
    assert_size_stride(arg289_1, (512, ), (1, ))
    assert_size_stride(arg290_1, (512, ), (1, ))
    assert_size_stride(arg291_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg292_1, (256, ), (1, ))
    assert_size_stride(arg293_1, (256, ), (1, ))
    assert_size_stride(arg294_1, (256, ), (1, ))
    assert_size_stride(arg295_1, (256, ), (1, ))
    assert_size_stride(arg296_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg297_1, (256, ), (1, ))
    assert_size_stride(arg298_1, (256, ), (1, ))
    assert_size_stride(arg299_1, (256, ), (1, ))
    assert_size_stride(arg300_1, (256, ), (1, ))
    assert_size_stride(arg301_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg302_1, (512, ), (1, ))
    assert_size_stride(arg303_1, (512, ), (1, ))
    assert_size_stride(arg304_1, (512, ), (1, ))
    assert_size_stride(arg305_1, (512, ), (1, ))
    assert_size_stride(arg306_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg307_1, (512, ), (1, ))
    assert_size_stride(arg308_1, (512, ), (1, ))
    assert_size_stride(arg309_1, (512, ), (1, ))
    assert_size_stride(arg310_1, (512, ), (1, ))
    assert_size_stride(arg311_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg312_1, (256, ), (1, ))
    assert_size_stride(arg313_1, (256, ), (1, ))
    assert_size_stride(arg314_1, (256, ), (1, ))
    assert_size_stride(arg315_1, (256, ), (1, ))
    assert_size_stride(arg316_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg317_1, (256, ), (1, ))
    assert_size_stride(arg318_1, (256, ), (1, ))
    assert_size_stride(arg319_1, (256, ), (1, ))
    assert_size_stride(arg320_1, (256, ), (1, ))
    assert_size_stride(arg321_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg322_1, (512, ), (1, ))
    assert_size_stride(arg323_1, (512, ), (1, ))
    assert_size_stride(arg324_1, (512, ), (1, ))
    assert_size_stride(arg325_1, (512, ), (1, ))
    assert_size_stride(arg326_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg327_1, (256, ), (1, ))
    assert_size_stride(arg328_1, (256, ), (1, ))
    assert_size_stride(arg329_1, (256, ), (1, ))
    assert_size_stride(arg330_1, (256, ), (1, ))
    assert_size_stride(arg331_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg332_1, (256, ), (1, ))
    assert_size_stride(arg333_1, (256, ), (1, ))
    assert_size_stride(arg334_1, (256, ), (1, ))
    assert_size_stride(arg335_1, (256, ), (1, ))
    assert_size_stride(arg336_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg337_1, (512, ), (1, ))
    assert_size_stride(arg338_1, (512, ), (1, ))
    assert_size_stride(arg339_1, (512, ), (1, ))
    assert_size_stride(arg340_1, (512, ), (1, ))
    assert_size_stride(arg341_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg342_1, (512, ), (1, ))
    assert_size_stride(arg343_1, (512, ), (1, ))
    assert_size_stride(arg344_1, (512, ), (1, ))
    assert_size_stride(arg345_1, (512, ), (1, ))
    assert_size_stride(arg346_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg347_1, (256, ), (1, ))
    assert_size_stride(arg348_1, (256, ), (1, ))
    assert_size_stride(arg349_1, (256, ), (1, ))
    assert_size_stride(arg350_1, (256, ), (1, ))
    assert_size_stride(arg351_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg352_1, (256, ), (1, ))
    assert_size_stride(arg353_1, (256, ), (1, ))
    assert_size_stride(arg354_1, (256, ), (1, ))
    assert_size_stride(arg355_1, (256, ), (1, ))
    assert_size_stride(arg356_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg357_1, (512, ), (1, ))
    assert_size_stride(arg358_1, (512, ), (1, ))
    assert_size_stride(arg359_1, (512, ), (1, ))
    assert_size_stride(arg360_1, (512, ), (1, ))
    assert_size_stride(arg361_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg362_1, (256, ), (1, ))
    assert_size_stride(arg363_1, (256, ), (1, ))
    assert_size_stride(arg364_1, (256, ), (1, ))
    assert_size_stride(arg365_1, (256, ), (1, ))
    assert_size_stride(arg366_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg367_1, (256, ), (1, ))
    assert_size_stride(arg368_1, (256, ), (1, ))
    assert_size_stride(arg369_1, (256, ), (1, ))
    assert_size_stride(arg370_1, (256, ), (1, ))
    assert_size_stride(arg371_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg372_1, (512, ), (1, ))
    assert_size_stride(arg373_1, (512, ), (1, ))
    assert_size_stride(arg374_1, (512, ), (1, ))
    assert_size_stride(arg375_1, (512, ), (1, ))
    assert_size_stride(arg376_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg377_1, (512, ), (1, ))
    assert_size_stride(arg378_1, (512, ), (1, ))
    assert_size_stride(arg379_1, (512, ), (1, ))
    assert_size_stride(arg380_1, (512, ), (1, ))
    assert_size_stride(arg381_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg382_1, (256, ), (1, ))
    assert_size_stride(arg383_1, (256, ), (1, ))
    assert_size_stride(arg384_1, (256, ), (1, ))
    assert_size_stride(arg385_1, (256, ), (1, ))
    assert_size_stride(arg386_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg387_1, (256, ), (1, ))
    assert_size_stride(arg388_1, (256, ), (1, ))
    assert_size_stride(arg389_1, (256, ), (1, ))
    assert_size_stride(arg390_1, (256, ), (1, ))
    assert_size_stride(arg391_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg392_1, (512, ), (1, ))
    assert_size_stride(arg393_1, (512, ), (1, ))
    assert_size_stride(arg394_1, (512, ), (1, ))
    assert_size_stride(arg395_1, (512, ), (1, ))
    assert_size_stride(arg396_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg397_1, (256, ), (1, ))
    assert_size_stride(arg398_1, (256, ), (1, ))
    assert_size_stride(arg399_1, (256, ), (1, ))
    assert_size_stride(arg400_1, (256, ), (1, ))
    assert_size_stride(arg401_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg402_1, (256, ), (1, ))
    assert_size_stride(arg403_1, (256, ), (1, ))
    assert_size_stride(arg404_1, (256, ), (1, ))
    assert_size_stride(arg405_1, (256, ), (1, ))
    assert_size_stride(arg406_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg407_1, (512, ), (1, ))
    assert_size_stride(arg408_1, (512, ), (1, ))
    assert_size_stride(arg409_1, (512, ), (1, ))
    assert_size_stride(arg410_1, (512, ), (1, ))
    assert_size_stride(arg411_1, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg412_1, (512, ), (1, ))
    assert_size_stride(arg413_1, (512, ), (1, ))
    assert_size_stride(arg414_1, (512, ), (1, ))
    assert_size_stride(arg415_1, (512, ), (1, ))
    assert_size_stride(arg416_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg417_1, (256, ), (1, ))
    assert_size_stride(arg418_1, (256, ), (1, ))
    assert_size_stride(arg419_1, (256, ), (1, ))
    assert_size_stride(arg420_1, (256, ), (1, ))
    assert_size_stride(arg421_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg422_1, (256, ), (1, ))
    assert_size_stride(arg423_1, (256, ), (1, ))
    assert_size_stride(arg424_1, (256, ), (1, ))
    assert_size_stride(arg425_1, (256, ), (1, ))
    assert_size_stride(arg426_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg427_1, (512, ), (1, ))
    assert_size_stride(arg428_1, (512, ), (1, ))
    assert_size_stride(arg429_1, (512, ), (1, ))
    assert_size_stride(arg430_1, (512, ), (1, ))
    assert_size_stride(arg431_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg432_1, (256, ), (1, ))
    assert_size_stride(arg433_1, (256, ), (1, ))
    assert_size_stride(arg434_1, (256, ), (1, ))
    assert_size_stride(arg435_1, (256, ), (1, ))
    assert_size_stride(arg436_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg437_1, (256, ), (1, ))
    assert_size_stride(arg438_1, (256, ), (1, ))
    assert_size_stride(arg439_1, (256, ), (1, ))
    assert_size_stride(arg440_1, (256, ), (1, ))
    assert_size_stride(arg441_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg442_1, (512, ), (1, ))
    assert_size_stride(arg443_1, (512, ), (1, ))
    assert_size_stride(arg444_1, (512, ), (1, ))
    assert_size_stride(arg445_1, (512, ), (1, ))
    assert_size_stride(arg446_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg447_1, (512, ), (1, ))
    assert_size_stride(arg448_1, (512, ), (1, ))
    assert_size_stride(arg449_1, (512, ), (1, ))
    assert_size_stride(arg450_1, (512, ), (1, ))
    assert_size_stride(arg451_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg452_1, (256, ), (1, ))
    assert_size_stride(arg453_1, (256, ), (1, ))
    assert_size_stride(arg454_1, (256, ), (1, ))
    assert_size_stride(arg455_1, (256, ), (1, ))
    assert_size_stride(arg456_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg457_1, (256, ), (1, ))
    assert_size_stride(arg458_1, (256, ), (1, ))
    assert_size_stride(arg459_1, (256, ), (1, ))
    assert_size_stride(arg460_1, (256, ), (1, ))
    assert_size_stride(arg461_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg462_1, (512, ), (1, ))
    assert_size_stride(arg463_1, (512, ), (1, ))
    assert_size_stride(arg464_1, (512, ), (1, ))
    assert_size_stride(arg465_1, (512, ), (1, ))
    assert_size_stride(arg466_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg467_1, (256, ), (1, ))
    assert_size_stride(arg468_1, (256, ), (1, ))
    assert_size_stride(arg469_1, (256, ), (1, ))
    assert_size_stride(arg470_1, (256, ), (1, ))
    assert_size_stride(arg471_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg472_1, (256, ), (1, ))
    assert_size_stride(arg473_1, (256, ), (1, ))
    assert_size_stride(arg474_1, (256, ), (1, ))
    assert_size_stride(arg475_1, (256, ), (1, ))
    assert_size_stride(arg476_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg477_1, (512, ), (1, ))
    assert_size_stride(arg478_1, (512, ), (1, ))
    assert_size_stride(arg479_1, (512, ), (1, ))
    assert_size_stride(arg480_1, (512, ), (1, ))
    assert_size_stride(arg481_1, (512, 2816, 1, 1), (2816, 1, 1, 1))
    assert_size_stride(arg482_1, (512, ), (1, ))
    assert_size_stride(arg483_1, (512, ), (1, ))
    assert_size_stride(arg484_1, (512, ), (1, ))
    assert_size_stride(arg485_1, (512, ), (1, ))
    assert_size_stride(arg486_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg487_1, (1024, ), (1, ))
    assert_size_stride(arg488_1, (1024, ), (1, ))
    assert_size_stride(arg489_1, (1024, ), (1, ))
    assert_size_stride(arg490_1, (1024, ), (1, ))
    assert_size_stride(arg491_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg492_1, (512, ), (1, ))
    assert_size_stride(arg493_1, (512, ), (1, ))
    assert_size_stride(arg494_1, (512, ), (1, ))
    assert_size_stride(arg495_1, (512, ), (1, ))
    assert_size_stride(arg496_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg497_1, (512, ), (1, ))
    assert_size_stride(arg498_1, (512, ), (1, ))
    assert_size_stride(arg499_1, (512, ), (1, ))
    assert_size_stride(arg500_1, (512, ), (1, ))
    assert_size_stride(arg501_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg502_1, (1024, ), (1, ))
    assert_size_stride(arg503_1, (1024, ), (1, ))
    assert_size_stride(arg504_1, (1024, ), (1, ))
    assert_size_stride(arg505_1, (1024, ), (1, ))
    assert_size_stride(arg506_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg507_1, (512, ), (1, ))
    assert_size_stride(arg508_1, (512, ), (1, ))
    assert_size_stride(arg509_1, (512, ), (1, ))
    assert_size_stride(arg510_1, (512, ), (1, ))
    assert_size_stride(arg511_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg512_1, (512, ), (1, ))
    assert_size_stride(arg513_1, (512, ), (1, ))
    assert_size_stride(arg514_1, (512, ), (1, ))
    assert_size_stride(arg515_1, (512, ), (1, ))
    assert_size_stride(arg516_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg517_1, (1024, ), (1, ))
    assert_size_stride(arg518_1, (1024, ), (1, ))
    assert_size_stride(arg519_1, (1024, ), (1, ))
    assert_size_stride(arg520_1, (1024, ), (1, ))
    assert_size_stride(arg521_1, (1024, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(arg522_1, (1024, ), (1, ))
    assert_size_stride(arg523_1, (1024, ), (1, ))
    assert_size_stride(arg524_1, (1024, ), (1, ))
    assert_size_stride(arg525_1, (1024, ), (1, ))
    assert_size_stride(arg526_1, (1000, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg527_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((16, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 48, 49, grid=grid(48, 49), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 16, 224, 224), (802816, 1, 3584, 16))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_19, input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg6_1, buf4, 256, 9, grid=grid(256, 9), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [input_19, input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 16, 224, 224), (802816, 1, 3584, 16))
        del buf3
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf7 = empty_strided_cuda((32, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(arg11_1, buf7, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf8 = extern_kernels.convolution(buf6, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del buf6
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf9, arg12_1, arg13_1, arg14_1, arg15_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        # Topologically Sorted Source Nodes: [out_280], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 64, 112, 112), (802816, 1, 7168, 64))
        del arg21_1
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [out_281, out_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf11, arg22_1, arg23_1, arg24_1, arg25_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        buf12 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [out_281, out_282, out_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(arg26_1, buf12, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg26_1
        # Topologically Sorted Source Nodes: [out_281, out_282, out_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf13 = extern_kernels.convolution(buf11, buf12, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 64, 56, 56), (200704, 1, 3584, 64))
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [out_284, out_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf14, arg27_1, arg28_1, arg29_1, arg30_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        # Topologically Sorted Source Nodes: [out_284, out_285, out_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 128, 56, 56), (401408, 1, 7168, 128))
        del arg31_1
        del buf14
        buf16 = empty_strided_cuda((8, 32, 56, 56), (100352, 1, 1792, 32), torch.float32)
        # Topologically Sorted Source Nodes: [bottom_9], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_9.run(buf9, buf16, 802816, grid=grid(802816), stream=stream0)
        # Topologically Sorted Source Nodes: [bottom_9, input_27], Original ATen: [aten.max_pool2d_with_indices, aten.convolution]
        buf17 = extern_kernels.convolution(buf16, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 128, 56, 56), (401408, 1, 7168, 128))
        del arg16_1
        del buf16
        buf18 = buf15; del buf15  # reuse
        buf19 = reinterpret_tensor(buf9, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [out_287, input_28, out_288, out_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf18, arg32_1, arg33_1, arg34_1, arg35_1, buf17, arg17_1, arg18_1, arg19_1, arg20_1, buf19, 3211264, grid=grid(3211264), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        del buf17
        del buf18
        # Topologically Sorted Source Nodes: [out_289, out_290], Original ATen: [aten.relu, aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg36_1
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [out_291, out_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf21, arg37_1, arg38_1, arg39_1, arg40_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        buf22 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [out_291, out_292, out_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(arg41_1, buf22, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg41_1
        # Topologically Sorted Source Nodes: [out_291, out_292, out_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf23 = extern_kernels.convolution(buf21, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del buf21
        del buf22
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [out_294, out_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf24, arg42_1, arg43_1, arg44_1, arg45_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        # Topologically Sorted Source Nodes: [out_294, out_295, out_296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf25 = extern_kernels.convolution(buf24, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (8, 128, 56, 56), (401408, 1, 7168, 128))
        del arg46_1
        buf28 = reinterpret_tensor(buf11, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf11  # reuse
        buf26 = reinterpret_tensor(buf28, (8, 128, 56, 56), (802816, 1, 14336, 256), 0)  # alias
        buf27 = reinterpret_tensor(buf28, (8, 128, 56, 56), (802816, 1, 14336, 256), 128)  # alias
        # Topologically Sorted Source Nodes: [out_297, out_298, out_299, cat_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_11.run(buf25, arg47_1, arg48_1, arg49_1, arg50_1, buf19, buf26, buf27, 3211264, grid=grid(3211264), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        del buf19
        del buf25
        del buf27
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 128, 56, 56), (401408, 1, 7168, 128))
        del arg51_1
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_61, x_62, x_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf30, arg52_1, arg53_1, arg54_1, arg55_1, buf26, 3211264, grid=grid(3211264), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        del buf26
        del buf28
        # Topologically Sorted Source Nodes: [out_300], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 128, 56, 56), (401408, 1, 7168, 128))
        del arg61_1
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [out_301, out_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf32, arg62_1, arg63_1, arg64_1, arg65_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        buf33 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_301, out_302, out_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg66_1, buf33, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg66_1
        # Topologically Sorted Source Nodes: [out_301, out_302, out_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf34 = extern_kernels.convolution(buf32, buf33, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 128, 28, 28), (100352, 1, 3584, 128))
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [out_304, out_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf35, arg67_1, arg68_1, arg69_1, arg70_1, 802816, grid=grid(802816), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        # Topologically Sorted Source Nodes: [out_304, out_305, out_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg71_1
        buf37 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [bottom_12], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_16.run(buf30, buf37, 802816, grid=grid(802816), stream=stream0)
        # Topologically Sorted Source Nodes: [bottom_12, input_29], Original ATen: [aten.max_pool2d_with_indices, aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg56_1
        del buf37
        buf39 = buf36; del buf36  # reuse
        buf40 = reinterpret_tensor(buf24, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [out_307, input_30, out_308, out_309], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf39, arg72_1, arg73_1, arg74_1, arg75_1, buf38, arg57_1, arg58_1, arg59_1, arg60_1, buf40, 1605632, grid=grid(1605632), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        del buf38
        del buf39
        # Topologically Sorted Source Nodes: [out_309, out_310], Original ATen: [aten.relu, aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del arg76_1
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [out_311, out_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf42, arg77_1, arg78_1, arg79_1, arg80_1, 802816, grid=grid(802816), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        buf43 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [out_311, out_312, out_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg81_1, buf43, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg81_1
        # Topologically Sorted Source Nodes: [out_311, out_312, out_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf44 = extern_kernels.convolution(buf42, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del buf42
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [out_314, out_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf45, arg82_1, arg83_1, arg84_1, arg85_1, 802816, grid=grid(802816), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        # Topologically Sorted Source Nodes: [out_314, out_315, out_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf46 = extern_kernels.convolution(buf45, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg86_1
        del buf45
        buf49 = reinterpret_tensor(buf32, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf32  # reuse
        buf47 = reinterpret_tensor(buf49, (8, 256, 28, 28), (401408, 1, 14336, 512), 0)  # alias
        buf48 = reinterpret_tensor(buf49, (8, 256, 28, 28), (401408, 1, 14336, 512), 256)  # alias
        # Topologically Sorted Source Nodes: [out_317, out_318, out_319, cat_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_18.run(buf46, arg87_1, arg88_1, arg89_1, arg90_1, buf40, buf47, buf48, 1605632, grid=grid(1605632), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        del buf40
        del buf46
        del buf48
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg91_1
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_65, x_66, x_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf51, arg92_1, arg93_1, arg94_1, arg95_1, buf47, 1605632, grid=grid(1605632), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf47
        # Topologically Sorted Source Nodes: [out_320], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del arg96_1
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [out_321, out_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf53, arg97_1, arg98_1, arg99_1, arg100_1, 802816, grid=grid(802816), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        buf54 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [out_321, out_322, out_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg101_1, buf54, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg101_1
        # Topologically Sorted Source Nodes: [out_321, out_322, out_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf55 = extern_kernels.convolution(buf53, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del buf53
        buf56 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [out_324, out_325], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf56, arg102_1, arg103_1, arg104_1, arg105_1, 802816, grid=grid(802816), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        # Topologically Sorted Source Nodes: [out_324, out_325, out_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf57 = extern_kernels.convolution(buf56, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg106_1
        del buf56
        buf58 = buf57; del buf57  # reuse
        buf68 = empty_strided_cuda((8, 768, 28, 28), (602112, 1, 21504, 768), torch.float32)
        buf67 = reinterpret_tensor(buf68, (8, 256, 28, 28), (602112, 1, 21504, 768), 512)  # alias
        # Topologically Sorted Source Nodes: [out_327, out_328, out_329, cat_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_20.run(buf58, arg107_1, arg108_1, arg109_1, arg110_1, buf51, buf67, 1605632, grid=grid(1605632), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        del buf51
        # Topologically Sorted Source Nodes: [out_330], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del arg111_1
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [out_331, out_332], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf60, arg112_1, arg113_1, arg114_1, arg115_1, 802816, grid=grid(802816), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        buf61 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [out_331, out_332, out_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg116_1, buf61, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg116_1
        # Topologically Sorted Source Nodes: [out_331, out_332, out_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf62 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del buf60
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [out_334, out_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf63, arg117_1, arg118_1, arg119_1, arg120_1, 802816, grid=grid(802816), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        # Topologically Sorted Source Nodes: [out_334, out_335, out_336], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg121_1
        del buf63
        buf65 = reinterpret_tensor(buf68, (8, 256, 28, 28), (602112, 1, 21504, 768), 0)  # alias
        buf66 = reinterpret_tensor(buf68, (8, 256, 28, 28), (602112, 1, 21504, 768), 256)  # alias
        # Topologically Sorted Source Nodes: [out_337, out_338, out_339, cat_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_21.run(buf64, arg122_1, arg123_1, arg124_1, arg125_1, buf58, buf65, buf66, 1605632, grid=grid(1605632), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        del buf58
        del buf64
        del buf66
        del buf67
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg126_1
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_69, x_70, x_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf70, arg127_1, arg128_1, arg129_1, arg130_1, buf65, 1605632, grid=grid(1605632), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        del buf65
        del buf68
        # Topologically Sorted Source Nodes: [out_340], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del arg131_1
        buf72 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [out_341, out_342], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf72, arg132_1, arg133_1, arg134_1, arg135_1, 802816, grid=grid(802816), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        buf73 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [out_341, out_342, out_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg136_1, buf73, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg136_1
        # Topologically Sorted Source Nodes: [out_341, out_342, out_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf74 = extern_kernels.convolution(buf72, buf73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del buf72
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [out_344, out_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf75, arg137_1, arg138_1, arg139_1, arg140_1, 802816, grid=grid(802816), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        # Topologically Sorted Source Nodes: [out_344, out_345, out_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg141_1
        del buf75
        buf77 = buf76; del buf76  # reuse
        buf107 = empty_strided_cuda((8, 1152, 28, 28), (903168, 1, 32256, 1152), torch.float32)
        buf105 = reinterpret_tensor(buf107, (8, 256, 28, 28), (903168, 1, 32256, 1152), 640)  # alias
        # Topologically Sorted Source Nodes: [out_347, out_348, out_349, cat_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_23.run(buf77, arg142_1, arg143_1, arg144_1, arg145_1, buf70, buf105, 1605632, grid=grid(1605632), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        del buf70
        # Topologically Sorted Source Nodes: [out_350], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del arg146_1
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [out_351, out_352], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf79, arg147_1, arg148_1, arg149_1, arg150_1, 802816, grid=grid(802816), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        buf80 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [out_351, out_352, out_353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg151_1, buf80, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg151_1
        # Topologically Sorted Source Nodes: [out_351, out_352, out_353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf81 = extern_kernels.convolution(buf79, buf80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del buf79
        buf82 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [out_354, out_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf82, arg152_1, arg153_1, arg154_1, arg155_1, 802816, grid=grid(802816), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        # Topologically Sorted Source Nodes: [out_354, out_355, out_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf83 = extern_kernels.convolution(buf82, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg156_1
        del buf82
        buf86 = buf49; del buf49  # reuse
        buf84 = reinterpret_tensor(buf86, (8, 256, 28, 28), (401408, 1, 14336, 512), 0)  # alias
        buf85 = reinterpret_tensor(buf86, (8, 256, 28, 28), (401408, 1, 14336, 512), 256)  # alias
        # Topologically Sorted Source Nodes: [out_357, out_358, out_359, cat_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_18.run(buf83, arg157_1, arg158_1, arg159_1, arg160_1, buf77, buf84, buf85, 1605632, grid=grid(1605632), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        del buf77
        del buf83
        del buf85
        # Topologically Sorted Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg161_1
        buf88 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_73, x_74, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf88, arg162_1, arg163_1, arg164_1, arg165_1, buf84, 1605632, grid=grid(1605632), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        del buf84
        del buf86
        # Topologically Sorted Source Nodes: [out_360], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del arg166_1
        buf90 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [out_361, out_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf90, arg167_1, arg168_1, arg169_1, arg170_1, 802816, grid=grid(802816), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        buf91 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [out_361, out_362, out_363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg171_1, buf91, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg171_1
        # Topologically Sorted Source Nodes: [out_361, out_362, out_363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf92 = extern_kernels.convolution(buf90, buf91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del buf90
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [out_364, out_365], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf93, arg172_1, arg173_1, arg174_1, arg175_1, 802816, grid=grid(802816), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        # Topologically Sorted Source Nodes: [out_364, out_365, out_366], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg176_1
        del buf93
        buf95 = buf94; del buf94  # reuse
        buf106 = reinterpret_tensor(buf107, (8, 256, 28, 28), (903168, 1, 32256, 1152), 896)  # alias
        # Topologically Sorted Source Nodes: [out_367, out_368, out_369, cat_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_23.run(buf95, arg177_1, arg178_1, arg179_1, arg180_1, buf88, buf106, 1605632, grid=grid(1605632), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        del buf88
        # Topologically Sorted Source Nodes: [out_370], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del arg181_1
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [out_371, out_372], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf97, arg182_1, arg183_1, arg184_1, arg185_1, 802816, grid=grid(802816), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        buf98 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [out_371, out_372, out_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg186_1, buf98, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg186_1
        # Topologically Sorted Source Nodes: [out_371, out_372, out_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf99 = extern_kernels.convolution(buf97, buf98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del buf97
        del buf98
        buf100 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [out_374, out_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf100, arg187_1, arg188_1, arg189_1, arg190_1, 802816, grid=grid(802816), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        # Topologically Sorted Source Nodes: [out_374, out_375, out_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf101 = extern_kernels.convolution(buf100, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg191_1
        buf102 = reinterpret_tensor(buf107, (8, 256, 28, 28), (903168, 1, 32256, 1152), 0)  # alias
        buf103 = reinterpret_tensor(buf107, (8, 256, 28, 28), (903168, 1, 32256, 1152), 256)  # alias
        # Topologically Sorted Source Nodes: [out_377, out_378, out_379, cat_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_24.run(buf101, arg192_1, arg193_1, arg194_1, arg195_1, buf95, buf102, buf103, 1605632, grid=grid(1605632), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        del buf101
        del buf95
        buf104 = reinterpret_tensor(buf107, (8, 128, 28, 28), (903168, 1, 32256, 1152), 512)  # alias
        # Topologically Sorted Source Nodes: [bottom_10], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_25.run(buf30, buf104, 802816, grid=grid(802816), stream=stream0)
        del buf103
        del buf104
        del buf105
        del buf106
        # Topologically Sorted Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg196_1
        buf109 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_77, x_78, x_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26.run(buf109, arg197_1, arg198_1, arg199_1, arg200_1, buf102, 1605632, grid=grid(1605632), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        del buf102
        del buf107
        # Topologically Sorted Source Nodes: [out_380], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg206_1
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [out_381, out_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf111, arg207_1, arg208_1, arg209_1, arg210_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        buf112 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_381, out_382, out_383], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg211_1, buf112, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg211_1
        # Topologically Sorted Source Nodes: [out_381, out_382, out_383], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf113 = extern_kernels.convolution(buf111, buf112, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 256, 14, 14), (50176, 1, 3584, 256))
        buf114 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [out_384, out_385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf114, arg212_1, arg213_1, arg214_1, arg215_1, 401408, grid=grid(401408), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        # Topologically Sorted Source Nodes: [out_384, out_385, out_386], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf115 = extern_kernels.convolution(buf114, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg216_1
        buf116 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [bottom_16], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_30.run(buf109, buf116, 401408, grid=grid(401408), stream=stream0)
        # Topologically Sorted Source Nodes: [bottom_16, input_31], Original ATen: [aten.max_pool2d_with_indices, aten.convolution]
        buf117 = extern_kernels.convolution(buf116, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg201_1
        del buf116
        buf118 = buf115; del buf115  # reuse
        buf119 = reinterpret_tensor(buf100, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [out_387, input_32, out_388, out_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31.run(buf118, arg217_1, arg218_1, arg219_1, arg220_1, buf117, arg202_1, arg203_1, arg204_1, arg205_1, buf119, 802816, grid=grid(802816), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        del buf117
        del buf118
        # Topologically Sorted Source Nodes: [out_389, out_390], Original ATen: [aten.relu, aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg221_1
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [out_391, out_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf121, arg222_1, arg223_1, arg224_1, arg225_1, 401408, grid=grid(401408), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        buf122 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [out_391, out_392, out_393], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg226_1, buf122, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg226_1
        # Topologically Sorted Source Nodes: [out_391, out_392, out_393], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf123 = extern_kernels.convolution(buf121, buf122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf121
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [out_394, out_395], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf124, arg227_1, arg228_1, arg229_1, arg230_1, 401408, grid=grid(401408), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        # Topologically Sorted Source Nodes: [out_394, out_395, out_396], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf125 = extern_kernels.convolution(buf124, arg231_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg231_1
        del buf124
        buf128 = reinterpret_tensor(buf111, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf111  # reuse
        buf126 = reinterpret_tensor(buf128, (8, 512, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
        buf127 = reinterpret_tensor(buf128, (8, 512, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
        # Topologically Sorted Source Nodes: [out_397, out_398, out_399, cat_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32.run(buf125, arg232_1, arg233_1, arg234_1, arg235_1, buf119, buf126, buf127, 802816, grid=grid(802816), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        del buf119
        del buf125
        del buf127
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg236_1
        buf130 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [x_81, x_82, x_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf130, arg237_1, arg238_1, arg239_1, arg240_1, buf126, 802816, grid=grid(802816), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        del buf126
        # Topologically Sorted Source Nodes: [out_400], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg241_1
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [out_401, out_402], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf132, arg242_1, arg243_1, arg244_1, arg245_1, 401408, grid=grid(401408), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        buf133 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [out_401, out_402, out_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg246_1, buf133, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg246_1
        # Topologically Sorted Source Nodes: [out_401, out_402, out_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf134 = extern_kernels.convolution(buf132, buf133, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf132
        buf135 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [out_404, out_405], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf135, arg247_1, arg248_1, arg249_1, arg250_1, 401408, grid=grid(401408), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        # Topologically Sorted Source Nodes: [out_404, out_405, out_406], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg251_1
        del buf135
        buf137 = buf136; del buf136  # reuse
        buf147 = empty_strided_cuda((8, 1536, 14, 14), (301056, 1, 21504, 1536), torch.float32)
        buf146 = reinterpret_tensor(buf147, (8, 512, 14, 14), (301056, 1, 21504, 1536), 1024)  # alias
        # Topologically Sorted Source Nodes: [out_407, out_408, out_409, cat_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_34.run(buf137, arg252_1, arg253_1, arg254_1, arg255_1, buf130, buf146, 802816, grid=grid(802816), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        del buf130
        # Topologically Sorted Source Nodes: [out_410], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg256_1
        buf139 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [out_411, out_412], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf139, arg257_1, arg258_1, arg259_1, arg260_1, 401408, grid=grid(401408), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        buf140 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [out_411, out_412, out_413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg261_1, buf140, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg261_1
        # Topologically Sorted Source Nodes: [out_411, out_412, out_413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf141 = extern_kernels.convolution(buf139, buf140, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf139
        buf142 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [out_414, out_415], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf142, arg262_1, arg263_1, arg264_1, arg265_1, 401408, grid=grid(401408), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        # Topologically Sorted Source Nodes: [out_414, out_415, out_416], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf143 = extern_kernels.convolution(buf142, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg266_1
        del buf142
        buf144 = reinterpret_tensor(buf147, (8, 512, 14, 14), (301056, 1, 21504, 1536), 0)  # alias
        buf145 = reinterpret_tensor(buf147, (8, 512, 14, 14), (301056, 1, 21504, 1536), 512)  # alias
        # Topologically Sorted Source Nodes: [out_417, out_418, out_419, cat_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_35.run(buf143, arg267_1, arg268_1, arg269_1, arg270_1, buf137, buf144, buf145, 802816, grid=grid(802816), stream=stream0)
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        del buf137
        del buf143
        del buf145
        del buf146
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg271_1
        buf149 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_85, x_86, x_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf149, arg272_1, arg273_1, arg274_1, arg275_1, buf144, 802816, grid=grid(802816), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        del buf144
        # Topologically Sorted Source Nodes: [out_420], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg276_1
        buf151 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [out_421, out_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf151, arg277_1, arg278_1, arg279_1, arg280_1, 401408, grid=grid(401408), stream=stream0)
        del arg277_1
        del arg278_1
        del arg279_1
        del arg280_1
        buf152 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [out_421, out_422, out_423], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg281_1, buf152, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg281_1
        # Topologically Sorted Source Nodes: [out_421, out_422, out_423], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf153 = extern_kernels.convolution(buf151, buf152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf151
        buf154 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [out_424, out_425], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf154, arg282_1, arg283_1, arg284_1, arg285_1, 401408, grid=grid(401408), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        # Topologically Sorted Source Nodes: [out_424, out_425, out_426], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf155 = extern_kernels.convolution(buf154, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg286_1
        del buf154
        buf156 = buf155; del buf155  # reuse
        buf185 = reinterpret_tensor(buf30, (8, 2048, 14, 14), (401408, 1, 28672, 2048), 0); del buf30  # reuse
        buf183 = reinterpret_tensor(buf185, (8, 512, 14, 14), (401408, 1, 28672, 2048), 1024)  # alias
        # Topologically Sorted Source Nodes: [out_427, out_428, out_429, cat_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_37.run(buf156, arg287_1, arg288_1, arg289_1, arg290_1, buf149, buf183, 802816, grid=grid(802816), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        del buf149
        # Topologically Sorted Source Nodes: [out_430], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg291_1
        buf158 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [out_431, out_432], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf158, arg292_1, arg293_1, arg294_1, arg295_1, 401408, grid=grid(401408), stream=stream0)
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        buf159 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [out_431, out_432, out_433], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg296_1, buf159, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg296_1
        # Topologically Sorted Source Nodes: [out_431, out_432, out_433], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf160 = extern_kernels.convolution(buf158, buf159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf158
        buf161 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [out_434, out_435], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf161, arg297_1, arg298_1, arg299_1, arg300_1, 401408, grid=grid(401408), stream=stream0)
        del arg297_1
        del arg298_1
        del arg299_1
        del arg300_1
        # Topologically Sorted Source Nodes: [out_434, out_435, out_436], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf162 = extern_kernels.convolution(buf161, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg301_1
        del buf161
        buf165 = buf128; del buf128  # reuse
        buf163 = reinterpret_tensor(buf165, (8, 512, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
        buf164 = reinterpret_tensor(buf165, (8, 512, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
        # Topologically Sorted Source Nodes: [out_437, out_438, out_439, cat_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32.run(buf162, arg302_1, arg303_1, arg304_1, arg305_1, buf156, buf163, buf164, 802816, grid=grid(802816), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        del buf156
        del buf162
        del buf164
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg306_1
        buf167 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [x_89, x_90, x_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf167, arg307_1, arg308_1, arg309_1, arg310_1, buf163, 802816, grid=grid(802816), stream=stream0)
        del arg307_1
        del arg308_1
        del arg309_1
        del arg310_1
        del buf163
        # Topologically Sorted Source Nodes: [out_440], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, arg311_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg311_1
        buf169 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [out_441, out_442], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf169, arg312_1, arg313_1, arg314_1, arg315_1, 401408, grid=grid(401408), stream=stream0)
        del arg312_1
        del arg313_1
        del arg314_1
        del arg315_1
        buf170 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [out_441, out_442, out_443], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg316_1, buf170, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg316_1
        # Topologically Sorted Source Nodes: [out_441, out_442, out_443], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf171 = extern_kernels.convolution(buf169, buf170, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf169
        buf172 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [out_444, out_445], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf172, arg317_1, arg318_1, arg319_1, arg320_1, 401408, grid=grid(401408), stream=stream0)
        del arg317_1
        del arg318_1
        del arg319_1
        del arg320_1
        # Topologically Sorted Source Nodes: [out_444, out_445, out_446], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf173 = extern_kernels.convolution(buf172, arg321_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg321_1
        del buf172
        buf174 = buf173; del buf173  # reuse
        buf184 = reinterpret_tensor(buf185, (8, 512, 14, 14), (401408, 1, 28672, 2048), 1536)  # alias
        # Topologically Sorted Source Nodes: [out_447, out_448, out_449, cat_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_37.run(buf174, arg322_1, arg323_1, arg324_1, arg325_1, buf167, buf184, 802816, grid=grid(802816), stream=stream0)
        del arg322_1
        del arg323_1
        del arg324_1
        del arg325_1
        del buf167
        # Topologically Sorted Source Nodes: [out_450], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, arg326_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg326_1
        buf176 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [out_451, out_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf176, arg327_1, arg328_1, arg329_1, arg330_1, 401408, grid=grid(401408), stream=stream0)
        del arg327_1
        del arg328_1
        del arg329_1
        del arg330_1
        buf177 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [out_451, out_452, out_453], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg331_1, buf177, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg331_1
        # Topologically Sorted Source Nodes: [out_451, out_452, out_453], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf176
        buf179 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [out_454, out_455], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf179, arg332_1, arg333_1, arg334_1, arg335_1, 401408, grid=grid(401408), stream=stream0)
        del arg332_1
        del arg333_1
        del arg334_1
        del arg335_1
        # Topologically Sorted Source Nodes: [out_454, out_455, out_456], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf180 = extern_kernels.convolution(buf179, arg336_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg336_1
        del buf179
        buf181 = reinterpret_tensor(buf185, (8, 512, 14, 14), (401408, 1, 28672, 2048), 0)  # alias
        buf182 = reinterpret_tensor(buf185, (8, 512, 14, 14), (401408, 1, 28672, 2048), 512)  # alias
        # Topologically Sorted Source Nodes: [out_457, out_458, out_459, cat_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_38.run(buf180, arg337_1, arg338_1, arg339_1, arg340_1, buf174, buf181, buf182, 802816, grid=grid(802816), stream=stream0)
        del arg337_1
        del arg338_1
        del arg339_1
        del arg340_1
        del buf174
        del buf180
        del buf182
        del buf183
        del buf184
        # Topologically Sorted Source Nodes: [x_92], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, arg341_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg341_1
        buf187 = buf186; del buf186  # reuse
        buf262 = empty_strided_cuda((8, 2816, 14, 14), (551936, 1, 39424, 2816), torch.float32)
        buf259 = reinterpret_tensor(buf262, (8, 512, 14, 14), (551936, 1, 39424, 2816), 1280)  # alias
        # Topologically Sorted Source Nodes: [x_93, x_94, x_95, cat_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_39.run(buf187, arg342_1, arg343_1, arg344_1, arg345_1, buf181, buf259, 802816, grid=grid(802816), stream=stream0)
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        del buf181
        del buf185
        # Topologically Sorted Source Nodes: [out_460], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, arg346_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg346_1
        buf189 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [out_461, out_462], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf189, arg347_1, arg348_1, arg349_1, arg350_1, 401408, grid=grid(401408), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        del arg350_1
        buf190 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [out_461, out_462, out_463], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg351_1, buf190, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg351_1
        # Topologically Sorted Source Nodes: [out_461, out_462, out_463], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf191 = extern_kernels.convolution(buf189, buf190, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf189
        buf192 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [out_464, out_465], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf192, arg352_1, arg353_1, arg354_1, arg355_1, 401408, grid=grid(401408), stream=stream0)
        del arg352_1
        del arg353_1
        del arg354_1
        del arg355_1
        # Topologically Sorted Source Nodes: [out_464, out_465, out_466], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf193 = extern_kernels.convolution(buf192, arg356_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg356_1
        del buf192
        buf194 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [out_467, out_468, out_469], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40.run(buf194, buf193, arg357_1, arg358_1, arg359_1, arg360_1, 802816, grid=grid(802816), stream=stream0)
        del arg357_1
        del arg358_1
        del arg359_1
        del arg360_1
        del buf193
        # Topologically Sorted Source Nodes: [out_470], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, arg361_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg361_1
        buf196 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [out_471, out_472], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf196, arg362_1, arg363_1, arg364_1, arg365_1, 401408, grid=grid(401408), stream=stream0)
        del arg362_1
        del arg363_1
        del arg364_1
        del arg365_1
        buf197 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [out_471, out_472, out_473], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg366_1, buf197, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg366_1
        # Topologically Sorted Source Nodes: [out_471, out_472, out_473], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf198 = extern_kernels.convolution(buf196, buf197, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf196
        buf199 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [out_474, out_475], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf199, arg367_1, arg368_1, arg369_1, arg370_1, 401408, grid=grid(401408), stream=stream0)
        del arg367_1
        del arg368_1
        del arg369_1
        del arg370_1
        # Topologically Sorted Source Nodes: [out_474, out_475, out_476], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf200 = extern_kernels.convolution(buf199, arg371_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg371_1
        del buf199
        buf203 = buf165; del buf165  # reuse
        buf201 = reinterpret_tensor(buf203, (8, 512, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
        buf202 = reinterpret_tensor(buf203, (8, 512, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
        # Topologically Sorted Source Nodes: [out_477, out_478, out_479, cat_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32.run(buf200, arg372_1, arg373_1, arg374_1, arg375_1, buf194, buf201, buf202, 802816, grid=grid(802816), stream=stream0)
        del arg372_1
        del arg373_1
        del arg374_1
        del arg375_1
        del buf194
        del buf200
        del buf202
        # Topologically Sorted Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, arg376_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg376_1
        buf205 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_97, x_98, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf205, arg377_1, arg378_1, arg379_1, arg380_1, buf201, 802816, grid=grid(802816), stream=stream0)
        del arg377_1
        del arg378_1
        del arg379_1
        del arg380_1
        del buf201
        # Topologically Sorted Source Nodes: [out_480], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, arg381_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg381_1
        buf207 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [out_481, out_482], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf207, arg382_1, arg383_1, arg384_1, arg385_1, 401408, grid=grid(401408), stream=stream0)
        del arg382_1
        del arg383_1
        del arg384_1
        del arg385_1
        buf208 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [out_481, out_482, out_483], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg386_1, buf208, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg386_1
        # Topologically Sorted Source Nodes: [out_481, out_482, out_483], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf209 = extern_kernels.convolution(buf207, buf208, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf207
        buf210 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [out_484, out_485], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf210, arg387_1, arg388_1, arg389_1, arg390_1, 401408, grid=grid(401408), stream=stream0)
        del arg387_1
        del arg388_1
        del arg389_1
        del arg390_1
        # Topologically Sorted Source Nodes: [out_484, out_485, out_486], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf211 = extern_kernels.convolution(buf210, arg391_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg391_1
        del buf210
        buf212 = buf211; del buf211  # reuse
        buf222 = buf147; del buf147  # reuse
        buf221 = reinterpret_tensor(buf222, (8, 512, 14, 14), (301056, 1, 21504, 1536), 1024)  # alias
        # Topologically Sorted Source Nodes: [out_487, out_488, out_489, cat_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_34.run(buf212, arg392_1, arg393_1, arg394_1, arg395_1, buf205, buf221, 802816, grid=grid(802816), stream=stream0)
        del arg392_1
        del arg393_1
        del arg394_1
        del arg395_1
        del buf205
        # Topologically Sorted Source Nodes: [out_490], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, arg396_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg396_1
        buf214 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [out_491, out_492], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf214, arg397_1, arg398_1, arg399_1, arg400_1, 401408, grid=grid(401408), stream=stream0)
        del arg397_1
        del arg398_1
        del arg399_1
        del arg400_1
        buf215 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [out_491, out_492, out_493], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg401_1, buf215, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg401_1
        # Topologically Sorted Source Nodes: [out_491, out_492, out_493], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf216 = extern_kernels.convolution(buf214, buf215, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf214
        buf217 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [out_494, out_495], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf217, arg402_1, arg403_1, arg404_1, arg405_1, 401408, grid=grid(401408), stream=stream0)
        del arg402_1
        del arg403_1
        del arg404_1
        del arg405_1
        # Topologically Sorted Source Nodes: [out_494, out_495, out_496], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf218 = extern_kernels.convolution(buf217, arg406_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg406_1
        del buf217
        buf219 = reinterpret_tensor(buf222, (8, 512, 14, 14), (301056, 1, 21504, 1536), 0)  # alias
        buf220 = reinterpret_tensor(buf222, (8, 512, 14, 14), (301056, 1, 21504, 1536), 512)  # alias
        # Topologically Sorted Source Nodes: [out_497, out_498, out_499, cat_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_35.run(buf218, arg407_1, arg408_1, arg409_1, arg410_1, buf212, buf219, buf220, 802816, grid=grid(802816), stream=stream0)
        del arg407_1
        del arg408_1
        del arg409_1
        del arg410_1
        del buf212
        del buf218
        del buf220
        del buf221
        # Topologically Sorted Source Nodes: [x_100], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, arg411_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg411_1
        buf224 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [x_101, x_102, x_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf224, arg412_1, arg413_1, arg414_1, arg415_1, buf219, 802816, grid=grid(802816), stream=stream0)
        del arg412_1
        del arg413_1
        del arg414_1
        del arg415_1
        del buf219
        del buf222
        # Topologically Sorted Source Nodes: [out_500], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, arg416_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg416_1
        buf226 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [out_501, out_502], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf226, arg417_1, arg418_1, arg419_1, arg420_1, 401408, grid=grid(401408), stream=stream0)
        del arg417_1
        del arg418_1
        del arg419_1
        del arg420_1
        buf227 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [out_501, out_502, out_503], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg421_1, buf227, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg421_1
        # Topologically Sorted Source Nodes: [out_501, out_502, out_503], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf228 = extern_kernels.convolution(buf226, buf227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf226
        buf229 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [out_504, out_505], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf229, arg422_1, arg423_1, arg424_1, arg425_1, 401408, grid=grid(401408), stream=stream0)
        del arg422_1
        del arg423_1
        del arg424_1
        del arg425_1
        # Topologically Sorted Source Nodes: [out_504, out_505, out_506], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf230 = extern_kernels.convolution(buf229, arg426_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg426_1
        del buf229
        buf231 = buf230; del buf230  # reuse
        buf260 = reinterpret_tensor(buf262, (8, 512, 14, 14), (551936, 1, 39424, 2816), 1792)  # alias
        # Topologically Sorted Source Nodes: [out_507, out_508, out_509, cat_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_41.run(buf231, arg427_1, arg428_1, arg429_1, arg430_1, buf224, buf260, 802816, grid=grid(802816), stream=stream0)
        del arg427_1
        del arg428_1
        del arg429_1
        del arg430_1
        del buf224
        # Topologically Sorted Source Nodes: [out_510], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, arg431_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg431_1
        buf233 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [out_511, out_512], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf233, arg432_1, arg433_1, arg434_1, arg435_1, 401408, grid=grid(401408), stream=stream0)
        del arg432_1
        del arg433_1
        del arg434_1
        del arg435_1
        buf234 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [out_511, out_512, out_513], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg436_1, buf234, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg436_1
        # Topologically Sorted Source Nodes: [out_511, out_512, out_513], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf235 = extern_kernels.convolution(buf233, buf234, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf233
        buf236 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [out_514, out_515], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf236, arg437_1, arg438_1, arg439_1, arg440_1, 401408, grid=grid(401408), stream=stream0)
        del arg437_1
        del arg438_1
        del arg439_1
        del arg440_1
        # Topologically Sorted Source Nodes: [out_514, out_515, out_516], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf237 = extern_kernels.convolution(buf236, arg441_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg441_1
        del buf236
        buf240 = buf203; del buf203  # reuse
        buf238 = reinterpret_tensor(buf240, (8, 512, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
        buf239 = reinterpret_tensor(buf240, (8, 512, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
        # Topologically Sorted Source Nodes: [out_517, out_518, out_519, cat_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32.run(buf237, arg442_1, arg443_1, arg444_1, arg445_1, buf231, buf238, buf239, 802816, grid=grid(802816), stream=stream0)
        del arg442_1
        del arg443_1
        del arg444_1
        del arg445_1
        del buf231
        del buf237
        del buf239
        # Topologically Sorted Source Nodes: [x_104], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, arg446_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg446_1
        buf242 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [x_105, x_106, x_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf242, arg447_1, arg448_1, arg449_1, arg450_1, buf238, 802816, grid=grid(802816), stream=stream0)
        del arg447_1
        del arg448_1
        del arg449_1
        del arg450_1
        del buf238
        del buf240
        # Topologically Sorted Source Nodes: [out_520], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, arg451_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg451_1
        buf244 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [out_521, out_522], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf244, arg452_1, arg453_1, arg454_1, arg455_1, 401408, grid=grid(401408), stream=stream0)
        del arg452_1
        del arg453_1
        del arg454_1
        del arg455_1
        buf245 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [out_521, out_522, out_523], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg456_1, buf245, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg456_1
        # Topologically Sorted Source Nodes: [out_521, out_522, out_523], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf246 = extern_kernels.convolution(buf244, buf245, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf244
        buf247 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [out_524, out_525], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf247, arg457_1, arg458_1, arg459_1, arg460_1, 401408, grid=grid(401408), stream=stream0)
        del arg457_1
        del arg458_1
        del arg459_1
        del arg460_1
        # Topologically Sorted Source Nodes: [out_524, out_525, out_526], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf248 = extern_kernels.convolution(buf247, arg461_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg461_1
        del buf247
        buf249 = buf248; del buf248  # reuse
        buf261 = reinterpret_tensor(buf262, (8, 512, 14, 14), (551936, 1, 39424, 2816), 2304)  # alias
        # Topologically Sorted Source Nodes: [out_527, out_528, out_529, cat_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_41.run(buf249, arg462_1, arg463_1, arg464_1, arg465_1, buf242, buf261, 802816, grid=grid(802816), stream=stream0)
        del arg462_1
        del arg463_1
        del arg464_1
        del arg465_1
        del buf242
        # Topologically Sorted Source Nodes: [out_530], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, arg466_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del arg466_1
        buf251 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [out_531, out_532], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf251, arg467_1, arg468_1, arg469_1, arg470_1, 401408, grid=grid(401408), stream=stream0)
        del arg467_1
        del arg468_1
        del arg469_1
        del arg470_1
        buf252 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [out_531, out_532, out_533], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg471_1, buf252, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg471_1
        # Topologically Sorted Source Nodes: [out_531, out_532, out_533], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf253 = extern_kernels.convolution(buf251, buf252, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 256, 14, 14), (50176, 1, 3584, 256))
        del buf251
        del buf252
        buf254 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [out_534, out_535], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf254, arg472_1, arg473_1, arg474_1, arg475_1, 401408, grid=grid(401408), stream=stream0)
        del arg472_1
        del arg473_1
        del arg474_1
        del arg475_1
        # Topologically Sorted Source Nodes: [out_534, out_535, out_536], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf255 = extern_kernels.convolution(buf254, arg476_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg476_1
        buf256 = reinterpret_tensor(buf262, (8, 512, 14, 14), (551936, 1, 39424, 2816), 0)  # alias
        buf257 = reinterpret_tensor(buf262, (8, 512, 14, 14), (551936, 1, 39424, 2816), 512)  # alias
        # Topologically Sorted Source Nodes: [out_537, out_538, out_539, cat_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_42.run(buf255, arg477_1, arg478_1, arg479_1, arg480_1, buf249, buf256, buf257, 802816, grid=grid(802816), stream=stream0)
        del arg477_1
        del arg478_1
        del arg479_1
        del arg480_1
        del buf249
        del buf255
        buf258 = reinterpret_tensor(buf262, (8, 256, 14, 14), (551936, 1, 39424, 2816), 1024)  # alias
        # Topologically Sorted Source Nodes: [bottom_13], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_43.run(buf109, buf258, 401408, grid=grid(401408), stream=stream0)
        del buf109
        del buf257
        del buf258
        del buf259
        del buf260
        del buf261
        # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, arg481_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg481_1
        buf264 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [x_109, x_110, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf264, arg482_1, arg483_1, arg484_1, arg485_1, buf256, 802816, grid=grid(802816), stream=stream0)
        del arg482_1
        del arg483_1
        del arg484_1
        del arg485_1
        del buf256
        del buf262
        buf265 = empty_strided_cuda((8, 512, 7, 7), (25088, 1, 3584, 512), torch.float32)
        buf284 = empty_strided_cuda((8, 2560, 7, 7), (125440, 1, 17920, 2560), torch.float32)
        buf283 = reinterpret_tensor(buf284, (8, 512, 7, 7), (125440, 1, 17920, 2560), 2048)  # alias
        # Topologically Sorted Source Nodes: [bottom_17, cat_27], Original ATen: [aten.max_pool2d_with_indices, aten.cat]
        triton_poi_fused_cat_max_pool2d_with_indices_45.run(buf264, buf265, buf283, 200704, grid=grid(200704), stream=stream0)
        # Topologically Sorted Source Nodes: [out_540], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf264, arg491_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg491_1
        del buf264
        buf267 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [out_541, out_542], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf267, arg492_1, arg493_1, arg494_1, arg495_1, 802816, grid=grid(802816), stream=stream0)
        del arg492_1
        del arg493_1
        del arg494_1
        del arg495_1
        buf268 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_541, out_542, out_543], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47.run(arg496_1, buf268, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg496_1
        # Topologically Sorted Source Nodes: [out_541, out_542, out_543], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf269 = extern_kernels.convolution(buf267, buf268, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (8, 512, 7, 7), (25088, 1, 3584, 512))
        del buf267
        buf270 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [out_544, out_545], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf270, arg497_1, arg498_1, arg499_1, arg500_1, 200704, grid=grid(200704), stream=stream0)
        del arg497_1
        del arg498_1
        del arg499_1
        del arg500_1
        # Topologically Sorted Source Nodes: [out_544, out_545, out_546], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf271 = extern_kernels.convolution(buf270, arg501_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
        del arg501_1
        del buf270
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf265, arg486_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
        del arg486_1
        del buf265
        buf273 = buf271; del buf271  # reuse
        buf274 = reinterpret_tensor(buf254, (8, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [out_547, input_34, out_548, out_549], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_49.run(buf273, arg502_1, arg503_1, arg504_1, arg505_1, buf272, arg487_1, arg488_1, arg489_1, arg490_1, buf274, 401408, grid=grid(401408), stream=stream0)
        del arg487_1
        del arg488_1
        del arg489_1
        del arg490_1
        del arg502_1
        del arg503_1
        del arg504_1
        del arg505_1
        del buf272
        del buf273
        # Topologically Sorted Source Nodes: [out_549, out_550], Original ATen: [aten.relu, aten.convolution]
        buf275 = extern_kernels.convolution(buf274, arg506_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 512, 7, 7), (25088, 1, 3584, 512))
        del arg506_1
        buf276 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [out_551, out_552], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf276, arg507_1, arg508_1, arg509_1, arg510_1, 200704, grid=grid(200704), stream=stream0)
        del arg507_1
        del arg508_1
        del arg509_1
        del arg510_1
        buf277 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [out_551, out_552, out_553], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47.run(arg511_1, buf277, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg511_1
        # Topologically Sorted Source Nodes: [out_551, out_552, out_553], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf278 = extern_kernels.convolution(buf276, buf277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 512, 7, 7), (25088, 1, 3584, 512))
        del buf276
        del buf277
        buf279 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [out_554, out_555], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf279, arg512_1, arg513_1, arg514_1, arg515_1, 200704, grid=grid(200704), stream=stream0)
        del arg512_1
        del arg513_1
        del arg514_1
        del arg515_1
        # Topologically Sorted Source Nodes: [out_554, out_555, out_556], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf280 = extern_kernels.convolution(buf279, arg516_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
        del arg516_1
        del buf279
        buf281 = reinterpret_tensor(buf284, (8, 1024, 7, 7), (125440, 1, 17920, 2560), 0)  # alias
        buf282 = reinterpret_tensor(buf284, (8, 1024, 7, 7), (125440, 1, 17920, 2560), 1024)  # alias
        # Topologically Sorted Source Nodes: [out_557, out_558, out_559, cat_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.cat]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_50.run(buf280, arg517_1, arg518_1, arg519_1, arg520_1, buf274, buf281, buf282, 401408, grid=grid(401408), stream=stream0)
        del arg517_1
        del arg518_1
        del arg519_1
        del arg520_1
        del buf274
        del buf280
        del buf282
        del buf283
        # Topologically Sorted Source Nodes: [x_112], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, arg521_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
        del arg521_1
        buf287 = empty_strided_cuda((8, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_113, x_114, x_115, x_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51.run(buf285, arg522_1, arg523_1, arg524_1, arg525_1, buf281, buf287, 8192, 49, grid=grid(8192), stream=stream0)
        del arg522_1
        del arg523_1
        del arg524_1
        del arg525_1
        del buf281
        del buf284
        del buf285
        # Topologically Sorted Source Nodes: [x_113, x_114, x_115, x_116, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean, aten.convolution]
        buf288 = extern_kernels.convolution(buf287, arg526_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 1000, 1, 1), (1000, 1, 1, 1))
        del arg526_1
        del buf287
        buf289 = reinterpret_tensor(buf288, (8, 1000, 1, 1), (1000, 1, 8000, 8000), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [x_113, x_114, x_115, x_116, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_52.run(buf289, arg527_1, 8000, grid=grid(8000), stream=stream0)
        del arg527_1
    return (reinterpret_tensor(buf289, (8, 1000), (1000, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((256, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((512, 2816, 1, 1), (2816, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((1024, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((1000, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dla102', benchmark_compiled_module)
