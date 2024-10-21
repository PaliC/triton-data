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
# Topologically Sorted Source Nodes: [x_228], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_228 => convolution_67
# Graph fragment:
#   %convolution_67 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
# Topologically Sorted Source Nodes: [x_228], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_228 => convolution_67
# Graph fragment:
#   %convolution_67 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/gz/cgzbeqcb7ne3tbwzsbhstvfdwyiqhg4euybgogh44w2pxe2nf7lz.py
# Topologically Sorted Source Nodes: [x_229, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_229 => add_158, mul_269, mul_270, sub_67
#   x_230 => gt_67, mul_271, where_67
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_67, %unsqueeze_537), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_269, %unsqueeze_541), kwargs = {})
#   %add_158 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_270, %unsqueeze_543), kwargs = {})
#   %gt_67 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_158, 0), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_158, 0.01), kwargs = {})
#   %where_67 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_67, %add_158, %mul_271), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rc/crcifzyxfk2pougti6o4mx4iqtljp5jqwxxanl5lfd6rpuofgeeb.py
# Topologically Sorted Source Nodes: [x_230, x_231], Original ATen: [aten.leaky_relu, aten.convolution]
# Source node to ATen node mapping:
#   x_230 => gt_67, mul_271, where_67
#   x_231 => convolution_68
# Graph fragment:
#   %gt_67 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_158, 0), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_158, 0.01), kwargs = {})
#   %where_67 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_67, %add_158, %mul_271), kwargs = {})
#   %convolution_68 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_67, %arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_leaky_relu_3 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/hc/chcaao5sp47ildfykwpckhdqkjuriq6dglxj6eotukvadk7j3zva.py
# Topologically Sorted Source Nodes: [x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_232 => add_160, mul_273, mul_274, sub_68
#   x_233 => gt_68, mul_275, where_68
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_545), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_273, %unsqueeze_549), kwargs = {})
#   %add_160 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_274, %unsqueeze_551), kwargs = {})
#   %gt_68 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_160, 0), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_160, 0.01), kwargs = {})
#   %where_68 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_68, %add_160, %mul_275), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xi/cxibzc7cbozhuy6jsqxu3677ta36maykhy22v7qrnu5uer7ix5gm.py
# Topologically Sorted Source Nodes: [x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_235 => add_162, mul_277, mul_278, sub_69
#   x_236 => gt_69, mul_279, where_69
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_69, %unsqueeze_553), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_277, %unsqueeze_557), kwargs = {})
#   %add_162 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_278, %unsqueeze_559), kwargs = {})
#   %gt_69 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_162, 0), kwargs = {})
#   %mul_279 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_162, 0.01), kwargs = {})
#   %where_69 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_69, %add_162, %mul_279), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 16384
    y3 = (yindex // 16384)
    tmp0 = tl.load(in_out_ptr0 + (x1 + (128*y0)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (y2 + (16384*x1) + (2097152*y3)), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gp/cgpbxqlqofosdh3ovo22gwhy3ypftetxxevyczt5bfllr4vvcpjx.py
# Topologically Sorted Source Nodes: [x_237], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_237 => convolution_70
# Graph fragment:
#   %convolution_70 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_33, %arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_6 = async_compile.triton('triton_poi_fused_convolution_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (1048576 + x2 + (16384*y0) + (2097152*y1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (1048576*y1)), tmp0, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/r5/cr5wt4jkjw6hwtoppdk7deai3spvuzq3elalgymdnxzynmyimbwr.py
# Topologically Sorted Source Nodes: [x_238, x_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_238 => add_164, mul_281, mul_282, sub_70
#   x_239 => gt_70, mul_283, where_70
# Graph fragment:
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_70, %unsqueeze_561), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_563), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_281, %unsqueeze_565), kwargs = {})
#   %add_164 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_282, %unsqueeze_567), kwargs = {})
#   %gt_70 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_164, 0), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_164, 0.01), kwargs = {})
#   %where_70 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_70, %add_164, %mul_283), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yb/cybmp4dtmvs56ee7ssywju4gzybvryoqqu5cmgdald3yejzzjvcm.py
# Topologically Sorted Source Nodes: [x_241, x_242, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   x_241 => add_166, mul_285, mul_286, sub_71
#   x_242 => gt_71, mul_287, where_71
#   x_243 => add_167
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_569), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_571), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_285, %unsqueeze_573), kwargs = {})
#   %add_166 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_286, %unsqueeze_575), kwargs = {})
#   %gt_71 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_166, 0), kwargs = {})
#   %mul_287 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_166, 0.01), kwargs = {})
#   %where_71 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_71, %add_166, %mul_287), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_71, %getitem_33), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 16384
    y3 = (yindex // 16384)
    tmp0 = tl.load(in_out_ptr0 + (x1 + (64*y0)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (1048576 + y2 + (16384*x1) + (2097152*y3)), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr0 + (x1 + (64*y0)), tmp22, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/md/cmdk2loe7wgivtvfvxjkgomz2iyw2kmmkfglcm3ilbxltlpsbzdv.py
# Topologically Sorted Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_245 => add_169, mul_289, mul_290, sub_72
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_72, %unsqueeze_577), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_581), kwargs = {})
#   %add_169 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %unsqueeze_583), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/af/cafzteey3munmiigm6a3osnpxp7cw46wzonpnyg64nwzbjvtezsr.py
# Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_5 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_34, %where_72], 1), kwargs = {})
triton_poi_fused_cat_10 = async_compile.triton('triton_poi_fused_cat_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128) % 16384
    x2 = (xindex // 2097152)
    x3 = (xindex // 128)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + (16384*x0) + (2097152*x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((64*x3) + ((-64) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.01
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.where(tmp4, tmp5, tmp16)
    tl.store(out_ptr0 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eq/ceqgydelngyzdep25sc4goh5okrb4em2abt7ofmdtzx5xqhigxtm.py
# Topologically Sorted Source Nodes: [x_249, x_250], Original ATen: [aten.leaky_relu, aten.convolution]
# Source node to ATen node mapping:
#   x_249 => gt_73, mul_295, where_73
#   x_250 => convolution_74
# Graph fragment:
#   %gt_73 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_171, 0), kwargs = {})
#   %mul_295 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_171, 0.01), kwargs = {})
#   %where_73 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_73, %add_171, %mul_295), kwargs = {})
#   %convolution_74 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_73, %arg36_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_leaky_relu_11 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
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


# kernel path: /tmp/torchinductor_sahanp/lm/clmdyzchkjy2numc7wlzgbbm45ct5bgny44l2ulmdjyvlks627y4.py
# Topologically Sorted Source Nodes: [x_251, x_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_251 => add_173, mul_297, mul_298, sub_74
#   x_252 => gt_74, mul_299, where_74
# Graph fragment:
#   %sub_74 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_74, %unsqueeze_593), kwargs = {})
#   %mul_297 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_74, %unsqueeze_595), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_297, %unsqueeze_597), kwargs = {})
#   %add_173 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_298, %unsqueeze_599), kwargs = {})
#   %gt_74 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_173, 0), kwargs = {})
#   %mul_299 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_173, 0.01), kwargs = {})
#   %where_74 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_74, %add_173, %mul_299), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nn/cnnshwx6pk2jeihp2gbfs65n6pj4bfrctpy2zbqptesinc3psoon.py
# Topologically Sorted Source Nodes: [x_254, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_254 => add_175, mul_301, mul_302, sub_75
#   x_255 => gt_75, mul_303, where_75
# Graph fragment:
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_75, %unsqueeze_601), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_75, %unsqueeze_603), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_301, %unsqueeze_605), kwargs = {})
#   %add_175 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_302, %unsqueeze_607), kwargs = {})
#   %gt_75 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_175, 0), kwargs = {})
#   %mul_303 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_175, 0.01), kwargs = {})
#   %where_75 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_75, %add_175, %mul_303), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 4096
    y3 = (yindex // 4096)
    tmp0 = tl.load(in_out_ptr0 + (x1 + (128*y0)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (y2 + (4096*x1) + (524288*y3)), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7y/c7yjwnusnyalxzwwb5xynksd3xlvsrfpnyunxzxf37fblpdk4gfl.py
# Topologically Sorted Source Nodes: [x_256], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_256 => convolution_76
# Graph fragment:
#   %convolution_76 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_39, %arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_14 = async_compile.triton('triton_poi_fused_convolution_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (262144 + x2 + (4096*y0) + (524288*y1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (262144*y1)), tmp0, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jw/cjw3ox6jvuyqmxohr5qx5vq2lreua3gxe2ya6l4swz2xetjwfpmu.py
# Topologically Sorted Source Nodes: [x_257, x_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_257 => add_177, mul_305, mul_306, sub_76
#   x_258 => gt_76, mul_307, where_76
# Graph fragment:
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_76, %unsqueeze_609), kwargs = {})
#   %mul_305 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %unsqueeze_611), kwargs = {})
#   %mul_306 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_305, %unsqueeze_613), kwargs = {})
#   %add_177 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_306, %unsqueeze_615), kwargs = {})
#   %gt_76 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_177, 0), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_177, 0.01), kwargs = {})
#   %where_76 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_76, %add_177, %mul_307), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kn/cknimci45zerc4myfmg2ehs2yoyzkfbewjfrcrw3hcznalmq3yxx.py
# Topologically Sorted Source Nodes: [x_258, x_259], Original ATen: [aten.leaky_relu, aten.convolution]
# Source node to ATen node mapping:
#   x_258 => gt_76, mul_307, where_76
#   x_259 => convolution_77
# Graph fragment:
#   %gt_76 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_177, 0), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_177, 0.01), kwargs = {})
#   %where_76 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_76, %add_177, %mul_307), kwargs = {})
#   %convolution_77 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_76, %arg51_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_leaky_relu_16 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/pm/cpm7ybla33wutqhbu2evsph2l4maisa6yc5fk47py2bkvw7ztwd6.py
# Topologically Sorted Source Nodes: [x_260, x_261, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   x_260 => add_179, mul_309, mul_310, sub_77
#   x_261 => gt_77, mul_311, where_77
#   x_262 => add_180
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_617), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_309, %unsqueeze_621), kwargs = {})
#   %add_179 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_310, %unsqueeze_623), kwargs = {})
#   %gt_77 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_179, 0), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_179, 0.01), kwargs = {})
#   %where_77 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_77, %add_179, %mul_311), kwargs = {})
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_77, %getitem_39), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 4096
    y3 = (yindex // 4096)
    tmp0 = tl.load(in_out_ptr0 + (x1 + (64*y0)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (262144 + y2 + (4096*x1) + (524288*y3)), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr0 + (x1 + (64*y0)), tmp22, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ms/cmsghp7xo4zdt65sfbw7nmo5uinbwtls4a2mp7ervlrpdibgmyl3.py
# Topologically Sorted Source Nodes: [x_267, x_268, x_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   x_267 => add_184, mul_317, mul_318, sub_79
#   x_268 => gt_79, mul_319, where_79
#   x_269 => add_185
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_79, %unsqueeze_633), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_635), kwargs = {})
#   %mul_318 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_317, %unsqueeze_637), kwargs = {})
#   %add_184 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_318, %unsqueeze_639), kwargs = {})
#   %gt_79 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_184, 0), kwargs = {})
#   %mul_319 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_184, 0.01), kwargs = {})
#   %where_79 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_79, %add_184, %mul_319), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_79, %add_180), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_18', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_18(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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
    tmp21 = tl.load(in_out_ptr1 + (x2), None)
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr1 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4o/c4ot6cgol5nykqf5mighgu3jr5c2lsr25vdyvje4bqjvejdkbep4.py
# Topologically Sorted Source Nodes: [x_271], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_271 => add_187, mul_321, mul_322, sub_80
# Graph fragment:
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_80, %unsqueeze_641), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_643), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_321, %unsqueeze_645), kwargs = {})
#   %add_187 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_322, %unsqueeze_647), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6a/c6az5e3koviy2urh2nye3hbl5s4edny52dofojhupxchl7vazotn.py
# Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_6 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_40, %where_80], 1), kwargs = {})
triton_poi_fused_cat_20 = async_compile.triton('triton_poi_fused_cat_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128) % 4096
    x2 = (xindex // 524288)
    x3 = (xindex // 128)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + (4096*x0) + (524288*x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((64*x3) + ((-64) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.01
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.where(tmp4, tmp5, tmp16)
    tl.store(out_ptr0 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/et/cetjxpfynm5ncui324xjgq4gmnvblzse7lacfmdhmsurbhffwlhv.py
# Topologically Sorted Source Nodes: [x_275, x_276], Original ATen: [aten.leaky_relu, aten.convolution]
# Source node to ATen node mapping:
#   x_275 => gt_81, mul_327, where_81
#   x_276 => convolution_82
# Graph fragment:
#   %gt_81 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_189, 0), kwargs = {})
#   %mul_327 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_189, 0.01), kwargs = {})
#   %where_81 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_81, %add_189, %mul_327), kwargs = {})
#   %convolution_82 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_81, %arg76_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_leaky_relu_21 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/st/cstvbeknr3u5d4qeolaf4bccichj5cxhiauqmvjobsfagrvxzww4.py
# Topologically Sorted Source Nodes: [x_277, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_277 => add_191, mul_329, mul_330, sub_82
#   x_278 => gt_82, mul_331, where_82
# Graph fragment:
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_82, %unsqueeze_657), kwargs = {})
#   %mul_329 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_82, %unsqueeze_659), kwargs = {})
#   %mul_330 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_329, %unsqueeze_661), kwargs = {})
#   %add_191 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_330, %unsqueeze_663), kwargs = {})
#   %gt_82 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_191, 0), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_191, 0.01), kwargs = {})
#   %where_82 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_82, %add_191, %mul_331), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/td/ctdyvj4zlljapwvzoxdfmifjiw4o7lxkmax4eexpfi636mvdxm4o.py
# Topologically Sorted Source Nodes: [x_280, x_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_280 => add_193, mul_333, mul_334, sub_83
#   x_281 => gt_83, mul_335, where_83
# Graph fragment:
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_83, %unsqueeze_665), kwargs = {})
#   %mul_333 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_667), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_333, %unsqueeze_669), kwargs = {})
#   %add_193 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_334, %unsqueeze_671), kwargs = {})
#   %gt_83 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_193, 0), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_193, 0.01), kwargs = {})
#   %where_83 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_83, %add_193, %mul_335), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 1024
    y3 = (yindex // 1024)
    tmp0 = tl.load(in_out_ptr0 + (x1 + (256*y0)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (y2 + (1024*x1) + (262144*y3)), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7d/c7ddmrufbwxeye73tbv2kkpt3kshohpo5ih53yuctj6fjbsminue.py
# Topologically Sorted Source Nodes: [x_282], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_282 => convolution_84
# Graph fragment:
#   %convolution_84 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_45, %arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_24 = async_compile.triton('triton_poi_fused_convolution_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (131072 + x2 + (1024*y0) + (262144*y1)), xmask)
    tl.store(out_ptr0 + (y0 + (128*x2) + (131072*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xk/cxktntohtwovsul2ibvatb7aegp5343s4acm2cpmoozihnyqncpg.py
# Topologically Sorted Source Nodes: [x_283, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_283 => add_195, mul_337, mul_338, sub_84
#   x_284 => gt_84, mul_339, where_84
# Graph fragment:
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_84, %unsqueeze_673), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_675), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_337, %unsqueeze_677), kwargs = {})
#   %add_195 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_338, %unsqueeze_679), kwargs = {})
#   %gt_84 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_195, 0), kwargs = {})
#   %mul_339 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_195, 0.01), kwargs = {})
#   %where_84 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_84, %add_195, %mul_339), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qb/cqbtr3e2yhugna7ge62ihzw7h5ykphghfq4rhve7upm7castla33.py
# Topologically Sorted Source Nodes: [x_284, x_285], Original ATen: [aten.leaky_relu, aten.convolution]
# Source node to ATen node mapping:
#   x_284 => gt_84, mul_339, where_84
#   x_285 => convolution_85
# Graph fragment:
#   %gt_84 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_195, 0), kwargs = {})
#   %mul_339 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_195, 0.01), kwargs = {})
#   %where_84 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_84, %add_195, %mul_339), kwargs = {})
#   %convolution_85 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_84, %arg91_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_leaky_relu_26 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/rg/crgi6sumufcezo2lpom7bcdva54ptg5e6uk5dvsqomqt6s5i46ni.py
# Topologically Sorted Source Nodes: [x_286, x_287, x_288], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   x_286 => add_197, mul_341, mul_342, sub_85
#   x_287 => gt_85, mul_343, where_85
#   x_288 => add_198
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_85, %unsqueeze_681), kwargs = {})
#   %mul_341 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_342 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_341, %unsqueeze_685), kwargs = {})
#   %add_197 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_342, %unsqueeze_687), kwargs = {})
#   %gt_85 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_197, 0), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_197, 0.01), kwargs = {})
#   %where_85 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_85, %add_197, %mul_343), kwargs = {})
#   %add_198 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_85, %getitem_45), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 1024
    y3 = (yindex // 1024)
    tmp0 = tl.load(in_out_ptr0 + (x1 + (128*y0)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (131072 + y2 + (1024*x1) + (262144*y3)), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr0 + (x1 + (128*y0)), tmp22, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4l/c4li3nycath63qtwcwa2p73yueoj5welmejpjq2ubhqlobasufx5.py
# Topologically Sorted Source Nodes: [x_293, x_294, x_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   x_293 => add_202, mul_349, mul_350, sub_87
#   x_294 => gt_87, mul_351, where_87
#   x_295 => add_203
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_697), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_350 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_349, %unsqueeze_701), kwargs = {})
#   %add_202 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_350, %unsqueeze_703), kwargs = {})
#   %gt_87 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_202, 0), kwargs = {})
#   %mul_351 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_202, 0.01), kwargs = {})
#   %where_87 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_87, %add_202, %mul_351), kwargs = {})
#   %add_203 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_87, %add_198), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_28', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_28(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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
    tmp21 = tl.load(in_out_ptr1 + (x2), None)
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr1 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7j/c7ja6jr3mpdq54bhygc2rtmh4q3kqzqyreikydm76pjvlnbzuc7j.py
# Topologically Sorted Source Nodes: [x_339], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_339 => add_235, mul_401, mul_402, sub_100
# Graph fragment:
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_801), kwargs = {})
#   %mul_401 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_803), kwargs = {})
#   %mul_402 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_401, %unsqueeze_805), kwargs = {})
#   %add_235 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_402, %unsqueeze_807), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wa/cwadjohrfjz5khkp6n2azcqh54hpbkee7a2hiqz6ypelzxshu4g4.py
# Topologically Sorted Source Nodes: [cat_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_7 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_46, %where_100], 1), kwargs = {})
triton_poi_fused_cat_30 = async_compile.triton('triton_poi_fused_cat_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_30(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256) % 1024
    x2 = (xindex // 262144)
    x3 = (xindex // 256)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + (1024*x0) + (262144*x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 256, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((128*x3) + ((-128) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.01
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.where(tmp4, tmp5, tmp16)
    tl.store(out_ptr0 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6z/c6z2nbxrq3brixv7mleu47zj3ivmbm3emjim3wopnzye6uxcpp5z.py
# Topologically Sorted Source Nodes: [x_343, x_344], Original ATen: [aten.leaky_relu, aten.convolution]
# Source node to ATen node mapping:
#   x_343 => gt_101, mul_407, where_101
#   x_344 => convolution_102
# Graph fragment:
#   %gt_101 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_237, 0), kwargs = {})
#   %mul_407 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_237, 0.01), kwargs = {})
#   %where_101 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_101, %add_237, %mul_407), kwargs = {})
#   %convolution_102 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_101, %arg176_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_leaky_relu_31 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/4r/c4r5itxmwqgmmsq34dv5wzfyuo2xux7tecoiwipvcrxsn2bsw3jp.py
# Topologically Sorted Source Nodes: [x_345, x_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_345 => add_239, mul_409, mul_410, sub_102
#   x_346 => gt_102, mul_411, where_102
# Graph fragment:
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_102, %unsqueeze_817), kwargs = {})
#   %mul_409 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %unsqueeze_819), kwargs = {})
#   %mul_410 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_409, %unsqueeze_821), kwargs = {})
#   %add_239 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_410, %unsqueeze_823), kwargs = {})
#   %gt_102 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_239, 0), kwargs = {})
#   %mul_411 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_239, 0.01), kwargs = {})
#   %where_102 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_102, %add_239, %mul_411), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zh/czh27rshyvso7z4nqv2hx4upidanqq4nnzxjillbwa3njjqarpqs.py
# Topologically Sorted Source Nodes: [x_348, x_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_348 => add_241, mul_413, mul_414, sub_103
#   x_349 => gt_103, mul_415, where_103
# Graph fragment:
#   %sub_103 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_103, %unsqueeze_825), kwargs = {})
#   %mul_413 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_103, %unsqueeze_827), kwargs = {})
#   %mul_414 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_413, %unsqueeze_829), kwargs = {})
#   %add_241 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_414, %unsqueeze_831), kwargs = {})
#   %gt_103 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_241, 0), kwargs = {})
#   %mul_415 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_241, 0.01), kwargs = {})
#   %where_103 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_103, %add_241, %mul_415), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 256
    y3 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x1 + (512*y0)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (y2 + (256*x1) + (131072*y3)), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sx/csx4irebulkodyrq5faranhs4zrbwhbcm4qrwtfqyi5uvs44can7.py
# Topologically Sorted Source Nodes: [x_350], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_350 => convolution_104
# Graph fragment:
#   %convolution_104 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_51, %arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_34 = async_compile.triton('triton_poi_fused_convolution_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (65536 + x2 + (256*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (65536*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bu/cbu2f2fw5czvwptmn2uo6yezykjecytgaci6zlnjh6tryvohghxe.py
# Topologically Sorted Source Nodes: [x_351, x_352], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_351 => add_243, mul_417, mul_418, sub_104
#   x_352 => gt_104, mul_419, where_104
# Graph fragment:
#   %sub_104 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_104, %unsqueeze_833), kwargs = {})
#   %mul_417 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_104, %unsqueeze_835), kwargs = {})
#   %mul_418 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_417, %unsqueeze_837), kwargs = {})
#   %add_243 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_418, %unsqueeze_839), kwargs = {})
#   %gt_104 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_243, 0), kwargs = {})
#   %mul_419 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_243, 0.01), kwargs = {})
#   %where_104 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_104, %add_243, %mul_419), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/s3/cs36wgjrwqbrl37ly4ylhopsykxtxsb6ryhvolr7tedyhibzcwfa.py
# Topologically Sorted Source Nodes: [x_352, x_353], Original ATen: [aten.leaky_relu, aten.convolution]
# Source node to ATen node mapping:
#   x_352 => gt_104, mul_419, where_104
#   x_353 => convolution_105
# Graph fragment:
#   %gt_104 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_243, 0), kwargs = {})
#   %mul_419 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_243, 0.01), kwargs = {})
#   %where_104 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_104, %add_243, %mul_419), kwargs = {})
#   %convolution_105 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_104, %arg191_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_leaky_relu_36 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ls/clsl2utf3ge66vko5nncdwn4m7j5dt7hbcry6wfwjvh3uembtu27.py
# Topologically Sorted Source Nodes: [x_354, x_355, x_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   x_354 => add_245, mul_421, mul_422, sub_105
#   x_355 => gt_105, mul_423, where_105
#   x_356 => add_246
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_841), kwargs = {})
#   %mul_421 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_422 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_421, %unsqueeze_845), kwargs = {})
#   %add_245 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_422, %unsqueeze_847), kwargs = {})
#   %gt_105 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_245, 0), kwargs = {})
#   %mul_423 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_245, 0.01), kwargs = {})
#   %where_105 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_105, %add_245, %mul_423), kwargs = {})
#   %add_246 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_105, %getitem_51), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 256
    y3 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x1 + (256*y0)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (65536 + y2 + (256*x1) + (131072*y3)), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr0 + (x1 + (256*y0)), tmp22, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cz/cczy57bft7xcgvrc3imesmiwilrjszebeufsov2gjqqvonk62x75.py
# Topologically Sorted Source Nodes: [x_361, x_362, x_363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   x_361 => add_250, mul_429, mul_430, sub_107
#   x_362 => gt_107, mul_431, where_107
#   x_363 => add_251
# Graph fragment:
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_107, %unsqueeze_857), kwargs = {})
#   %mul_429 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %unsqueeze_859), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_429, %unsqueeze_861), kwargs = {})
#   %add_250 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_430, %unsqueeze_863), kwargs = {})
#   %gt_107 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_250, 0), kwargs = {})
#   %mul_431 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_250, 0.01), kwargs = {})
#   %where_107 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_107, %add_250, %mul_431), kwargs = {})
#   %add_251 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_107, %add_246), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_38', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_38(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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
    tmp21 = tl.load(in_out_ptr1 + (x2), None)
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr1 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/63/c63cnj3eklci45gtsx6hevmzbg7ghp3255dkv6x4ir65zxqyvqfh.py
# Topologically Sorted Source Nodes: [x_407], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_407 => add_283, mul_481, mul_482, sub_120
# Graph fragment:
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_120, %unsqueeze_961), kwargs = {})
#   %mul_481 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_120, %unsqueeze_963), kwargs = {})
#   %mul_482 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_481, %unsqueeze_965), kwargs = {})
#   %add_283 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_482, %unsqueeze_967), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pq/cpqbo36trcw76szyj42wwadgs2jgem6vnttful7jxnq6vca77evz.py
# Topologically Sorted Source Nodes: [cat_8], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_8 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_52, %where_120], 1), kwargs = {})
triton_poi_fused_cat_40 = async_compile.triton('triton_poi_fused_cat_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_40(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512) % 256
    x2 = (xindex // 131072)
    x3 = (xindex // 512)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + (256*x0) + (131072*x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 512, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((256*x3) + ((-256) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.01
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.where(tmp4, tmp5, tmp16)
    tl.store(out_ptr0 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5y/c5yw37u4u3zyeipvgpq7mveuvdfquwopladbwhnen74i2l4llafi.py
# Topologically Sorted Source Nodes: [x_411, x_412], Original ATen: [aten.leaky_relu, aten.convolution]
# Source node to ATen node mapping:
#   x_411 => gt_121, mul_487, where_121
#   x_412 => convolution_122
# Graph fragment:
#   %gt_121 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_285, 0), kwargs = {})
#   %mul_487 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_285, 0.01), kwargs = {})
#   %where_121 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_121, %add_285, %mul_487), kwargs = {})
#   %convolution_122 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_121, %arg276_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_leaky_relu_41 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_41(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/cr/ccrlx636xxa2tactigo5muc72kra62i4od5ydqnswstw3lztgqkc.py
# Topologically Sorted Source Nodes: [x_413, x_414], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_413 => add_287, mul_489, mul_490, sub_122
#   x_414 => gt_122, mul_491, where_122
# Graph fragment:
#   %sub_122 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_122, %unsqueeze_977), kwargs = {})
#   %mul_489 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_122, %unsqueeze_979), kwargs = {})
#   %mul_490 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_489, %unsqueeze_981), kwargs = {})
#   %add_287 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_490, %unsqueeze_983), kwargs = {})
#   %gt_122 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_287, 0), kwargs = {})
#   %mul_491 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_287, 0.01), kwargs = {})
#   %where_122 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_122, %add_287, %mul_491), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_42', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jf/cjfjcn5gshdsex3uzcok4fuyhqczjoxyegqj2yplw2malx5qvlxu.py
# Topologically Sorted Source Nodes: [x_416, x_417], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_416 => add_289, mul_493, mul_494, sub_123
#   x_417 => gt_123, mul_495, where_123
# Graph fragment:
#   %sub_123 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_123, %unsqueeze_985), kwargs = {})
#   %mul_493 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_123, %unsqueeze_987), kwargs = {})
#   %mul_494 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_493, %unsqueeze_989), kwargs = {})
#   %add_289 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_494, %unsqueeze_991), kwargs = {})
#   %gt_123 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_289, 0), kwargs = {})
#   %mul_495 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_289, 0.01), kwargs = {})
#   %where_123 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_123, %add_289, %mul_495), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_43', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 64
    y3 = (yindex // 64)
    tmp0 = tl.load(in_out_ptr0 + (x1 + (1024*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (y2 + (64*x1) + (65536*y3)), tmp20, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zl/czltqsn5fxk7r5k4phmbckxkvt3fechpxde2suaeemdc6ugbufxe.py
# Topologically Sorted Source Nodes: [x_418], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_418 => convolution_124
# Graph fragment:
#   %convolution_124 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_57, %arg286_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_44 = async_compile.triton('triton_poi_fused_convolution_44', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_44(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (32768 + x2 + (64*y0) + (65536*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (32768*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/t2/ct2lmmqydxz74hj2kfavhg3s7fidmatrqapzvytmitj7sba7vv6k.py
# Topologically Sorted Source Nodes: [x_419, x_420], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_419 => add_291, mul_497, mul_498, sub_124
#   x_420 => gt_124, mul_499, where_124
# Graph fragment:
#   %sub_124 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_124, %unsqueeze_993), kwargs = {})
#   %mul_497 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_124, %unsqueeze_995), kwargs = {})
#   %mul_498 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_497, %unsqueeze_997), kwargs = {})
#   %add_291 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_498, %unsqueeze_999), kwargs = {})
#   %gt_124 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_291, 0), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_291, 0.01), kwargs = {})
#   %where_124 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_124, %add_291, %mul_499), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uu/cuu35ihogd2d32erfn7u4vg7habue3gcudlrgb6crau3cbcclhqa.py
# Topologically Sorted Source Nodes: [x_420, x_421], Original ATen: [aten.leaky_relu, aten.convolution]
# Source node to ATen node mapping:
#   x_420 => gt_124, mul_499, where_124
#   x_421 => convolution_125
# Graph fragment:
#   %gt_124 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_291, 0), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_291, 0.01), kwargs = {})
#   %where_124 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_124, %add_291, %mul_499), kwargs = {})
#   %convolution_125 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_124, %arg291_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_leaky_relu_46 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_46(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/qe/cqemyrp47c54qnwfff3ie4baimex7s3l34jmjy7bbel32cui27gz.py
# Topologically Sorted Source Nodes: [x_422, x_423, x_424], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   x_422 => add_293, mul_501, mul_502, sub_125
#   x_423 => gt_125, mul_503, where_125
#   x_424 => add_294
# Graph fragment:
#   %sub_125 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_125, %unsqueeze_1001), kwargs = {})
#   %mul_501 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_125, %unsqueeze_1003), kwargs = {})
#   %mul_502 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_501, %unsqueeze_1005), kwargs = {})
#   %add_293 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_502, %unsqueeze_1007), kwargs = {})
#   %gt_125 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_293, 0), kwargs = {})
#   %mul_503 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_293, 0.01), kwargs = {})
#   %where_125 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_125, %add_293, %mul_503), kwargs = {})
#   %add_294 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_125, %getitem_57), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 64
    y3 = (yindex // 64)
    tmp0 = tl.load(in_out_ptr0 + (x1 + (512*y0)), xmask & ymask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (32768 + y2 + (64*x1) + (65536*y3)), xmask & ymask)
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr0 + (x1 + (512*y0)), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/e3/ce3whpqbyxyhbbt6e46yahamu4ukwykmvhr5nufch3lubhnra7lu.py
# Topologically Sorted Source Nodes: [x_429, x_430, x_431], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   x_429 => add_298, mul_509, mul_510, sub_127
#   x_430 => gt_127, mul_511, where_127
#   x_431 => add_299
# Graph fragment:
#   %sub_127 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_127, %unsqueeze_1017), kwargs = {})
#   %mul_509 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_127, %unsqueeze_1019), kwargs = {})
#   %mul_510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_509, %unsqueeze_1021), kwargs = {})
#   %add_298 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_510, %unsqueeze_1023), kwargs = {})
#   %gt_127 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_298, 0), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_298, 0.01), kwargs = {})
#   %where_127 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_127, %add_298, %mul_511), kwargs = {})
#   %add_299 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_127, %add_294), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_48', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_48', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_48(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
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
    tmp21 = tl.load(in_out_ptr1 + (x2), None)
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr1 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cn/ccnc6un5o3je2n5e6oaupmqz5paz3cocgm3sprcx3riuyrizllqq.py
# Topologically Sorted Source Nodes: [x_447], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_447 => add_311, mul_529, mul_530, sub_132
# Graph fragment:
#   %sub_132 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_132, %unsqueeze_1057), kwargs = {})
#   %mul_529 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_132, %unsqueeze_1059), kwargs = {})
#   %mul_530 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_529, %unsqueeze_1061), kwargs = {})
#   %add_311 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_530, %unsqueeze_1063), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ny/cnycrx3f6rn3b26gusqzp4wmtj5wt4q2mezvm2oscibwn5i7kaak.py
# Topologically Sorted Source Nodes: [cat_9], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_9 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_58, %where_132], 1), kwargs = {})
triton_poi_fused_cat_50 = async_compile.triton('triton_poi_fused_cat_50', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_50(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 64
    x2 = (xindex // 65536)
    x3 = (xindex // 1024)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + (64*x0) + (65536*x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 1024, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((512*x3) + ((-512) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.01
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.where(tmp4, tmp5, tmp16)
    tl.store(out_ptr0 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kz/ckz62fptndwuv4hy7sumxzh4drcbipcxeqkl7brryujgtgll3miw.py
# Topologically Sorted Source Nodes: [x_450], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_450 => add_313, mul_533, mul_534, sub_133
# Graph fragment:
#   %sub_133 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_133, %unsqueeze_1065), kwargs = {})
#   %mul_533 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_133, %unsqueeze_1067), kwargs = {})
#   %mul_534 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_533, %unsqueeze_1069), kwargs = {})
#   %add_313 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_534, %unsqueeze_1071), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gp/cgpuprneekz5f6nppy4uwvplbtnmkqnlbh274ao56hk6fgimlmgn.py
# Topologically Sorted Source Nodes: [x_451, x_452], Original ATen: [aten.leaky_relu, aten.mean]
# Source node to ATen node mapping:
#   x_451 => gt_133, mul_535, where_133
#   x_452 => mean_1
# Graph fragment:
#   %gt_133 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_313, 0), kwargs = {})
#   %mul_535 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_313, 0.01), kwargs = {})
#   %where_133 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_133, %add_313, %mul_535), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%where_133, [-1, -2], True), kwargs = {})
triton_per_fused_leaky_relu_mean_52 = async_compile.triton('triton_per_fused_leaky_relu_mean_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_leaky_relu_mean_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_leaky_relu_mean_52(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 64
    RBLOCK: tl.constexpr = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (65536*x1)), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None]
    tmp9 = 64.0
    tmp10 = tmp8 / tmp9
    tl.store(out_ptr1 + (x3), tmp10, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg17_1, (32, ), (1, ))
    assert_size_stride(arg18_1, (32, ), (1, ))
    assert_size_stride(arg19_1, (32, ), (1, ))
    assert_size_stride(arg20_1, (32, ), (1, ))
    assert_size_stride(arg21_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg27_1, (64, ), (1, ))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, ), (1, ))
    assert_size_stride(arg31_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg32_1, (64, ), (1, ))
    assert_size_stride(arg33_1, (64, ), (1, ))
    assert_size_stride(arg34_1, (64, ), (1, ))
    assert_size_stride(arg35_1, (64, ), (1, ))
    assert_size_stride(arg36_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg47_1, (64, ), (1, ))
    assert_size_stride(arg48_1, (64, ), (1, ))
    assert_size_stride(arg49_1, (64, ), (1, ))
    assert_size_stride(arg50_1, (64, ), (1, ))
    assert_size_stride(arg51_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg52_1, (64, ), (1, ))
    assert_size_stride(arg53_1, (64, ), (1, ))
    assert_size_stride(arg54_1, (64, ), (1, ))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg57_1, (64, ), (1, ))
    assert_size_stride(arg58_1, (64, ), (1, ))
    assert_size_stride(arg59_1, (64, ), (1, ))
    assert_size_stride(arg60_1, (64, ), (1, ))
    assert_size_stride(arg61_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg62_1, (64, ), (1, ))
    assert_size_stride(arg63_1, (64, ), (1, ))
    assert_size_stride(arg64_1, (64, ), (1, ))
    assert_size_stride(arg65_1, (64, ), (1, ))
    assert_size_stride(arg66_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg67_1, (64, ), (1, ))
    assert_size_stride(arg68_1, (64, ), (1, ))
    assert_size_stride(arg69_1, (64, ), (1, ))
    assert_size_stride(arg70_1, (64, ), (1, ))
    assert_size_stride(arg71_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (128, ), (1, ))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg77_1, (256, ), (1, ))
    assert_size_stride(arg78_1, (256, ), (1, ))
    assert_size_stride(arg79_1, (256, ), (1, ))
    assert_size_stride(arg80_1, (256, ), (1, ))
    assert_size_stride(arg81_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (256, ), (1, ))
    assert_size_stride(arg84_1, (256, ), (1, ))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg87_1, (128, ), (1, ))
    assert_size_stride(arg88_1, (128, ), (1, ))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (128, ), (1, ))
    assert_size_stride(arg96_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg102_1, (128, ), (1, ))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, ), (1, ))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg112_1, (128, ), (1, ))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg117_1, (128, ), (1, ))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (128, ), (1, ))
    assert_size_stride(arg124_1, (128, ), (1, ))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (128, ), (1, ))
    assert_size_stride(arg129_1, (128, ), (1, ))
    assert_size_stride(arg130_1, (128, ), (1, ))
    assert_size_stride(arg131_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (128, ), (1, ))
    assert_size_stride(arg135_1, (128, ), (1, ))
    assert_size_stride(arg136_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg142_1, (128, ), (1, ))
    assert_size_stride(arg143_1, (128, ), (1, ))
    assert_size_stride(arg144_1, (128, ), (1, ))
    assert_size_stride(arg145_1, (128, ), (1, ))
    assert_size_stride(arg146_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg147_1, (128, ), (1, ))
    assert_size_stride(arg148_1, (128, ), (1, ))
    assert_size_stride(arg149_1, (128, ), (1, ))
    assert_size_stride(arg150_1, (128, ), (1, ))
    assert_size_stride(arg151_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg152_1, (128, ), (1, ))
    assert_size_stride(arg153_1, (128, ), (1, ))
    assert_size_stride(arg154_1, (128, ), (1, ))
    assert_size_stride(arg155_1, (128, ), (1, ))
    assert_size_stride(arg156_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg157_1, (128, ), (1, ))
    assert_size_stride(arg158_1, (128, ), (1, ))
    assert_size_stride(arg159_1, (128, ), (1, ))
    assert_size_stride(arg160_1, (128, ), (1, ))
    assert_size_stride(arg161_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (128, ), (1, ))
    assert_size_stride(arg164_1, (128, ), (1, ))
    assert_size_stride(arg165_1, (128, ), (1, ))
    assert_size_stride(arg166_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg167_1, (128, ), (1, ))
    assert_size_stride(arg168_1, (128, ), (1, ))
    assert_size_stride(arg169_1, (128, ), (1, ))
    assert_size_stride(arg170_1, (128, ), (1, ))
    assert_size_stride(arg171_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg172_1, (256, ), (1, ))
    assert_size_stride(arg173_1, (256, ), (1, ))
    assert_size_stride(arg174_1, (256, ), (1, ))
    assert_size_stride(arg175_1, (256, ), (1, ))
    assert_size_stride(arg176_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg177_1, (512, ), (1, ))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (512, ), (1, ))
    assert_size_stride(arg180_1, (512, ), (1, ))
    assert_size_stride(arg181_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg182_1, (512, ), (1, ))
    assert_size_stride(arg183_1, (512, ), (1, ))
    assert_size_stride(arg184_1, (512, ), (1, ))
    assert_size_stride(arg185_1, (512, ), (1, ))
    assert_size_stride(arg186_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg187_1, (256, ), (1, ))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (256, ), (1, ))
    assert_size_stride(arg190_1, (256, ), (1, ))
    assert_size_stride(arg191_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (256, ), (1, ))
    assert_size_stride(arg194_1, (256, ), (1, ))
    assert_size_stride(arg195_1, (256, ), (1, ))
    assert_size_stride(arg196_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg197_1, (256, ), (1, ))
    assert_size_stride(arg198_1, (256, ), (1, ))
    assert_size_stride(arg199_1, (256, ), (1, ))
    assert_size_stride(arg200_1, (256, ), (1, ))
    assert_size_stride(arg201_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg202_1, (256, ), (1, ))
    assert_size_stride(arg203_1, (256, ), (1, ))
    assert_size_stride(arg204_1, (256, ), (1, ))
    assert_size_stride(arg205_1, (256, ), (1, ))
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
    assert_size_stride(arg216_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg217_1, (256, ), (1, ))
    assert_size_stride(arg218_1, (256, ), (1, ))
    assert_size_stride(arg219_1, (256, ), (1, ))
    assert_size_stride(arg220_1, (256, ), (1, ))
    assert_size_stride(arg221_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg222_1, (256, ), (1, ))
    assert_size_stride(arg223_1, (256, ), (1, ))
    assert_size_stride(arg224_1, (256, ), (1, ))
    assert_size_stride(arg225_1, (256, ), (1, ))
    assert_size_stride(arg226_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg227_1, (256, ), (1, ))
    assert_size_stride(arg228_1, (256, ), (1, ))
    assert_size_stride(arg229_1, (256, ), (1, ))
    assert_size_stride(arg230_1, (256, ), (1, ))
    assert_size_stride(arg231_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg232_1, (256, ), (1, ))
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (256, ), (1, ))
    assert_size_stride(arg235_1, (256, ), (1, ))
    assert_size_stride(arg236_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg237_1, (256, ), (1, ))
    assert_size_stride(arg238_1, (256, ), (1, ))
    assert_size_stride(arg239_1, (256, ), (1, ))
    assert_size_stride(arg240_1, (256, ), (1, ))
    assert_size_stride(arg241_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg242_1, (256, ), (1, ))
    assert_size_stride(arg243_1, (256, ), (1, ))
    assert_size_stride(arg244_1, (256, ), (1, ))
    assert_size_stride(arg245_1, (256, ), (1, ))
    assert_size_stride(arg246_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg247_1, (256, ), (1, ))
    assert_size_stride(arg248_1, (256, ), (1, ))
    assert_size_stride(arg249_1, (256, ), (1, ))
    assert_size_stride(arg250_1, (256, ), (1, ))
    assert_size_stride(arg251_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg252_1, (256, ), (1, ))
    assert_size_stride(arg253_1, (256, ), (1, ))
    assert_size_stride(arg254_1, (256, ), (1, ))
    assert_size_stride(arg255_1, (256, ), (1, ))
    assert_size_stride(arg256_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg257_1, (256, ), (1, ))
    assert_size_stride(arg258_1, (256, ), (1, ))
    assert_size_stride(arg259_1, (256, ), (1, ))
    assert_size_stride(arg260_1, (256, ), (1, ))
    assert_size_stride(arg261_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg262_1, (256, ), (1, ))
    assert_size_stride(arg263_1, (256, ), (1, ))
    assert_size_stride(arg264_1, (256, ), (1, ))
    assert_size_stride(arg265_1, (256, ), (1, ))
    assert_size_stride(arg266_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg267_1, (256, ), (1, ))
    assert_size_stride(arg268_1, (256, ), (1, ))
    assert_size_stride(arg269_1, (256, ), (1, ))
    assert_size_stride(arg270_1, (256, ), (1, ))
    assert_size_stride(arg271_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg272_1, (512, ), (1, ))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (512, ), (1, ))
    assert_size_stride(arg275_1, (512, ), (1, ))
    assert_size_stride(arg276_1, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg277_1, (1024, ), (1, ))
    assert_size_stride(arg278_1, (1024, ), (1, ))
    assert_size_stride(arg279_1, (1024, ), (1, ))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg282_1, (1024, ), (1, ))
    assert_size_stride(arg283_1, (1024, ), (1, ))
    assert_size_stride(arg284_1, (1024, ), (1, ))
    assert_size_stride(arg285_1, (1024, ), (1, ))
    assert_size_stride(arg286_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg287_1, (512, ), (1, ))
    assert_size_stride(arg288_1, (512, ), (1, ))
    assert_size_stride(arg289_1, (512, ), (1, ))
    assert_size_stride(arg290_1, (512, ), (1, ))
    assert_size_stride(arg291_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg292_1, (512, ), (1, ))
    assert_size_stride(arg293_1, (512, ), (1, ))
    assert_size_stride(arg294_1, (512, ), (1, ))
    assert_size_stride(arg295_1, (512, ), (1, ))
    assert_size_stride(arg296_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg297_1, (512, ), (1, ))
    assert_size_stride(arg298_1, (512, ), (1, ))
    assert_size_stride(arg299_1, (512, ), (1, ))
    assert_size_stride(arg300_1, (512, ), (1, ))
    assert_size_stride(arg301_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg302_1, (512, ), (1, ))
    assert_size_stride(arg303_1, (512, ), (1, ))
    assert_size_stride(arg304_1, (512, ), (1, ))
    assert_size_stride(arg305_1, (512, ), (1, ))
    assert_size_stride(arg306_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg307_1, (512, ), (1, ))
    assert_size_stride(arg308_1, (512, ), (1, ))
    assert_size_stride(arg309_1, (512, ), (1, ))
    assert_size_stride(arg310_1, (512, ), (1, ))
    assert_size_stride(arg311_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg312_1, (512, ), (1, ))
    assert_size_stride(arg313_1, (512, ), (1, ))
    assert_size_stride(arg314_1, (512, ), (1, ))
    assert_size_stride(arg315_1, (512, ), (1, ))
    assert_size_stride(arg316_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg317_1, (512, ), (1, ))
    assert_size_stride(arg318_1, (512, ), (1, ))
    assert_size_stride(arg319_1, (512, ), (1, ))
    assert_size_stride(arg320_1, (512, ), (1, ))
    assert_size_stride(arg321_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg322_1, (512, ), (1, ))
    assert_size_stride(arg323_1, (512, ), (1, ))
    assert_size_stride(arg324_1, (512, ), (1, ))
    assert_size_stride(arg325_1, (512, ), (1, ))
    assert_size_stride(arg326_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg327_1, (512, ), (1, ))
    assert_size_stride(arg328_1, (512, ), (1, ))
    assert_size_stride(arg329_1, (512, ), (1, ))
    assert_size_stride(arg330_1, (512, ), (1, ))
    assert_size_stride(arg331_1, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg332_1, (1024, ), (1, ))
    assert_size_stride(arg333_1, (1024, ), (1, ))
    assert_size_stride(arg334_1, (1024, ), (1, ))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg337_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 256, 256), (196608, 1, 768, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_228], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_228], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_228], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 256, 256), (2097152, 1, 8192, 32))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((8, 32, 256, 256), (2097152, 1, 8192, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_229, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, buf4, 16777216, grid=grid(16777216), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf3
        buf5 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_230, x_231], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_3.run(arg6_1, buf5, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [x_230, x_231], Original ATen: [aten.leaky_relu, aten.convolution]
        buf6 = extern_kernels.convolution(buf4, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 64, 128, 128), (1048576, 1, 8192, 64))
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided_cuda((8, 64, 128, 128), (1048576, 1, 8192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4.run(buf7, arg7_1, arg8_1, arg9_1, arg10_1, buf8, 8388608, grid=grid(8388608), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        del buf7
        # Topologically Sorted Source Nodes: [x_233, x_234], Original ATen: [aten.leaky_relu, aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 128, 128, 128), (2097152, 1, 16384, 128))
        del arg11_1
        buf10 = buf9; del buf9  # reuse
        buf11 = reinterpret_tensor(buf4, (8, 128, 128, 128), (2097152, 16384, 128, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5.run(buf10, arg12_1, arg13_1, arg14_1, arg15_1, buf11, 131072, 128, grid=grid(131072, 128), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        buf12 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_237], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf11, buf12, 512, 16384, grid=grid(512, 16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_237], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 32, 128, 128), (524288, 1, 4096, 32))
        del arg16_1
        buf14 = buf13; del buf13  # reuse
        buf15 = empty_strided_cuda((8, 32, 128, 128), (524288, 1, 4096, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_238, x_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7.run(buf14, arg17_1, arg18_1, arg19_1, arg20_1, buf15, 4194304, grid=grid(4194304), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del buf14
        buf16 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_239, x_240], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_3.run(arg21_1, buf16, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg21_1
        # Topologically Sorted Source Nodes: [x_239, x_240], Original ATen: [aten.leaky_relu, aten.convolution]
        buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 64, 128, 128), (1048576, 1, 8192, 64))
        del buf16
        buf18 = buf17; del buf17  # reuse
        buf19 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_242, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_8.run(buf18, arg22_1, arg23_1, arg24_1, arg25_1, buf11, buf19, 131072, 64, grid=grid(131072, 64), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del buf18
        # Topologically Sorted Source Nodes: [x_242, x_243, x_244], Original ATen: [aten.leaky_relu, aten.add, aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 64, 128, 128), (1048576, 1, 8192, 64))
        del arg26_1
        del buf19
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_9.run(buf21, arg27_1, arg28_1, arg29_1, arg30_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        buf22 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf11, buf21, buf22, 16777216, grid=grid(16777216), stream=stream0)
        del buf11
        # Topologically Sorted Source Nodes: [cat_5, x_247], Original ATen: [aten.cat, aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 64, 128, 128), (1048576, 1, 8192, 64))
        del arg31_1
        del buf22
        buf24 = buf23; del buf23  # reuse
        buf25 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4.run(buf24, arg32_1, arg33_1, arg34_1, arg35_1, buf25, 8388608, grid=grid(8388608), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        del buf24
        buf26 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_249, x_250], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_11.run(arg36_1, buf26, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg36_1
        # Topologically Sorted Source Nodes: [x_249, x_250], Original ATen: [aten.leaky_relu, aten.convolution]
        buf27 = extern_kernels.convolution(buf25, buf26, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 128, 64, 64), (524288, 1, 8192, 128))
        del buf25
        del buf26
        buf28 = buf27; del buf27  # reuse
        buf29 = reinterpret_tensor(buf15, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_251, x_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12.run(buf28, arg37_1, arg38_1, arg39_1, arg40_1, buf29, 4194304, grid=grid(4194304), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        del buf28
        # Topologically Sorted Source Nodes: [x_252, x_253], Original ATen: [aten.leaky_relu, aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 128, 64, 64), (524288, 1, 8192, 128))
        del arg41_1
        buf31 = buf30; del buf30  # reuse
        buf32 = reinterpret_tensor(buf29, (8, 128, 64, 64), (524288, 4096, 64, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_254, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13.run(buf31, arg42_1, arg43_1, arg44_1, arg45_1, buf32, 32768, 128, grid=grid(32768, 128), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        buf33 = empty_strided_cuda((8, 64, 64, 64), (262144, 1, 4096, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_256], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf32, buf33, 512, 4096, grid=grid(512, 4096), stream=stream0)
        # Topologically Sorted Source Nodes: [x_256], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del arg46_1
        buf35 = buf34; del buf34  # reuse
        buf36 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_257, x_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_15.run(buf35, arg47_1, arg48_1, arg49_1, arg50_1, buf36, 2097152, grid=grid(2097152), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        del buf35
        buf37 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_258, x_259], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_16.run(arg51_1, buf37, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg51_1
        # Topologically Sorted Source Nodes: [x_258, x_259], Original ATen: [aten.leaky_relu, aten.convolution]
        buf38 = extern_kernels.convolution(buf36, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 64, 64, 64), (262144, 1, 4096, 64))
        buf39 = buf38; del buf38  # reuse
        buf40 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_260, x_261, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_17.run(buf39, arg52_1, arg53_1, arg54_1, arg55_1, buf32, buf40, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        # Topologically Sorted Source Nodes: [x_263], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del arg56_1
        buf42 = buf41; del buf41  # reuse
        buf43 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_264, x_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_15.run(buf42, arg57_1, arg58_1, arg59_1, arg60_1, buf43, 2097152, grid=grid(2097152), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf42
        buf44 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_265, x_266], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_16.run(arg61_1, buf44, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg61_1
        # Topologically Sorted Source Nodes: [x_265, x_266], Original ATen: [aten.leaky_relu, aten.convolution]
        buf45 = extern_kernels.convolution(buf43, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del buf43
        del buf44
        buf46 = buf45; del buf45  # reuse
        buf47 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_267, x_268, x_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_18.run(buf46, buf47, arg62_1, arg63_1, arg64_1, arg65_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        del buf46
        # Topologically Sorted Source Nodes: [x_268, x_269, x_270], Original ATen: [aten.leaky_relu, aten.add, aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del arg66_1
        del buf47
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_271], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf49, arg67_1, arg68_1, arg69_1, arg70_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        buf50 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
        triton_poi_fused_cat_20.run(buf32, buf49, buf50, 4194304, grid=grid(4194304), stream=stream0)
        del buf32
        # Topologically Sorted Source Nodes: [cat_6, x_273], Original ATen: [aten.cat, aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 128, 64, 64), (524288, 1, 8192, 128))
        del arg71_1
        buf52 = buf51; del buf51  # reuse
        buf53 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_274, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12.run(buf52, arg72_1, arg73_1, arg74_1, arg75_1, buf53, 4194304, grid=grid(4194304), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        del buf52
        buf54 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_275, x_276], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_21.run(arg76_1, buf54, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg76_1
        # Topologically Sorted Source Nodes: [x_275, x_276], Original ATen: [aten.leaky_relu, aten.convolution]
        buf55 = extern_kernels.convolution(buf53, buf54, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 256, 32, 32), (262144, 1, 8192, 256))
        del buf53
        del buf54
        buf56 = buf55; del buf55  # reuse
        buf57 = reinterpret_tensor(buf49, (8, 256, 32, 32), (262144, 1, 8192, 256), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [x_277, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22.run(buf56, arg77_1, arg78_1, arg79_1, arg80_1, buf57, 2097152, grid=grid(2097152), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        del buf56
        # Topologically Sorted Source Nodes: [x_278, x_279], Original ATen: [aten.leaky_relu, aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 256, 32, 32), (262144, 1, 8192, 256))
        del arg81_1
        buf59 = buf58; del buf58  # reuse
        buf60 = reinterpret_tensor(buf57, (8, 256, 32, 32), (262144, 1024, 32, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_280, x_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23.run(buf59, arg82_1, arg83_1, arg84_1, arg85_1, buf60, 8192, 256, grid=grid(8192, 256), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        buf61 = empty_strided_cuda((8, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_282], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf60, buf61, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_282], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del arg86_1
        buf63 = buf62; del buf62  # reuse
        buf64 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_283, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf63, arg87_1, arg88_1, arg89_1, arg90_1, buf64, 1048576, grid=grid(1048576), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        del buf63
        buf65 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_284, x_285], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_26.run(arg91_1, buf65, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg91_1
        # Topologically Sorted Source Nodes: [x_284, x_285], Original ATen: [aten.leaky_relu, aten.convolution]
        buf66 = extern_kernels.convolution(buf64, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 128, 32, 32), (131072, 1, 4096, 128))
        buf67 = buf66; del buf66  # reuse
        buf68 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_286, x_287, x_288], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_27.run(buf67, arg92_1, arg93_1, arg94_1, arg95_1, buf60, buf68, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        # Topologically Sorted Source Nodes: [x_289], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del arg96_1
        buf70 = buf69; del buf69  # reuse
        buf71 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_290, x_291], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf70, arg97_1, arg98_1, arg99_1, arg100_1, buf71, 1048576, grid=grid(1048576), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf70
        buf72 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_291, x_292], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_26.run(arg101_1, buf72, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg101_1
        # Topologically Sorted Source Nodes: [x_291, x_292], Original ATen: [aten.leaky_relu, aten.convolution]
        buf73 = extern_kernels.convolution(buf71, buf72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del buf71
        buf74 = buf73; del buf73  # reuse
        buf75 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_293, x_294, x_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_28.run(buf74, buf75, arg102_1, arg103_1, arg104_1, arg105_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        # Topologically Sorted Source Nodes: [x_296], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del arg106_1
        buf77 = buf76; del buf76  # reuse
        buf78 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_297, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf77, arg107_1, arg108_1, arg109_1, arg110_1, buf78, 1048576, grid=grid(1048576), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        del buf77
        buf79 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_298, x_299], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_26.run(arg111_1, buf79, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg111_1
        # Topologically Sorted Source Nodes: [x_298, x_299], Original ATen: [aten.leaky_relu, aten.convolution]
        buf80 = extern_kernels.convolution(buf78, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del buf78
        buf81 = buf80; del buf80  # reuse
        buf82 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_300, x_301, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_28.run(buf81, buf82, arg112_1, arg113_1, arg114_1, arg115_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        # Topologically Sorted Source Nodes: [x_303], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del arg116_1
        buf84 = buf83; del buf83  # reuse
        buf85 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_304, x_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf84, arg117_1, arg118_1, arg119_1, arg120_1, buf85, 1048576, grid=grid(1048576), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        del buf84
        buf86 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [x_305, x_306], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_26.run(arg121_1, buf86, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg121_1
        # Topologically Sorted Source Nodes: [x_305, x_306], Original ATen: [aten.leaky_relu, aten.convolution]
        buf87 = extern_kernels.convolution(buf85, buf86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del buf85
        buf88 = buf87; del buf87  # reuse
        buf89 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [x_307, x_308, x_309], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_28.run(buf88, buf89, arg122_1, arg123_1, arg124_1, arg125_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        # Topologically Sorted Source Nodes: [x_310], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del arg126_1
        buf91 = buf90; del buf90  # reuse
        buf92 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_311, x_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf91, arg127_1, arg128_1, arg129_1, arg130_1, buf92, 1048576, grid=grid(1048576), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        del buf91
        buf93 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_312, x_313], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_26.run(arg131_1, buf93, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg131_1
        # Topologically Sorted Source Nodes: [x_312, x_313], Original ATen: [aten.leaky_relu, aten.convolution]
        buf94 = extern_kernels.convolution(buf92, buf93, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del buf92
        buf95 = buf94; del buf94  # reuse
        buf96 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [x_314, x_315, x_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_28.run(buf95, buf96, arg132_1, arg133_1, arg134_1, arg135_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        # Topologically Sorted Source Nodes: [x_317], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del arg136_1
        buf98 = buf97; del buf97  # reuse
        buf99 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [x_318, x_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf98, arg137_1, arg138_1, arg139_1, arg140_1, buf99, 1048576, grid=grid(1048576), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        del buf98
        buf100 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_319, x_320], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_26.run(arg141_1, buf100, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg141_1
        # Topologically Sorted Source Nodes: [x_319, x_320], Original ATen: [aten.leaky_relu, aten.convolution]
        buf101 = extern_kernels.convolution(buf99, buf100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del buf99
        buf102 = buf101; del buf101  # reuse
        buf103 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_321, x_322, x_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_28.run(buf102, buf103, arg142_1, arg143_1, arg144_1, arg145_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        # Topologically Sorted Source Nodes: [x_324], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del arg146_1
        buf105 = buf104; del buf104  # reuse
        buf106 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_325, x_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf105, arg147_1, arg148_1, arg149_1, arg150_1, buf106, 1048576, grid=grid(1048576), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        del buf105
        buf107 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_326, x_327], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_26.run(arg151_1, buf107, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg151_1
        # Topologically Sorted Source Nodes: [x_326, x_327], Original ATen: [aten.leaky_relu, aten.convolution]
        buf108 = extern_kernels.convolution(buf106, buf107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del buf106
        buf109 = buf108; del buf108  # reuse
        buf110 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [x_328, x_329, x_330], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_28.run(buf109, buf110, arg152_1, arg153_1, arg154_1, arg155_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        # Topologically Sorted Source Nodes: [x_331], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del arg156_1
        buf112 = buf111; del buf111  # reuse
        buf113 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_332, x_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf112, arg157_1, arg158_1, arg159_1, arg160_1, buf113, 1048576, grid=grid(1048576), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        del buf112
        buf114 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [x_333, x_334], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_26.run(arg161_1, buf114, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg161_1
        # Topologically Sorted Source Nodes: [x_333, x_334], Original ATen: [aten.leaky_relu, aten.convolution]
        buf115 = extern_kernels.convolution(buf113, buf114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del buf113
        del buf114
        buf116 = buf115; del buf115  # reuse
        buf117 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_335, x_336, x_337], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_28.run(buf116, buf117, arg162_1, arg163_1, arg164_1, arg165_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        del buf116
        # Topologically Sorted Source Nodes: [x_336, x_337, x_338], Original ATen: [aten.leaky_relu, aten.add, aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del arg166_1
        del buf117
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_339], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_29.run(buf119, arg167_1, arg168_1, arg169_1, arg170_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        buf120 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [cat_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf60, buf119, buf120, 2097152, grid=grid(2097152), stream=stream0)
        del buf60
        # Topologically Sorted Source Nodes: [cat_7, x_341], Original ATen: [aten.cat, aten.convolution]
        buf121 = extern_kernels.convolution(buf120, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 256, 32, 32), (262144, 1, 8192, 256))
        del arg171_1
        buf122 = buf121; del buf121  # reuse
        buf123 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [x_342, x_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22.run(buf122, arg172_1, arg173_1, arg174_1, arg175_1, buf123, 2097152, grid=grid(2097152), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        del buf122
        buf124 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_343, x_344], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_31.run(arg176_1, buf124, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg176_1
        # Topologically Sorted Source Nodes: [x_343, x_344], Original ATen: [aten.leaky_relu, aten.convolution]
        buf125 = extern_kernels.convolution(buf123, buf124, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 512, 16, 16), (131072, 1, 8192, 512))
        del buf123
        del buf124
        buf126 = buf125; del buf125  # reuse
        buf127 = reinterpret_tensor(buf119, (8, 512, 16, 16), (131072, 1, 8192, 512), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [x_345, x_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_32.run(buf126, arg177_1, arg178_1, arg179_1, arg180_1, buf127, 1048576, grid=grid(1048576), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        del buf126
        # Topologically Sorted Source Nodes: [x_346, x_347], Original ATen: [aten.leaky_relu, aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 512, 16, 16), (131072, 1, 8192, 512))
        del arg181_1
        buf129 = buf128; del buf128  # reuse
        buf130 = reinterpret_tensor(buf127, (8, 512, 16, 16), (131072, 256, 16, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_348, x_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_33.run(buf129, arg182_1, arg183_1, arg184_1, arg185_1, buf130, 2048, 512, grid=grid(2048, 512), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        buf131 = empty_strided_cuda((8, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_350], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf130, buf131, 2048, 256, grid=grid(2048, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [x_350], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del arg186_1
        buf133 = buf132; del buf132  # reuse
        buf134 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_351, x_352], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf133, arg187_1, arg188_1, arg189_1, arg190_1, buf134, 524288, grid=grid(524288), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        del buf133
        buf135 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_352, x_353], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_36.run(arg191_1, buf135, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg191_1
        # Topologically Sorted Source Nodes: [x_352, x_353], Original ATen: [aten.leaky_relu, aten.convolution]
        buf136 = extern_kernels.convolution(buf134, buf135, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 256, 16, 16), (65536, 1, 4096, 256))
        buf137 = buf136; del buf136  # reuse
        buf138 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [x_354, x_355, x_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_37.run(buf137, arg192_1, arg193_1, arg194_1, arg195_1, buf130, buf138, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        # Topologically Sorted Source Nodes: [x_357], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del arg196_1
        buf140 = buf139; del buf139  # reuse
        buf141 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [x_358, x_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf140, arg197_1, arg198_1, arg199_1, arg200_1, buf141, 524288, grid=grid(524288), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        del buf140
        buf142 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [x_359, x_360], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_36.run(arg201_1, buf142, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg201_1
        # Topologically Sorted Source Nodes: [x_359, x_360], Original ATen: [aten.leaky_relu, aten.convolution]
        buf143 = extern_kernels.convolution(buf141, buf142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del buf141
        buf144 = buf143; del buf143  # reuse
        buf145 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [x_361, x_362, x_363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_38.run(buf144, buf145, arg202_1, arg203_1, arg204_1, arg205_1, 524288, grid=grid(524288), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        # Topologically Sorted Source Nodes: [x_364], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del arg206_1
        buf147 = buf146; del buf146  # reuse
        buf148 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_365, x_366], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf147, arg207_1, arg208_1, arg209_1, arg210_1, buf148, 524288, grid=grid(524288), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        del buf147
        buf149 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [x_366, x_367], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_36.run(arg211_1, buf149, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg211_1
        # Topologically Sorted Source Nodes: [x_366, x_367], Original ATen: [aten.leaky_relu, aten.convolution]
        buf150 = extern_kernels.convolution(buf148, buf149, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del buf148
        buf151 = buf150; del buf150  # reuse
        buf152 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [x_368, x_369, x_370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_38.run(buf151, buf152, arg212_1, arg213_1, arg214_1, arg215_1, 524288, grid=grid(524288), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        # Topologically Sorted Source Nodes: [x_371], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del arg216_1
        buf154 = buf153; del buf153  # reuse
        buf155 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [x_372, x_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf154, arg217_1, arg218_1, arg219_1, arg220_1, buf155, 524288, grid=grid(524288), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        del buf154
        buf156 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [x_373, x_374], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_36.run(arg221_1, buf156, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg221_1
        # Topologically Sorted Source Nodes: [x_373, x_374], Original ATen: [aten.leaky_relu, aten.convolution]
        buf157 = extern_kernels.convolution(buf155, buf156, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del buf155
        buf158 = buf157; del buf157  # reuse
        buf159 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [x_375, x_376, x_377], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_38.run(buf158, buf159, arg222_1, arg223_1, arg224_1, arg225_1, 524288, grid=grid(524288), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        # Topologically Sorted Source Nodes: [x_378], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del arg226_1
        buf161 = buf160; del buf160  # reuse
        buf162 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [x_379, x_380], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf161, arg227_1, arg228_1, arg229_1, arg230_1, buf162, 524288, grid=grid(524288), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        del buf161
        buf163 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [x_380, x_381], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_36.run(arg231_1, buf163, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg231_1
        # Topologically Sorted Source Nodes: [x_380, x_381], Original ATen: [aten.leaky_relu, aten.convolution]
        buf164 = extern_kernels.convolution(buf162, buf163, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del buf162
        buf165 = buf164; del buf164  # reuse
        buf166 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_382, x_383, x_384], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_38.run(buf165, buf166, arg232_1, arg233_1, arg234_1, arg235_1, 524288, grid=grid(524288), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        # Topologically Sorted Source Nodes: [x_385], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del arg236_1
        buf168 = buf167; del buf167  # reuse
        buf169 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_386, x_387], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf168, arg237_1, arg238_1, arg239_1, arg240_1, buf169, 524288, grid=grid(524288), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        del buf168
        buf170 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_387, x_388], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_36.run(arg241_1, buf170, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg241_1
        # Topologically Sorted Source Nodes: [x_387, x_388], Original ATen: [aten.leaky_relu, aten.convolution]
        buf171 = extern_kernels.convolution(buf169, buf170, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del buf169
        buf172 = buf171; del buf171  # reuse
        buf173 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [x_389, x_390, x_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_38.run(buf172, buf173, arg242_1, arg243_1, arg244_1, arg245_1, 524288, grid=grid(524288), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        # Topologically Sorted Source Nodes: [x_392], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del arg246_1
        buf175 = buf174; del buf174  # reuse
        buf176 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [x_393, x_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf175, arg247_1, arg248_1, arg249_1, arg250_1, buf176, 524288, grid=grid(524288), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        del buf175
        buf177 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [x_394, x_395], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_36.run(arg251_1, buf177, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg251_1
        # Topologically Sorted Source Nodes: [x_394, x_395], Original ATen: [aten.leaky_relu, aten.convolution]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del buf176
        buf179 = buf178; del buf178  # reuse
        buf180 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [x_396, x_397, x_398], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_38.run(buf179, buf180, arg252_1, arg253_1, arg254_1, arg255_1, 524288, grid=grid(524288), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        # Topologically Sorted Source Nodes: [x_399], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del arg256_1
        buf182 = buf181; del buf181  # reuse
        buf183 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_400, x_401], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf182, arg257_1, arg258_1, arg259_1, arg260_1, buf183, 524288, grid=grid(524288), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        del buf182
        buf184 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_401, x_402], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_36.run(arg261_1, buf184, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg261_1
        # Topologically Sorted Source Nodes: [x_401, x_402], Original ATen: [aten.leaky_relu, aten.convolution]
        buf185 = extern_kernels.convolution(buf183, buf184, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del buf183
        del buf184
        buf186 = buf185; del buf185  # reuse
        buf187 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [x_403, x_404, x_405], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_38.run(buf186, buf187, arg262_1, arg263_1, arg264_1, arg265_1, 524288, grid=grid(524288), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        del buf186
        # Topologically Sorted Source Nodes: [x_404, x_405, x_406], Original ATen: [aten.leaky_relu, aten.add, aten.convolution]
        buf188 = extern_kernels.convolution(buf187, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del arg266_1
        del buf187
        buf189 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_407], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_39.run(buf189, arg267_1, arg268_1, arg269_1, arg270_1, 524288, grid=grid(524288), stream=stream0)
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        buf190 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [cat_8], Original ATen: [aten.cat]
        triton_poi_fused_cat_40.run(buf130, buf189, buf190, 1048576, grid=grid(1048576), stream=stream0)
        del buf130
        # Topologically Sorted Source Nodes: [cat_8, x_409], Original ATen: [aten.cat, aten.convolution]
        buf191 = extern_kernels.convolution(buf190, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 512, 16, 16), (131072, 1, 8192, 512))
        del arg271_1
        buf192 = buf191; del buf191  # reuse
        buf193 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_410, x_411], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_32.run(buf192, arg272_1, arg273_1, arg274_1, arg275_1, buf193, 1048576, grid=grid(1048576), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        del buf192
        buf194 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_411, x_412], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_41.run(arg276_1, buf194, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del arg276_1
        # Topologically Sorted Source Nodes: [x_411, x_412], Original ATen: [aten.leaky_relu, aten.convolution]
        buf195 = extern_kernels.convolution(buf193, buf194, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
        del buf193
        del buf194
        buf196 = buf195; del buf195  # reuse
        buf197 = reinterpret_tensor(buf189, (8, 1024, 8, 8), (65536, 1, 8192, 1024), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_413, x_414], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_42.run(buf196, arg277_1, arg278_1, arg279_1, arg280_1, buf197, 524288, grid=grid(524288), stream=stream0)
        del arg277_1
        del arg278_1
        del arg279_1
        del arg280_1
        del buf196
        # Topologically Sorted Source Nodes: [x_414, x_415], Original ATen: [aten.leaky_relu, aten.convolution]
        buf198 = extern_kernels.convolution(buf197, arg281_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
        del arg281_1
        buf199 = buf198; del buf198  # reuse
        buf200 = reinterpret_tensor(buf197, (8, 1024, 8, 8), (65536, 64, 8, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [x_416, x_417], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_43.run(buf199, arg282_1, arg283_1, arg284_1, arg285_1, buf200, 512, 1024, grid=grid(512, 1024), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        buf201 = empty_strided_cuda((8, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_418], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf200, buf201, 4096, 64, grid=grid(4096, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [x_418], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 512, 8, 8), (32768, 1, 4096, 512))
        del arg286_1
        buf203 = buf202; del buf202  # reuse
        buf204 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [x_419, x_420], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf203, arg287_1, arg288_1, arg289_1, arg290_1, buf204, 262144, grid=grid(262144), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        del buf203
        buf205 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_420, x_421], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_46.run(arg291_1, buf205, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg291_1
        # Topologically Sorted Source Nodes: [x_420, x_421], Original ATen: [aten.leaky_relu, aten.convolution]
        buf206 = extern_kernels.convolution(buf204, buf205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 512, 8, 8), (32768, 1, 4096, 512))
        buf207 = buf206; del buf206  # reuse
        buf208 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_422, x_423, x_424], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_47.run(buf207, arg292_1, arg293_1, arg294_1, arg295_1, buf200, buf208, 512, 512, grid=grid(512, 512), stream=stream0)
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        # Topologically Sorted Source Nodes: [x_425], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (8, 512, 8, 8), (32768, 1, 4096, 512))
        del arg296_1
        buf210 = buf209; del buf209  # reuse
        buf211 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_426, x_427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf210, arg297_1, arg298_1, arg299_1, arg300_1, buf211, 262144, grid=grid(262144), stream=stream0)
        del arg297_1
        del arg298_1
        del arg299_1
        del arg300_1
        del buf210
        buf212 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [x_427, x_428], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_46.run(arg301_1, buf212, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg301_1
        # Topologically Sorted Source Nodes: [x_427, x_428], Original ATen: [aten.leaky_relu, aten.convolution]
        buf213 = extern_kernels.convolution(buf211, buf212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 512, 8, 8), (32768, 1, 4096, 512))
        del buf211
        buf214 = buf213; del buf213  # reuse
        buf215 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [x_429, x_430, x_431], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_48.run(buf214, buf215, arg302_1, arg303_1, arg304_1, arg305_1, 262144, grid=grid(262144), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        # Topologically Sorted Source Nodes: [x_432], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 512, 8, 8), (32768, 1, 4096, 512))
        del arg306_1
        buf217 = buf216; del buf216  # reuse
        buf218 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [x_433, x_434], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf217, arg307_1, arg308_1, arg309_1, arg310_1, buf218, 262144, grid=grid(262144), stream=stream0)
        del arg307_1
        del arg308_1
        del arg309_1
        del arg310_1
        del buf217
        buf219 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [x_434, x_435], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_46.run(arg311_1, buf219, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg311_1
        # Topologically Sorted Source Nodes: [x_434, x_435], Original ATen: [aten.leaky_relu, aten.convolution]
        buf220 = extern_kernels.convolution(buf218, buf219, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (8, 512, 8, 8), (32768, 1, 4096, 512))
        del buf218
        buf221 = buf220; del buf220  # reuse
        buf222 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [x_436, x_437, x_438], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_48.run(buf221, buf222, arg312_1, arg313_1, arg314_1, arg315_1, 262144, grid=grid(262144), stream=stream0)
        del arg312_1
        del arg313_1
        del arg314_1
        del arg315_1
        # Topologically Sorted Source Nodes: [x_439], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 512, 8, 8), (32768, 1, 4096, 512))
        del arg316_1
        buf224 = buf223; del buf223  # reuse
        buf225 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [x_440, x_441], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf224, arg317_1, arg318_1, arg319_1, arg320_1, buf225, 262144, grid=grid(262144), stream=stream0)
        del arg317_1
        del arg318_1
        del arg319_1
        del arg320_1
        del buf224
        buf226 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_441, x_442], Original ATen: [aten.leaky_relu, aten.convolution]
        triton_poi_fused_convolution_leaky_relu_46.run(arg321_1, buf226, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg321_1
        # Topologically Sorted Source Nodes: [x_441, x_442], Original ATen: [aten.leaky_relu, aten.convolution]
        buf227 = extern_kernels.convolution(buf225, buf226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 512, 8, 8), (32768, 1, 4096, 512))
        del buf225
        del buf226
        buf228 = buf227; del buf227  # reuse
        buf229 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [x_443, x_444, x_445], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_48.run(buf228, buf229, arg322_1, arg323_1, arg324_1, arg325_1, 262144, grid=grid(262144), stream=stream0)
        del arg322_1
        del arg323_1
        del arg324_1
        del arg325_1
        del buf228
        # Topologically Sorted Source Nodes: [x_444, x_445, x_446], Original ATen: [aten.leaky_relu, aten.add, aten.convolution]
        buf230 = extern_kernels.convolution(buf229, arg326_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (8, 512, 8, 8), (32768, 1, 4096, 512))
        del arg326_1
        del buf229
        buf231 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [x_447], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_49.run(buf231, arg327_1, arg328_1, arg329_1, arg330_1, 262144, grid=grid(262144), stream=stream0)
        del arg327_1
        del arg328_1
        del arg329_1
        del arg330_1
        buf232 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [cat_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_50.run(buf200, buf231, buf232, 524288, grid=grid(524288), stream=stream0)
        del buf200
        del buf231
        # Topologically Sorted Source Nodes: [cat_9, x_449], Original ATen: [aten.cat, aten.convolution]
        buf233 = extern_kernels.convolution(buf232, arg331_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
        del arg331_1
        del buf232
        buf234 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_450], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_51.run(buf234, arg332_1, arg333_1, arg334_1, arg335_1, 524288, grid=grid(524288), stream=stream0)
        del arg332_1
        del arg333_1
        del arg334_1
        del arg335_1
        buf236 = empty_strided_cuda((8, 1024, 1, 1), (1024, 1, 8192, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [x_451, x_452], Original ATen: [aten.leaky_relu, aten.mean]
        triton_per_fused_leaky_relu_mean_52.run(buf234, buf236, 8192, 64, grid=grid(8192), stream=stream0)
        del buf234
        buf237 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_455], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg337_1, reinterpret_tensor(buf236, (8, 1024), (1024, 1), 0), reinterpret_tensor(arg336_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf237)
        del arg336_1
        del arg337_1
        del buf236
    return (buf237, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg216_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('cspdarknet53', benchmark_compiled_module)
