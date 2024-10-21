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
# Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_74 => convolution_95
# Graph fragment:
#   %convolution_95 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/lt/cltiu3szd73oc7uccpazylbbmynvcroh24nhc4luunkmkbitxhos.py
# Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_74 => convolution_95
# Graph fragment:
#   %convolution_95 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[64, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
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


# kernel path: /tmp/torchinductor_sahanp/bl/cblq32qcnxob3mnta5br2pgd5fqh55tqnsrh2iov7riitss2qwgi.py
# Topologically Sorted Source Nodes: [x_75, x_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_75 => add_184, mul_248, mul_249, sub_80
#   x_76 => relu_42
# Graph fragment:
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_95, %unsqueeze_641), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_643), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_248, %unsqueeze_645), kwargs = {})
#   %add_184 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_249, %unsqueeze_647), kwargs = {})
#   %relu_42 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_184,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/a2/ca2vrsvuhfuni5jkvyp56djajkukx4xkxkcqbr4lw6yfjt6ghllt.py
# Topologically Sorted Source Nodes: [input_182, input_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_182 => add_186, mul_251, mul_252, sub_81
#   input_183 => relu_43
# Graph fragment:
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_96, %unsqueeze_649), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_651), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_251, %unsqueeze_653), kwargs = {})
#   %add_186 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_252, %unsqueeze_655), kwargs = {})
#   %relu_43 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_186,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 8
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


# kernel path: /tmp/torchinductor_sahanp/oh/cohfp4gvophvmsye37yz6lnvsbkx5bjvbnjkmnwhfqudp3yuihb5.py
# Topologically Sorted Source Nodes: [out_32], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_32 => cat_32
# Graph fragment:
#   %cat_32 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_43, %relu_44], 1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((8*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 16, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((8*x1) + ((-8) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-8) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-8) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-8) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-8) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x2), tmp29, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/io/cioiw6x5sv5i2zkzh66kyorsceyig76ygj3rcdqjuhn6gb2lty5w.py
# Topologically Sorted Source Nodes: [input_188], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_188 => add_190, mul_257, mul_258, sub_83
# Graph fragment:
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_98, %unsqueeze_665), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_667), kwargs = {})
#   %mul_258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_257, %unsqueeze_669), kwargs = {})
#   %add_190 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_258, %unsqueeze_671), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 8
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


# kernel path: /tmp/torchinductor_sahanp/36/c36nkzopalmzhcaigrcmvfrh3j7q57iw7ul3drddt4zxlcgeqbkw.py
# Topologically Sorted Source Nodes: [out_33, x_79], Original ATen: [aten.cat, aten.add]
# Source node to ATen node mapping:
#   out_33 => cat_33
#   x_79 => add_193
# Graph fragment:
#   %cat_33 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_190, %add_192], 1), kwargs = {})
#   %add_193 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_33, %relu_42), kwargs = {})
triton_poi_fused_add_cat_6 = async_compile.triton('triton_poi_fused_add_cat_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16)
    x2 = xindex
    tmp28 = tl.load(in_out_ptr0 + (x2), None)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((8*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 16, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((8*x1) + ((-8) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-8) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-8) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-8) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-8) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp5, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mz/cmzkul3ajnbsdpjj6a5bcu3wlbyjyww6mv5ppggdnenxtoea3w2l.py
# Topologically Sorted Source Nodes: [input_192, input_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_192 => add_195, mul_263, mul_264, sub_85
#   input_193 => relu_45
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_681), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_264 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_263, %unsqueeze_685), kwargs = {})
#   %add_195 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_264, %unsqueeze_687), kwargs = {})
#   %relu_45 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_195,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ej/cejqy27qj27f5b4b6ounbqxbio6qleg3r6jrojna52msz5edjnyc.py
# Topologically Sorted Source Nodes: [out_34], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_34 => cat_34
# Graph fragment:
#   %cat_34 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_45, %relu_46], 1), kwargs = {})
triton_poi_fused_cat_8 = async_compile.triton('triton_poi_fused_cat_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 48
    x1 = (xindex // 48)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((24*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 48, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((24*x1) + ((-24) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-24) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-24) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-24) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-24) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x2), tmp29, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ik/cik3o7snbutx563l6qotlyuo5fcc6trb325n4dt2yuu46otbhbir.py
# Topologically Sorted Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_82 => add_199, mul_269, mul_270, sub_87
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_102, %unsqueeze_697), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_269, %unsqueeze_701), kwargs = {})
#   %add_199 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_270, %unsqueeze_703), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/63/c633yitngaiixdlzcclwigqehgatdktk3x7746ftdb35auu7qkwn.py
# Topologically Sorted Source Nodes: [input_198], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_198 => add_201, mul_272, mul_273, sub_88
# Graph fragment:
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_103, %unsqueeze_705), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_707), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_272, %unsqueeze_709), kwargs = {})
#   %add_201 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_273, %unsqueeze_711), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 12
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


# kernel path: /tmp/torchinductor_sahanp/sd/csdsusugmxebsr45w4n4h2qtb27w4obxzukk33oaw7xvg34igwep.py
# Topologically Sorted Source Nodes: [input_202], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_202 => add_205, mul_278, mul_279, sub_90
# Graph fragment:
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_721), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_723), kwargs = {})
#   %mul_279 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_278, %unsqueeze_725), kwargs = {})
#   %add_205 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_279, %unsqueeze_727), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
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


# kernel path: /tmp/torchinductor_sahanp/rm/crmfvu7cntdx2vfnfbca2ngoeydkqorzokakp3pt2jdyyqowp5fx.py
# Topologically Sorted Source Nodes: [out_35, input_204, x_84], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_204 => add_207, mul_281, mul_282, sub_91
#   out_35 => cat_35
#   x_84 => add_208
# Graph fragment:
#   %cat_35 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_201, %add_203], 1), kwargs = {})
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_106, %unsqueeze_729), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %unsqueeze_731), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_281, %unsqueeze_733), kwargs = {})
#   %add_207 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_282, %unsqueeze_735), kwargs = {})
#   %add_208 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_35, %add_207), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 24
    x1 = (xindex // 24)
    x2 = xindex
    tmp28 = tl.load(in_out_ptr0 + (x2), None)
    tmp29 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((12*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 24, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((12*x1) + ((-12) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-12) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-12) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-12) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-12) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp5, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 + tmp13
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tmp16 / tmp33
    tmp35 = tmp34 * tmp18
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = tmp27 + tmp40
    tl.store(in_out_ptr0 + (x2), tmp41, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h4/ch423oxmxmvobircrsvmumwjr7uokhi3icnqvtbsny2fh7lvvf7l.py
# Topologically Sorted Source Nodes: [input_206, input_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_206 => add_210, mul_284, mul_285, sub_92
#   input_207 => relu_47
# Graph fragment:
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_107, %unsqueeze_737), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_284, %unsqueeze_741), kwargs = {})
#   %add_210 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_285, %unsqueeze_743), kwargs = {})
#   %relu_47 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_210,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
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


# kernel path: /tmp/torchinductor_sahanp/jc/cjccqeknnjj5yhzvyf7ykl4wfrqeqyl4kg5clfhptnyqpqtmqupy.py
# Topologically Sorted Source Nodes: [out_36], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_36 => cat_36
# Graph fragment:
#   %cat_36 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_47, %relu_48], 1), kwargs = {})
triton_poi_fused_cat_14 = async_compile.triton('triton_poi_fused_cat_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 72
    x1 = (xindex // 72)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 36, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((36*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 72, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((36*x1) + ((-36) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-36) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-36) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-36) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-36) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x2), tmp29, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/px/cpxolq5tdfwuuqslg4hk3tl5wrksrzs6q6lixcagfzfjjcip4wfw.py
# Topologically Sorted Source Nodes: [out_37, x_87], Original ATen: [aten.cat, aten.add]
# Source node to ATen node mapping:
#   out_37 => cat_37
#   x_87 => add_217
# Graph fragment:
#   %cat_37 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_214, %add_216], 1), kwargs = {})
#   %add_217 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_37, %add_208), kwargs = {})
triton_poi_fused_add_cat_15 = async_compile.triton('triton_poi_fused_add_cat_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 24
    x1 = (xindex // 24)
    x2 = xindex
    tmp28 = tl.load(in_out_ptr0 + (x2), None)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((12*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 24, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((12*x1) + ((-12) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-12) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-12) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-12) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-12) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp5, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/so/csoyd7a53olnzlpphclej2ucjgonuhzt4vdgqbantsan6yr74pri.py
# Topologically Sorted Source Nodes: [x_90], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_90 => add_223, mul_302, mul_303, sub_98
# Graph fragment:
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_113, %unsqueeze_785), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_98, %unsqueeze_787), kwargs = {})
#   %mul_303 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_302, %unsqueeze_789), kwargs = {})
#   %add_223 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_303, %unsqueeze_791), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ee/cee6nwc4352nm53vbcnyefvfuurnlax2amixre46bu3tqjzjntkf.py
# Topologically Sorted Source Nodes: [x_se_28], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_28 => mean_8
# Graph fragment:
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_223, [2, 3], True), kwargs = {})
triton_red_fused_mean_17 = async_compile.triton('triton_red_fused_mean_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_17(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4032
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 72
    x1 = (xindex // 72)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (8064*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dy/cdywcmqs6svch7bshqmqhezkeqjx3st64l7zz7zmat4np36icjkr.py
# Topologically Sorted Source Nodes: [x_se_28], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_28 => mean_8
# Graph fragment:
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_223, [2, 3], True), kwargs = {})
triton_per_fused_mean_18 = async_compile.triton('triton_per_fused_mean_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_18(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 72
    x1 = (xindex // 72)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (504*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jj/cjj6e6crmkm3zdq3t6o4ociqzh27hxeqxwc435robr2tz7titctn.py
# Topologically Sorted Source Nodes: [x_se_28, x_se_29, x_se_30], Original ATen: [aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_se_28 => mean_8
#   x_se_29 => convolution_114
#   x_se_30 => relu_51
# Graph fragment:
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_223, [2, 3], True), kwargs = {})
#   %convolution_114 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_8, %arg96_1, %arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_51 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_114,), kwargs = {})
triton_poi_fused_convolution_mean_relu_19 = async_compile.triton('triton_poi_fused_convolution_mean_relu_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 20
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/o2/co2ornk2fawve4y34uurvdks5hwcdy424nmqbpsemrqyu6uwxj2v.py
# Topologically Sorted Source Nodes: [x_se_28, x_se_29, x_se_30, x_se_31, hardsigmoid_7, x_91], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_7 => add_224, clamp_max_7, clamp_min_7, div_7
#   x_91 => mul_304
#   x_se_28 => mean_8
#   x_se_29 => convolution_114
#   x_se_30 => relu_51
#   x_se_31 => convolution_115
# Graph fragment:
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_223, [2, 3], True), kwargs = {})
#   %convolution_114 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_8, %arg96_1, %arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_51 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_114,), kwargs = {})
#   %convolution_115 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_51, %arg98_1, %arg99_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_224 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_115, 3), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_224, 0), kwargs = {})
#   %clamp_max_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 6), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_7, 6), kwargs = {})
#   %mul_304 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_223, %div_7), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_20 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_20(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 72
    x2 = (xindex // 56448)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (72*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = 0.16666666666666666
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yr/cyryhgtve5o3zb6mbivgcrg6ibqufyjw6e7npcqywm4rsx4i7s5w.py
# Topologically Sorted Source Nodes: [input_222], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_222 => add_226, mul_306, mul_307, sub_99
# Graph fragment:
#   %sub_99 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_116, %unsqueeze_793), kwargs = {})
#   %mul_306 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_99, %unsqueeze_795), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_306, %unsqueeze_797), kwargs = {})
#   %add_226 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_307, %unsqueeze_799), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 20
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


# kernel path: /tmp/torchinductor_sahanp/o5/co573dvtz4b23rjjx2bs3mk5op2575psi3vkysbdwupojqzvffwe.py
# Topologically Sorted Source Nodes: [input_226], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_226 => add_230, mul_312, mul_313, sub_101
# Graph fragment:
#   %sub_101 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_118, %unsqueeze_809), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_101, %unsqueeze_811), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_312, %unsqueeze_813), kwargs = {})
#   %add_230 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_313, %unsqueeze_815), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
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


# kernel path: /tmp/torchinductor_sahanp/ns/cnsokjkhbexfu4rzb3cwvapu7xiarvmpjwzy4voudnte52ooas7s.py
# Topologically Sorted Source Nodes: [out_39, input_228, x_93], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_228 => add_232, mul_315, mul_316, sub_102
#   out_39 => cat_39
#   x_93 => add_233
# Graph fragment:
#   %cat_39 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_226, %add_228], 1), kwargs = {})
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_119, %unsqueeze_817), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %unsqueeze_819), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_315, %unsqueeze_821), kwargs = {})
#   %add_232 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_316, %unsqueeze_823), kwargs = {})
#   %add_233 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_39, %add_232), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40)
    x2 = xindex
    tmp28 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp29 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((20*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 40, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((20*x1) + ((-20) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-20) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-20) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-20) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-20) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp5, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 + tmp13
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tmp16 / tmp33
    tmp35 = tmp34 * tmp18
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = tmp27 + tmp40
    tl.store(in_out_ptr0 + (x2), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bp/cbpp4t3ejosz36msel64dfpkorgighgf7dvpjdkhktoa6dwwiosk.py
# Topologically Sorted Source Nodes: [input_230, input_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_230 => add_235, mul_318, mul_319, sub_103
#   input_231 => relu_52
# Graph fragment:
#   %sub_103 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_120, %unsqueeze_825), kwargs = {})
#   %mul_318 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_103, %unsqueeze_827), kwargs = {})
#   %mul_319 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_318, %unsqueeze_829), kwargs = {})
#   %add_235 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_319, %unsqueeze_831), kwargs = {})
#   %relu_52 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_235,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 60
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


# kernel path: /tmp/torchinductor_sahanp/v5/cv55n2rymqjriklzy5yxd33iwyeulz6mqmhw4chlrwlar35bz2sn.py
# Topologically Sorted Source Nodes: [out_40, x_se_32], Original ATen: [aten.cat, aten.mean]
# Source node to ATen node mapping:
#   out_40 => cat_40
#   x_se_32 => mean_9
# Graph fragment:
#   %cat_40 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_52, %relu_53], 1), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_40, [2, 3], True), kwargs = {})
triton_red_fused_cat_mean_25 = async_compile.triton('triton_red_fused_cat_mean_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_mean_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_mean_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 120
    x1 = (xindex // 120)
    x3 = xindex
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 60, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((60*r2) + (47040*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 120, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((60*r2) + (47040*x1) + ((-60) + x0)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to((-60) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tl.load(in_ptr3 + (tl.broadcast_to((-60) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.sqrt(tmp14)
        tmp16 = tl.full([1, 1], 1, tl.int32)
        tmp17 = tmp16 / tmp15
        tmp18 = 1.0
        tmp19 = tmp17 * tmp18
        tmp20 = tmp11 * tmp19
        tmp21 = tl.load(in_ptr4 + (tl.broadcast_to((-60) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 * tmp21
        tmp23 = tl.load(in_ptr5 + (tl.broadcast_to((-60) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 + tmp23
        tmp25 = tl.full([1, 1], 0, tl.int32)
        tmp26 = triton_helpers.maximum(tmp25, tmp24)
        tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
        tmp28 = tl.where(tmp6, tmp26, tmp27)
        tmp29 = tl.where(tmp4, tmp5, tmp28)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
        tl.store(out_ptr0 + (r2 + (784*x3)), tmp29, rmask & xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tmp33 = 784.0
    tmp34 = tmp31 / tmp33
    tl.store(out_ptr2 + (x3), tmp34, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dn/cdnl3j3fi7hauh3i422d6smcgbdlzeoo7n3kxxwki6ky57yifzgc.py
# Topologically Sorted Source Nodes: [x_se_32, x_se_33, x_se_34], Original ATen: [aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_se_32 => mean_9
#   x_se_33 => convolution_122
#   x_se_34 => relu_54
# Graph fragment:
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_40, [2, 3], True), kwargs = {})
#   %convolution_122 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_9, %arg130_1, %arg131_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_54 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_122,), kwargs = {})
triton_poi_fused_convolution_mean_relu_26 = async_compile.triton('triton_poi_fused_convolution_mean_relu_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_26(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gr/cgrqnxhtsf7uzflqn3l547gdd7wcrwgbt2nvoqexykubhihu3q2i.py
# Topologically Sorted Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35, hardsigmoid_8, x_95], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_8 => add_238, clamp_max_8, clamp_min_8, div_8
#   x_95 => mul_323
#   x_se_32 => mean_9
#   x_se_33 => convolution_122
#   x_se_34 => relu_54
#   x_se_35 => convolution_123
# Graph fragment:
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_40, [2, 3], True), kwargs = {})
#   %convolution_122 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_9, %arg130_1, %arg131_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_54 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_122,), kwargs = {})
#   %convolution_123 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_54, %arg132_1, %arg133_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_238 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_123, 3), kwargs = {})
#   %clamp_min_8 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_238, 0), kwargs = {})
#   %clamp_max_8 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_8, 6), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_8, 6), kwargs = {})
#   %mul_323 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_40, %div_8), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_27 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_27(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = 0.16666666666666666
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (y0 + (120*x2) + (94080*y1)), tmp12, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vh/cvh7fm7dynniw6zut6qu6kmfqcrjnxd22y5id3fccomvhgangwty.py
# Topologically Sorted Source Nodes: [out_41, x_97], Original ATen: [aten.cat, aten.add]
# Source node to ATen node mapping:
#   out_41 => cat_41
#   x_97 => add_243
# Graph fragment:
#   %cat_41 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_240, %add_242], 1), kwargs = {})
#   %add_243 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_41, %add_233), kwargs = {})
triton_poi_fused_add_cat_28 = async_compile.triton('triton_poi_fused_add_cat_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40)
    x2 = xindex
    tmp28 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((20*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 40, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((20*x1) + ((-20) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-20) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-20) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-20) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-20) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp5, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qw/cqwe576qyeqis6jq5wbmemllhshg4vay3hkiwi4gg3eqbiqfd57c.py
# Topologically Sorted Source Nodes: [input_240, input_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_240 => add_245, mul_331, mul_332, sub_107
#   input_241 => relu_55
# Graph fragment:
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_126, %unsqueeze_857), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %unsqueeze_859), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_331, %unsqueeze_861), kwargs = {})
#   %add_245 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_332, %unsqueeze_863), kwargs = {})
#   %relu_55 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_245,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/jl/cjl5zrwmf4ttsjjuungm5f7ogcfptmyutj67kk46cpzcyg63ejmb.py
# Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_42 => cat_42
# Graph fragment:
#   %cat_42 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_55, %relu_56], 1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 240
    x1 = (xindex // 240)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 120, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((120*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 240, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((120*x1) + ((-120) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-120) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-120) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-120) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-120) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/p6/cp6rhhz7fai4zocz5naxcfmru3vvcmjqsv7cxpb55dc2e623eynx.py
# Topologically Sorted Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_100 => add_249, mul_337, mul_338, sub_109
# Graph fragment:
#   %sub_109 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_128, %unsqueeze_873), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_109, %unsqueeze_875), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_337, %unsqueeze_877), kwargs = {})
#   %add_249 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_338, %unsqueeze_879), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fw/cfwuobviqf35ehhdei7wheyd7dyzxcbe2k4as25o4rfsosdd2g5i.py
# Topologically Sorted Source Nodes: [input_246], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_246 => add_251, mul_340, mul_341, sub_110
# Graph fragment:
#   %sub_110 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_129, %unsqueeze_881), kwargs = {})
#   %mul_340 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_110, %unsqueeze_883), kwargs = {})
#   %mul_341 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_340, %unsqueeze_885), kwargs = {})
#   %add_251 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_341, %unsqueeze_887), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
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


# kernel path: /tmp/torchinductor_sahanp/ss/cssouhmd3lumysbj6iiasakvbdbugksjlzrl25za2t3qh2rjj5jd.py
# Topologically Sorted Source Nodes: [out_43, input_252, x_102], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_252 => add_257, mul_349, mul_350, sub_113
#   out_43 => cat_43
#   x_102 => add_258
# Graph fragment:
#   %cat_43 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_251, %add_253], 1), kwargs = {})
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_132, %unsqueeze_905), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %unsqueeze_907), kwargs = {})
#   %mul_350 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_349, %unsqueeze_909), kwargs = {})
#   %add_257 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_350, %unsqueeze_911), kwargs = {})
#   %add_258 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_43, %add_257), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 80
    x1 = (xindex // 80)
    x2 = xindex
    tmp28 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp29 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 40, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((40*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 80, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((40*x1) + ((-40) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-40) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-40) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-40) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-40) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp5, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 + tmp13
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tmp16 / tmp33
    tmp35 = tmp34 * tmp18
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = tmp27 + tmp40
    tl.store(in_out_ptr0 + (x2), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gj/cgjtcrdsiw2pzf2m3oq3kw6qdzj7iwq45yd2hxmj23ku7czc3nal.py
# Topologically Sorted Source Nodes: [input_254, input_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_254 => add_260, mul_352, mul_353, sub_114
#   input_255 => relu_57
# Graph fragment:
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_133, %unsqueeze_913), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_915), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, %unsqueeze_917), kwargs = {})
#   %add_260 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_353, %unsqueeze_919), kwargs = {})
#   %relu_57 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_260,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 100
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


# kernel path: /tmp/torchinductor_sahanp/ow/cowdaa2tlucdrjrk45wi52ydlktfptomtea2yyo7jkk6yujszguz.py
# Topologically Sorted Source Nodes: [out_44], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_44 => cat_44
# Graph fragment:
#   %cat_44 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_57, %relu_58], 1), kwargs = {})
triton_poi_fused_cat_35 = async_compile.triton('triton_poi_fused_cat_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 313600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200
    x1 = (xindex // 200)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 100, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((100*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 200, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((100*x1) + ((-100) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-100) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-100) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-100) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-100) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wo/cwo4xpdnkqe5saxaenggy2tgw6oain2gyzsxfcsr5b2zsm3vabf6.py
# Topologically Sorted Source Nodes: [out_45, x_105], Original ATen: [aten.cat, aten.add]
# Source node to ATen node mapping:
#   out_45 => cat_45
#   x_105 => add_267
# Graph fragment:
#   %cat_45 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_264, %add_266], 1), kwargs = {})
#   %add_267 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_45, %add_258), kwargs = {})
triton_poi_fused_add_cat_36 = async_compile.triton('triton_poi_fused_add_cat_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 80
    x1 = (xindex // 80)
    x2 = xindex
    tmp28 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 40, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((40*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 80, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((40*x1) + ((-40) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-40) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-40) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-40) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-40) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp5, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6y/c6ysrp6p6dqwkbjb22j46owptmbcfb22ffllgz6nsxkigp3g44po.py
# Topologically Sorted Source Nodes: [input_264, input_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_264 => add_269, mul_364, mul_365, sub_118
#   input_265 => relu_59
# Graph fragment:
#   %sub_118 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_137, %unsqueeze_945), kwargs = {})
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_118, %unsqueeze_947), kwargs = {})
#   %mul_365 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_364, %unsqueeze_949), kwargs = {})
#   %add_269 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_365, %unsqueeze_951), kwargs = {})
#   %relu_59 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_269,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 92
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


# kernel path: /tmp/torchinductor_sahanp/36/c3646hoy7dlgdfgqopznwunstrbm32okcw3f5423ezk5jaua3yga.py
# Topologically Sorted Source Nodes: [out_46], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_46 => cat_46
# Graph fragment:
#   %cat_46 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_59, %relu_60], 1), kwargs = {})
triton_poi_fused_cat_38 = async_compile.triton('triton_poi_fused_cat_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 184
    x1 = (xindex // 184)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 92, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((92*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 184, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((92*x1) + ((-92) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-92) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-92) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-92) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-92) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hu/chu5rvkfyze6dw42fily64xw7sgixlpzlfv5bhwbcwjcj4iyujka.py
# Topologically Sorted Source Nodes: [input_284, input_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_284 => add_287, mul_388, mul_389, sub_126
#   input_285 => relu_63
# Graph fragment:
#   %sub_126 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_145, %unsqueeze_1009), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_126, %unsqueeze_1011), kwargs = {})
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_388, %unsqueeze_1013), kwargs = {})
#   %add_287 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_389, %unsqueeze_1015), kwargs = {})
#   %relu_63 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_287,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/rk/crknhxnpvrywglmumfs4l32qvezr3uehyfrok5dx47tamy4dtila.py
# Topologically Sorted Source Nodes: [out_50, x_se_36], Original ATen: [aten.cat, aten.mean]
# Source node to ATen node mapping:
#   out_50 => cat_50
#   x_se_36 => mean_10
# Graph fragment:
#   %cat_50 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_63, %relu_64], 1), kwargs = {})
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_50, [2, 3], True), kwargs = {})
triton_red_fused_cat_mean_40 = async_compile.triton('triton_red_fused_cat_mean_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_mean_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_mean_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    x3 = xindex
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 240, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((240*r2) + (47040*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 480, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((240*r2) + (47040*x1) + ((-240) + x0)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to((-240) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tl.load(in_ptr3 + (tl.broadcast_to((-240) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.sqrt(tmp14)
        tmp16 = tl.full([1, 1], 1, tl.int32)
        tmp17 = tmp16 / tmp15
        tmp18 = 1.0
        tmp19 = tmp17 * tmp18
        tmp20 = tmp11 * tmp19
        tmp21 = tl.load(in_ptr4 + (tl.broadcast_to((-240) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 * tmp21
        tmp23 = tl.load(in_ptr5 + (tl.broadcast_to((-240) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 + tmp23
        tmp25 = tl.full([1, 1], 0, tl.int32)
        tmp26 = triton_helpers.maximum(tmp25, tmp24)
        tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
        tmp28 = tl.where(tmp6, tmp26, tmp27)
        tmp29 = tl.where(tmp4, tmp5, tmp28)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
        tl.store(out_ptr0 + (r2 + (196*x3)), tmp29, rmask & xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tmp33 = 196.0
    tmp34 = tmp31 / tmp33
    tl.store(out_ptr2 + (x3), tmp34, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bd/cbd5oqsrr7eg73bwsa6c2cl6ac565223yiodyvxkmsxbfy32hhjm.py
# Topologically Sorted Source Nodes: [x_se_36, x_se_37, x_se_38], Original ATen: [aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_se_36 => mean_10
#   x_se_37 => convolution_147
#   x_se_38 => relu_65
# Graph fragment:
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_50, [2, 3], True), kwargs = {})
#   %convolution_147 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_10, %arg249_1, %arg250_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_65 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_147,), kwargs = {})
triton_poi_fused_convolution_mean_relu_41 = async_compile.triton('triton_poi_fused_convolution_mean_relu_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_41(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mw/cmww3of4baevitrybbb5owaazxeysvuqjhmu6t5h3gxmx6qwigsu.py
# Topologically Sorted Source Nodes: [x_se_36, x_se_37, x_se_38, x_se_39, hardsigmoid_9, x_113], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_9 => add_290, clamp_max_9, clamp_min_9, div_9
#   x_113 => mul_393
#   x_se_36 => mean_10
#   x_se_37 => convolution_147
#   x_se_38 => relu_65
#   x_se_39 => convolution_148
# Graph fragment:
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_50, [2, 3], True), kwargs = {})
#   %convolution_147 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_10, %arg249_1, %arg250_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_65 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_147,), kwargs = {})
#   %convolution_148 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_65, %arg251_1, %arg252_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_290 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_148, 3), kwargs = {})
#   %clamp_min_9 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_290, 0), kwargs = {})
#   %clamp_max_9 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_9, 6), kwargs = {})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_9, 6), kwargs = {})
#   %mul_393 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_50, %div_9), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_42 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_42(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 480
    y1 = (yindex // 480)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = 0.16666666666666666
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp12, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vx/cvxgkp3l4zo42qezoao5cnqsc4himjehbfweka6ncbdw3qvucgwg.py
# Topologically Sorted Source Nodes: [input_290], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_290 => add_292, mul_395, mul_396, sub_128
# Graph fragment:
#   %sub_128 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_149, %unsqueeze_1025), kwargs = {})
#   %mul_395 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_128, %unsqueeze_1027), kwargs = {})
#   %mul_396 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_395, %unsqueeze_1029), kwargs = {})
#   %add_292 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_396, %unsqueeze_1031), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/iv/civdyn6c5kh427p3yjbo2x2ro4233mncxtefiajxoz23amhecnya.py
# Topologically Sorted Source Nodes: [input_294], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_294 => add_296, mul_401, mul_402, sub_130
# Graph fragment:
#   %sub_130 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_151, %unsqueeze_1041), kwargs = {})
#   %mul_401 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_130, %unsqueeze_1043), kwargs = {})
#   %mul_402 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_401, %unsqueeze_1045), kwargs = {})
#   %add_296 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_402, %unsqueeze_1047), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/u6/cu6di3k655f6mbsyxyvpemqxzmtcccltkczwgx2vb5krpx22toec.py
# Topologically Sorted Source Nodes: [out_51, input_296, x_115], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_296 => add_298, mul_404, mul_405, sub_131
#   out_51 => cat_51
#   x_115 => add_299
# Graph fragment:
#   %cat_51 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_292, %add_294], 1), kwargs = {})
#   %sub_131 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_152, %unsqueeze_1049), kwargs = {})
#   %mul_404 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_131, %unsqueeze_1051), kwargs = {})
#   %mul_405 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_404, %unsqueeze_1053), kwargs = {})
#   %add_298 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_405, %unsqueeze_1055), kwargs = {})
#   %add_299 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_51, %add_298), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_45', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 112
    x1 = (xindex // 112)
    x2 = xindex
    tmp28 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp29 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((56*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 112, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((56*x1) + ((-56) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-56) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-56) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-56) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-56) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp5, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 + tmp13
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tmp16 / tmp33
    tmp35 = tmp34 * tmp18
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = tmp27 + tmp40
    tl.store(in_out_ptr0 + (x2), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pf/cpf5tu77cyn3y4dmuckf4iz7ivjb7vtvywjksmfv7szrfeggv5e4.py
# Topologically Sorted Source Nodes: [input_298, input_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_298 => add_301, mul_407, mul_408, sub_132
#   input_299 => relu_66
# Graph fragment:
#   %sub_132 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_153, %unsqueeze_1057), kwargs = {})
#   %mul_407 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_132, %unsqueeze_1059), kwargs = {})
#   %mul_408 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_407, %unsqueeze_1061), kwargs = {})
#   %add_301 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_408, %unsqueeze_1063), kwargs = {})
#   %relu_66 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_301,), kwargs = {})
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
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 336
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


# kernel path: /tmp/torchinductor_sahanp/5j/c5jw3dwal4u7qhrajytnbyb5pd57zubzrmkvonbbydorkt3sci6h.py
# Topologically Sorted Source Nodes: [out_52, x_se_40], Original ATen: [aten.cat, aten.mean]
# Source node to ATen node mapping:
#   out_52 => cat_52
#   x_se_40 => mean_11
# Graph fragment:
#   %cat_52 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_66, %relu_67], 1), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_52, [2, 3], True), kwargs = {})
triton_red_fused_cat_mean_47 = async_compile.triton('triton_red_fused_cat_mean_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_mean_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_mean_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    x3 = xindex
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 336, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((336*r2) + (65856*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 672, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((336*r2) + (65856*x1) + ((-336) + x0)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to((-336) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tl.load(in_ptr3 + (tl.broadcast_to((-336) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.sqrt(tmp14)
        tmp16 = tl.full([1, 1], 1, tl.int32)
        tmp17 = tmp16 / tmp15
        tmp18 = 1.0
        tmp19 = tmp17 * tmp18
        tmp20 = tmp11 * tmp19
        tmp21 = tl.load(in_ptr4 + (tl.broadcast_to((-336) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 * tmp21
        tmp23 = tl.load(in_ptr5 + (tl.broadcast_to((-336) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 + tmp23
        tmp25 = tl.full([1, 1], 0, tl.int32)
        tmp26 = triton_helpers.maximum(tmp25, tmp24)
        tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
        tmp28 = tl.where(tmp6, tmp26, tmp27)
        tmp29 = tl.where(tmp4, tmp5, tmp28)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
        tl.store(out_ptr0 + (r2 + (196*x3)), tmp29, rmask & xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tmp33 = 196.0
    tmp34 = tmp31 / tmp33
    tl.store(out_ptr2 + (x3), tmp34, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ho/chokdpzzvjzbl4ogeiaj57sce27utk3ilhqkdqv3fowmlykwmgp6.py
# Topologically Sorted Source Nodes: [x_se_40, x_se_41, x_se_42], Original ATen: [aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_se_40 => mean_11
#   x_se_41 => convolution_155
#   x_se_42 => relu_68
# Graph fragment:
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_52, [2, 3], True), kwargs = {})
#   %convolution_155 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_11, %arg283_1, %arg284_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_68 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_155,), kwargs = {})
triton_poi_fused_convolution_mean_relu_48 = async_compile.triton('triton_poi_fused_convolution_mean_relu_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_48(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 168
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/67/c67uo7hn7ntmpc2uf3zvv4t6d3vv2nzxwiyntojuz7rolxtz37qk.py
# Topologically Sorted Source Nodes: [x_se_40, x_se_41, x_se_42, x_se_43, hardsigmoid_10, x_117], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_10 => add_304, clamp_max_10, clamp_min_10, div_10
#   x_117 => mul_412
#   x_se_40 => mean_11
#   x_se_41 => convolution_155
#   x_se_42 => relu_68
#   x_se_43 => convolution_156
# Graph fragment:
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_52, [2, 3], True), kwargs = {})
#   %convolution_155 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_11, %arg283_1, %arg284_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_68 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_155,), kwargs = {})
#   %convolution_156 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_68, %arg285_1, %arg286_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_304 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_156, 3), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_304, 0), kwargs = {})
#   %clamp_max_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 6), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_10, 6), kwargs = {})
#   %mul_412 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_52, %div_10), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_49 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_49(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = 0.16666666666666666
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (y0 + (672*x2) + (131712*y1)), tmp12, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nn/cnn6v5kr65xb2y5h7b3bpmyrxyl6ocu3ryzc6qs5dj5vi7j4hbnz.py
# Topologically Sorted Source Nodes: [out_53, x_119], Original ATen: [aten.cat, aten.add]
# Source node to ATen node mapping:
#   out_53 => cat_53
#   x_119 => add_309
# Graph fragment:
#   %cat_53 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_306, %add_308], 1), kwargs = {})
#   %add_309 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_53, %add_299), kwargs = {})
triton_poi_fused_add_cat_50 = async_compile.triton('triton_poi_fused_add_cat_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 112
    x1 = (xindex // 112)
    x2 = xindex
    tmp28 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((56*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 112, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((56*x1) + ((-56) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-56) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-56) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-56) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-56) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp5, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pd/cpd7hf3v4biqcae5a3wnfy2sbzwushrszb2ofudpx3memwsg2gau.py
# Topologically Sorted Source Nodes: [out_54], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_54 => cat_54
# Graph fragment:
#   %cat_54 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_69, %relu_70], 1), kwargs = {})
triton_poi_fused_cat_51 = async_compile.triton('triton_poi_fused_cat_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 672
    x1 = (xindex // 672)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 336, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((336*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 672, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((336*x1) + ((-336) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-336) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-336) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-336) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-336) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/f2/cf2c3vjud5firllxczqoyibzfn7mdvcrbz6n4qvrjebu3i6pe7yh.py
# Topologically Sorted Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_122 => add_315, mul_426, mul_427, sub_138
# Graph fragment:
#   %sub_138 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_161, %unsqueeze_1105), kwargs = {})
#   %mul_426 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_138, %unsqueeze_1107), kwargs = {})
#   %mul_427 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_426, %unsqueeze_1109), kwargs = {})
#   %add_315 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_427, %unsqueeze_1111), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_52(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 263424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
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


# kernel path: /tmp/torchinductor_sahanp/dr/cdrsgamxcqejowbrgduhhb6yhmy352qux3mozwh6xnffh6rzycb7.py
# Topologically Sorted Source Nodes: [x_se_44], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_44 => mean_12
# Graph fragment:
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_315, [2, 3], True), kwargs = {})
triton_per_fused_mean_53 = async_compile.triton('triton_per_fused_mean_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_53(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (32928*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ik/cikwtixroqad3gvmbs4jd7dx5rxlinzjwa3admhffswicaaa7o6y.py
# Topologically Sorted Source Nodes: [x_se_44, x_se_45, x_se_46, x_se_47, hardsigmoid_11, x_123], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_11 => add_316, clamp_max_11, clamp_min_11, div_11
#   x_123 => mul_428
#   x_se_44 => mean_12
#   x_se_45 => convolution_162
#   x_se_46 => relu_71
#   x_se_47 => convolution_163
# Graph fragment:
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_315, [2, 3], True), kwargs = {})
#   %convolution_162 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_12, %arg312_1, %arg313_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_71 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_162,), kwargs = {})
#   %convolution_163 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_71, %arg314_1, %arg315_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_316 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_163, 3), kwargs = {})
#   %clamp_min_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_316, 0), kwargs = {})
#   %clamp_max_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_11, 6), kwargs = {})
#   %div_11 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_11, 6), kwargs = {})
#   %mul_428 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_315, %div_11), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_54 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_54', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_54(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 263424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 32928)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = 0.16666666666666666
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uu/cuuq3vrd4plovedwmjivwwb3vjw37xi5iismhdjisobffitwmsim.py
# Topologically Sorted Source Nodes: [input_314], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_314 => add_318, mul_430, mul_431, sub_139
# Graph fragment:
#   %sub_139 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_164, %unsqueeze_1113), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_139, %unsqueeze_1115), kwargs = {})
#   %mul_431 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_430, %unsqueeze_1117), kwargs = {})
#   %add_318 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_431, %unsqueeze_1119), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_55(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31360
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


# kernel path: /tmp/torchinductor_sahanp/qo/cqof6hv5zbjqqm573dqvgtkkat4llkrha2e3atxunvacdleg3wa5.py
# Topologically Sorted Source Nodes: [input_318], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_318 => add_322, mul_436, mul_437, sub_141
# Graph fragment:
#   %sub_141 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_166, %unsqueeze_1129), kwargs = {})
#   %mul_436 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_141, %unsqueeze_1131), kwargs = {})
#   %mul_437 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_436, %unsqueeze_1133), kwargs = {})
#   %add_322 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_437, %unsqueeze_1135), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_56', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_56(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
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


# kernel path: /tmp/torchinductor_sahanp/iw/ciwst3yqg4x6pdmvvxjl3ixweceraa5qrzbli6flwxt4bz77s4y2.py
# Topologically Sorted Source Nodes: [out_55, input_320, x_125], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_320 => add_324, mul_439, mul_440, sub_142
#   out_55 => cat_55
#   x_125 => add_325
# Graph fragment:
#   %cat_55 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_318, %add_320], 1), kwargs = {})
#   %sub_142 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_167, %unsqueeze_1137), kwargs = {})
#   %mul_439 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_142, %unsqueeze_1139), kwargs = {})
#   %mul_440 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_439, %unsqueeze_1141), kwargs = {})
#   %add_324 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_440, %unsqueeze_1143), kwargs = {})
#   %add_325 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_55, %add_324), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_57', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_57', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_57(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 160
    x1 = (xindex // 160)
    x2 = xindex
    tmp28 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp29 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((80*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((80*x1) + ((-80) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-80) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-80) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-80) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-80) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp5, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 + tmp13
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tmp16 / tmp33
    tmp35 = tmp34 * tmp18
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = tmp27 + tmp40
    tl.store(in_out_ptr0 + (x2), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pz/cpz5nmbivku4hf3uxz4jafnjshfrkewifiw7arwcq7lujkx2ngmi.py
# Topologically Sorted Source Nodes: [input_322, input_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_322 => add_327, mul_442, mul_443, sub_143
#   input_323 => relu_72
# Graph fragment:
#   %sub_143 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_168, %unsqueeze_1145), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_143, %unsqueeze_1147), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_442, %unsqueeze_1149), kwargs = {})
#   %add_327 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_443, %unsqueeze_1151), kwargs = {})
#   %relu_72 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_327,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_58 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_58', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
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


# kernel path: /tmp/torchinductor_sahanp/hy/chyndlsb7otsfuddnq7irzuxi25l6e24elgba5244xvvi2kbpmhm.py
# Topologically Sorted Source Nodes: [out_56], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_56 => cat_56
# Graph fragment:
#   %cat_56 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_72, %relu_73], 1), kwargs = {})
triton_poi_fused_cat_59 = async_compile.triton('triton_poi_fused_cat_59', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_59(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 960
    x1 = (xindex // 960)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 480, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((480*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 960, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((480*x1) + ((-480) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-480) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-480) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-480) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-480) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/v3/cv334skbc3iuau6mntphpjvsgflewde2hixoja5j3m2pmqxz6ydw.py
# Topologically Sorted Source Nodes: [out_57, x_128], Original ATen: [aten.cat, aten.add]
# Source node to ATen node mapping:
#   out_57 => cat_57
#   x_128 => add_334
# Graph fragment:
#   %cat_57 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_331, %add_333], 1), kwargs = {})
#   %add_334 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_57, %add_325), kwargs = {})
triton_poi_fused_add_cat_60 = async_compile.triton('triton_poi_fused_add_cat_60', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_60', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_60(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 160
    x1 = (xindex // 160)
    x2 = xindex
    tmp28 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((80*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((80*x1) + ((-80) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-80) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-80) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-80) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-80) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp5, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qm/cqmrxjr4lkuxqelu5hwlkvpkamuyd5ueelx35vho56qowx2fejog.py
# Topologically Sorted Source Nodes: [out_58, x_se_48], Original ATen: [aten.cat, aten.mean]
# Source node to ATen node mapping:
#   out_58 => cat_58
#   x_se_48 => mean_13
# Graph fragment:
#   %cat_58 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_74, %relu_75], 1), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_58, [2, 3], True), kwargs = {})
triton_per_fused_cat_mean_61 = async_compile.triton('triton_per_fused_cat_mean_61', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mean_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_mean_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 960
    r2 = rindex
    x1 = (xindex // 960)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 480, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((480*r2) + (23520*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 960, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((480*r2) + (23520*x1) + ((-480) + x0)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to((-480) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + (tl.broadcast_to((-480) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1, 1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + (tl.broadcast_to((-480) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + (tl.broadcast_to((-480) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1, 1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None]
    tmp34 = 49.0
    tmp35 = tmp33 / tmp34
    tl.store(out_ptr0 + (r2 + (49*x3)), tmp29, rmask & xmask)
    tl.store(out_ptr2 + (x3), tmp35, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bl/cblaqcgzjkf7n3btwikvimf6vtjhutvyuqklyhowks3lxkgzzjne.py
# Topologically Sorted Source Nodes: [x_se_48, x_se_49, x_se_50], Original ATen: [aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_se_48 => mean_13
#   x_se_49 => convolution_174
#   x_se_50 => relu_76
# Graph fragment:
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_58, [2, 3], True), kwargs = {})
#   %convolution_174 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_13, %arg366_1, %arg367_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_76 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_174,), kwargs = {})
triton_poi_fused_convolution_mean_relu_62 = async_compile.triton('triton_poi_fused_convolution_mean_relu_62', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_62(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tw/ctwtqp7ws5qczfuibmnagonkr2oleypq7kzqd3je2l5e6gfxnz3y.py
# Topologically Sorted Source Nodes: [x_se_48, x_se_49, x_se_50, x_se_51, hardsigmoid_12, x_130], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_12 => add_339, clamp_max_12, clamp_min_12, div_12
#   x_130 => mul_459
#   x_se_48 => mean_13
#   x_se_49 => convolution_174
#   x_se_50 => relu_76
#   x_se_51 => convolution_175
# Graph fragment:
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_58, [2, 3], True), kwargs = {})
#   %convolution_174 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_13, %arg366_1, %arg367_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_76 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_174,), kwargs = {})
#   %convolution_175 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_76, %arg368_1, %arg369_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_339 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_175, 3), kwargs = {})
#   %clamp_min_12 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_339, 0), kwargs = {})
#   %clamp_max_12 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_12, 6), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_12, 6), kwargs = {})
#   %mul_459 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_58, %div_12), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_63 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_63', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_63(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 960
    y1 = (yindex // 960)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = 0.16666666666666666
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (y0 + (960*x2) + (47040*y1)), tmp12, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ul/culkglw3w4s6qexhn3mysewcw6fhonek4klhfyn57ookown6uvyi.py
# Topologically Sorted Source Nodes: [x_141, x_142, x_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   x_141 => add_365, mul_492, mul_493, sub_159
#   x_142 => relu_82
#   x_143 => mean_15
# Graph fragment:
#   %sub_159 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_188, %unsqueeze_1273), kwargs = {})
#   %mul_492 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_159, %unsqueeze_1275), kwargs = {})
#   %mul_493 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_492, %unsqueeze_1277), kwargs = {})
#   %add_365 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_493, %unsqueeze_1279), kwargs = {})
#   %relu_82 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_365,), kwargs = {})
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_82, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_64 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_mean_relu_64', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_mean_relu_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 960
    x1 = (xindex // 960)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (47040*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/dm/cdmyzdlq2f3arepu7u4d22pkr7hcfaeih7f6bqwwf4ylmlgfdz2q.py
# Topologically Sorted Source Nodes: [x_141, x_142, x_143, x_144, x_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_141 => add_365, mul_492, mul_493, sub_159
#   x_142 => relu_82
#   x_143 => mean_15
#   x_144 => convolution_189
#   x_145 => relu_83
# Graph fragment:
#   %sub_159 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_188, %unsqueeze_1273), kwargs = {})
#   %mul_492 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_159, %unsqueeze_1275), kwargs = {})
#   %mul_493 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_492, %unsqueeze_1277), kwargs = {})
#   %add_365 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_493, %unsqueeze_1279), kwargs = {})
#   %relu_82 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_365,), kwargs = {})
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_82, [-1, -2], True), kwargs = {})
#   %convolution_189 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_15, %arg429_1, %arg430_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_83 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_189,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg7_1, (8, ), (1, ))
    assert_size_stride(arg8_1, (8, ), (1, ))
    assert_size_stride(arg9_1, (8, ), (1, ))
    assert_size_stride(arg10_1, (8, ), (1, ))
    assert_size_stride(arg11_1, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg12_1, (8, ), (1, ))
    assert_size_stride(arg13_1, (8, ), (1, ))
    assert_size_stride(arg14_1, (8, ), (1, ))
    assert_size_stride(arg15_1, (8, ), (1, ))
    assert_size_stride(arg16_1, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg17_1, (8, ), (1, ))
    assert_size_stride(arg18_1, (8, ), (1, ))
    assert_size_stride(arg19_1, (8, ), (1, ))
    assert_size_stride(arg20_1, (8, ), (1, ))
    assert_size_stride(arg21_1, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg22_1, (8, ), (1, ))
    assert_size_stride(arg23_1, (8, ), (1, ))
    assert_size_stride(arg24_1, (8, ), (1, ))
    assert_size_stride(arg25_1, (8, ), (1, ))
    assert_size_stride(arg26_1, (24, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg27_1, (24, ), (1, ))
    assert_size_stride(arg28_1, (24, ), (1, ))
    assert_size_stride(arg29_1, (24, ), (1, ))
    assert_size_stride(arg30_1, (24, ), (1, ))
    assert_size_stride(arg31_1, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg32_1, (24, ), (1, ))
    assert_size_stride(arg33_1, (24, ), (1, ))
    assert_size_stride(arg34_1, (24, ), (1, ))
    assert_size_stride(arg35_1, (24, ), (1, ))
    assert_size_stride(arg36_1, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg37_1, (48, ), (1, ))
    assert_size_stride(arg38_1, (48, ), (1, ))
    assert_size_stride(arg39_1, (48, ), (1, ))
    assert_size_stride(arg40_1, (48, ), (1, ))
    assert_size_stride(arg41_1, (12, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg42_1, (12, ), (1, ))
    assert_size_stride(arg43_1, (12, ), (1, ))
    assert_size_stride(arg44_1, (12, ), (1, ))
    assert_size_stride(arg45_1, (12, ), (1, ))
    assert_size_stride(arg46_1, (12, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg47_1, (12, ), (1, ))
    assert_size_stride(arg48_1, (12, ), (1, ))
    assert_size_stride(arg49_1, (12, ), (1, ))
    assert_size_stride(arg50_1, (12, ), (1, ))
    assert_size_stride(arg51_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg52_1, (16, ), (1, ))
    assert_size_stride(arg53_1, (16, ), (1, ))
    assert_size_stride(arg54_1, (16, ), (1, ))
    assert_size_stride(arg55_1, (16, ), (1, ))
    assert_size_stride(arg56_1, (24, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg57_1, (24, ), (1, ))
    assert_size_stride(arg58_1, (24, ), (1, ))
    assert_size_stride(arg59_1, (24, ), (1, ))
    assert_size_stride(arg60_1, (24, ), (1, ))
    assert_size_stride(arg61_1, (36, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg62_1, (36, ), (1, ))
    assert_size_stride(arg63_1, (36, ), (1, ))
    assert_size_stride(arg64_1, (36, ), (1, ))
    assert_size_stride(arg65_1, (36, ), (1, ))
    assert_size_stride(arg66_1, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg67_1, (36, ), (1, ))
    assert_size_stride(arg68_1, (36, ), (1, ))
    assert_size_stride(arg69_1, (36, ), (1, ))
    assert_size_stride(arg70_1, (36, ), (1, ))
    assert_size_stride(arg71_1, (12, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg72_1, (12, ), (1, ))
    assert_size_stride(arg73_1, (12, ), (1, ))
    assert_size_stride(arg74_1, (12, ), (1, ))
    assert_size_stride(arg75_1, (12, ), (1, ))
    assert_size_stride(arg76_1, (12, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg77_1, (12, ), (1, ))
    assert_size_stride(arg78_1, (12, ), (1, ))
    assert_size_stride(arg79_1, (12, ), (1, ))
    assert_size_stride(arg80_1, (12, ), (1, ))
    assert_size_stride(arg81_1, (36, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg82_1, (36, ), (1, ))
    assert_size_stride(arg83_1, (36, ), (1, ))
    assert_size_stride(arg84_1, (36, ), (1, ))
    assert_size_stride(arg85_1, (36, ), (1, ))
    assert_size_stride(arg86_1, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg87_1, (36, ), (1, ))
    assert_size_stride(arg88_1, (36, ), (1, ))
    assert_size_stride(arg89_1, (36, ), (1, ))
    assert_size_stride(arg90_1, (36, ), (1, ))
    assert_size_stride(arg91_1, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg92_1, (72, ), (1, ))
    assert_size_stride(arg93_1, (72, ), (1, ))
    assert_size_stride(arg94_1, (72, ), (1, ))
    assert_size_stride(arg95_1, (72, ), (1, ))
    assert_size_stride(arg96_1, (20, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg97_1, (20, ), (1, ))
    assert_size_stride(arg98_1, (72, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg99_1, (72, ), (1, ))
    assert_size_stride(arg100_1, (20, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg101_1, (20, ), (1, ))
    assert_size_stride(arg102_1, (20, ), (1, ))
    assert_size_stride(arg103_1, (20, ), (1, ))
    assert_size_stride(arg104_1, (20, ), (1, ))
    assert_size_stride(arg105_1, (20, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg106_1, (20, ), (1, ))
    assert_size_stride(arg107_1, (20, ), (1, ))
    assert_size_stride(arg108_1, (20, ), (1, ))
    assert_size_stride(arg109_1, (20, ), (1, ))
    assert_size_stride(arg110_1, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg111_1, (24, ), (1, ))
    assert_size_stride(arg112_1, (24, ), (1, ))
    assert_size_stride(arg113_1, (24, ), (1, ))
    assert_size_stride(arg114_1, (24, ), (1, ))
    assert_size_stride(arg115_1, (40, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg116_1, (40, ), (1, ))
    assert_size_stride(arg117_1, (40, ), (1, ))
    assert_size_stride(arg118_1, (40, ), (1, ))
    assert_size_stride(arg119_1, (40, ), (1, ))
    assert_size_stride(arg120_1, (60, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg121_1, (60, ), (1, ))
    assert_size_stride(arg122_1, (60, ), (1, ))
    assert_size_stride(arg123_1, (60, ), (1, ))
    assert_size_stride(arg124_1, (60, ), (1, ))
    assert_size_stride(arg125_1, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg126_1, (60, ), (1, ))
    assert_size_stride(arg127_1, (60, ), (1, ))
    assert_size_stride(arg128_1, (60, ), (1, ))
    assert_size_stride(arg129_1, (60, ), (1, ))
    assert_size_stride(arg130_1, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg131_1, (32, ), (1, ))
    assert_size_stride(arg132_1, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg133_1, (120, ), (1, ))
    assert_size_stride(arg134_1, (20, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg135_1, (20, ), (1, ))
    assert_size_stride(arg136_1, (20, ), (1, ))
    assert_size_stride(arg137_1, (20, ), (1, ))
    assert_size_stride(arg138_1, (20, ), (1, ))
    assert_size_stride(arg139_1, (20, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg140_1, (20, ), (1, ))
    assert_size_stride(arg141_1, (20, ), (1, ))
    assert_size_stride(arg142_1, (20, ), (1, ))
    assert_size_stride(arg143_1, (20, ), (1, ))
    assert_size_stride(arg144_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg145_1, (120, ), (1, ))
    assert_size_stride(arg146_1, (120, ), (1, ))
    assert_size_stride(arg147_1, (120, ), (1, ))
    assert_size_stride(arg148_1, (120, ), (1, ))
    assert_size_stride(arg149_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg150_1, (120, ), (1, ))
    assert_size_stride(arg151_1, (120, ), (1, ))
    assert_size_stride(arg152_1, (120, ), (1, ))
    assert_size_stride(arg153_1, (120, ), (1, ))
    assert_size_stride(arg154_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg155_1, (240, ), (1, ))
    assert_size_stride(arg156_1, (240, ), (1, ))
    assert_size_stride(arg157_1, (240, ), (1, ))
    assert_size_stride(arg158_1, (240, ), (1, ))
    assert_size_stride(arg159_1, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg160_1, (40, ), (1, ))
    assert_size_stride(arg161_1, (40, ), (1, ))
    assert_size_stride(arg162_1, (40, ), (1, ))
    assert_size_stride(arg163_1, (40, ), (1, ))
    assert_size_stride(arg164_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg165_1, (40, ), (1, ))
    assert_size_stride(arg166_1, (40, ), (1, ))
    assert_size_stride(arg167_1, (40, ), (1, ))
    assert_size_stride(arg168_1, (40, ), (1, ))
    assert_size_stride(arg169_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg170_1, (40, ), (1, ))
    assert_size_stride(arg171_1, (40, ), (1, ))
    assert_size_stride(arg172_1, (40, ), (1, ))
    assert_size_stride(arg173_1, (40, ), (1, ))
    assert_size_stride(arg174_1, (80, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg175_1, (80, ), (1, ))
    assert_size_stride(arg176_1, (80, ), (1, ))
    assert_size_stride(arg177_1, (80, ), (1, ))
    assert_size_stride(arg178_1, (80, ), (1, ))
    assert_size_stride(arg179_1, (100, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg180_1, (100, ), (1, ))
    assert_size_stride(arg181_1, (100, ), (1, ))
    assert_size_stride(arg182_1, (100, ), (1, ))
    assert_size_stride(arg183_1, (100, ), (1, ))
    assert_size_stride(arg184_1, (100, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg185_1, (100, ), (1, ))
    assert_size_stride(arg186_1, (100, ), (1, ))
    assert_size_stride(arg187_1, (100, ), (1, ))
    assert_size_stride(arg188_1, (100, ), (1, ))
    assert_size_stride(arg189_1, (40, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg190_1, (40, ), (1, ))
    assert_size_stride(arg191_1, (40, ), (1, ))
    assert_size_stride(arg192_1, (40, ), (1, ))
    assert_size_stride(arg193_1, (40, ), (1, ))
    assert_size_stride(arg194_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg195_1, (40, ), (1, ))
    assert_size_stride(arg196_1, (40, ), (1, ))
    assert_size_stride(arg197_1, (40, ), (1, ))
    assert_size_stride(arg198_1, (40, ), (1, ))
    assert_size_stride(arg199_1, (92, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg200_1, (92, ), (1, ))
    assert_size_stride(arg201_1, (92, ), (1, ))
    assert_size_stride(arg202_1, (92, ), (1, ))
    assert_size_stride(arg203_1, (92, ), (1, ))
    assert_size_stride(arg204_1, (92, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg205_1, (92, ), (1, ))
    assert_size_stride(arg206_1, (92, ), (1, ))
    assert_size_stride(arg207_1, (92, ), (1, ))
    assert_size_stride(arg208_1, (92, ), (1, ))
    assert_size_stride(arg209_1, (40, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg210_1, (40, ), (1, ))
    assert_size_stride(arg211_1, (40, ), (1, ))
    assert_size_stride(arg212_1, (40, ), (1, ))
    assert_size_stride(arg213_1, (40, ), (1, ))
    assert_size_stride(arg214_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg215_1, (40, ), (1, ))
    assert_size_stride(arg216_1, (40, ), (1, ))
    assert_size_stride(arg217_1, (40, ), (1, ))
    assert_size_stride(arg218_1, (40, ), (1, ))
    assert_size_stride(arg219_1, (92, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg220_1, (92, ), (1, ))
    assert_size_stride(arg221_1, (92, ), (1, ))
    assert_size_stride(arg222_1, (92, ), (1, ))
    assert_size_stride(arg223_1, (92, ), (1, ))
    assert_size_stride(arg224_1, (92, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg225_1, (92, ), (1, ))
    assert_size_stride(arg226_1, (92, ), (1, ))
    assert_size_stride(arg227_1, (92, ), (1, ))
    assert_size_stride(arg228_1, (92, ), (1, ))
    assert_size_stride(arg229_1, (40, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg230_1, (40, ), (1, ))
    assert_size_stride(arg231_1, (40, ), (1, ))
    assert_size_stride(arg232_1, (40, ), (1, ))
    assert_size_stride(arg233_1, (40, ), (1, ))
    assert_size_stride(arg234_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg235_1, (40, ), (1, ))
    assert_size_stride(arg236_1, (40, ), (1, ))
    assert_size_stride(arg237_1, (40, ), (1, ))
    assert_size_stride(arg238_1, (40, ), (1, ))
    assert_size_stride(arg239_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg240_1, (240, ), (1, ))
    assert_size_stride(arg241_1, (240, ), (1, ))
    assert_size_stride(arg242_1, (240, ), (1, ))
    assert_size_stride(arg243_1, (240, ), (1, ))
    assert_size_stride(arg244_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg245_1, (240, ), (1, ))
    assert_size_stride(arg246_1, (240, ), (1, ))
    assert_size_stride(arg247_1, (240, ), (1, ))
    assert_size_stride(arg248_1, (240, ), (1, ))
    assert_size_stride(arg249_1, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg250_1, (120, ), (1, ))
    assert_size_stride(arg251_1, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg252_1, (480, ), (1, ))
    assert_size_stride(arg253_1, (56, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg254_1, (56, ), (1, ))
    assert_size_stride(arg255_1, (56, ), (1, ))
    assert_size_stride(arg256_1, (56, ), (1, ))
    assert_size_stride(arg257_1, (56, ), (1, ))
    assert_size_stride(arg258_1, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg259_1, (56, ), (1, ))
    assert_size_stride(arg260_1, (56, ), (1, ))
    assert_size_stride(arg261_1, (56, ), (1, ))
    assert_size_stride(arg262_1, (56, ), (1, ))
    assert_size_stride(arg263_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg264_1, (80, ), (1, ))
    assert_size_stride(arg265_1, (80, ), (1, ))
    assert_size_stride(arg266_1, (80, ), (1, ))
    assert_size_stride(arg267_1, (80, ), (1, ))
    assert_size_stride(arg268_1, (112, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg269_1, (112, ), (1, ))
    assert_size_stride(arg270_1, (112, ), (1, ))
    assert_size_stride(arg271_1, (112, ), (1, ))
    assert_size_stride(arg272_1, (112, ), (1, ))
    assert_size_stride(arg273_1, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg274_1, (336, ), (1, ))
    assert_size_stride(arg275_1, (336, ), (1, ))
    assert_size_stride(arg276_1, (336, ), (1, ))
    assert_size_stride(arg277_1, (336, ), (1, ))
    assert_size_stride(arg278_1, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg279_1, (336, ), (1, ))
    assert_size_stride(arg280_1, (336, ), (1, ))
    assert_size_stride(arg281_1, (336, ), (1, ))
    assert_size_stride(arg282_1, (336, ), (1, ))
    assert_size_stride(arg283_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg284_1, (168, ), (1, ))
    assert_size_stride(arg285_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg286_1, (672, ), (1, ))
    assert_size_stride(arg287_1, (56, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg288_1, (56, ), (1, ))
    assert_size_stride(arg289_1, (56, ), (1, ))
    assert_size_stride(arg290_1, (56, ), (1, ))
    assert_size_stride(arg291_1, (56, ), (1, ))
    assert_size_stride(arg292_1, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg293_1, (56, ), (1, ))
    assert_size_stride(arg294_1, (56, ), (1, ))
    assert_size_stride(arg295_1, (56, ), (1, ))
    assert_size_stride(arg296_1, (56, ), (1, ))
    assert_size_stride(arg297_1, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg298_1, (336, ), (1, ))
    assert_size_stride(arg299_1, (336, ), (1, ))
    assert_size_stride(arg300_1, (336, ), (1, ))
    assert_size_stride(arg301_1, (336, ), (1, ))
    assert_size_stride(arg302_1, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg303_1, (336, ), (1, ))
    assert_size_stride(arg304_1, (336, ), (1, ))
    assert_size_stride(arg305_1, (336, ), (1, ))
    assert_size_stride(arg306_1, (336, ), (1, ))
    assert_size_stride(arg307_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg308_1, (672, ), (1, ))
    assert_size_stride(arg309_1, (672, ), (1, ))
    assert_size_stride(arg310_1, (672, ), (1, ))
    assert_size_stride(arg311_1, (672, ), (1, ))
    assert_size_stride(arg312_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg313_1, (168, ), (1, ))
    assert_size_stride(arg314_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg315_1, (672, ), (1, ))
    assert_size_stride(arg316_1, (80, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg317_1, (80, ), (1, ))
    assert_size_stride(arg318_1, (80, ), (1, ))
    assert_size_stride(arg319_1, (80, ), (1, ))
    assert_size_stride(arg320_1, (80, ), (1, ))
    assert_size_stride(arg321_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg322_1, (80, ), (1, ))
    assert_size_stride(arg323_1, (80, ), (1, ))
    assert_size_stride(arg324_1, (80, ), (1, ))
    assert_size_stride(arg325_1, (80, ), (1, ))
    assert_size_stride(arg326_1, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg327_1, (112, ), (1, ))
    assert_size_stride(arg328_1, (112, ), (1, ))
    assert_size_stride(arg329_1, (112, ), (1, ))
    assert_size_stride(arg330_1, (112, ), (1, ))
    assert_size_stride(arg331_1, (160, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg332_1, (160, ), (1, ))
    assert_size_stride(arg333_1, (160, ), (1, ))
    assert_size_stride(arg334_1, (160, ), (1, ))
    assert_size_stride(arg335_1, (160, ), (1, ))
    assert_size_stride(arg336_1, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg337_1, (480, ), (1, ))
    assert_size_stride(arg338_1, (480, ), (1, ))
    assert_size_stride(arg339_1, (480, ), (1, ))
    assert_size_stride(arg340_1, (480, ), (1, ))
    assert_size_stride(arg341_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg342_1, (480, ), (1, ))
    assert_size_stride(arg343_1, (480, ), (1, ))
    assert_size_stride(arg344_1, (480, ), (1, ))
    assert_size_stride(arg345_1, (480, ), (1, ))
    assert_size_stride(arg346_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg347_1, (80, ), (1, ))
    assert_size_stride(arg348_1, (80, ), (1, ))
    assert_size_stride(arg349_1, (80, ), (1, ))
    assert_size_stride(arg350_1, (80, ), (1, ))
    assert_size_stride(arg351_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg352_1, (80, ), (1, ))
    assert_size_stride(arg353_1, (80, ), (1, ))
    assert_size_stride(arg354_1, (80, ), (1, ))
    assert_size_stride(arg355_1, (80, ), (1, ))
    assert_size_stride(arg356_1, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg357_1, (480, ), (1, ))
    assert_size_stride(arg358_1, (480, ), (1, ))
    assert_size_stride(arg359_1, (480, ), (1, ))
    assert_size_stride(arg360_1, (480, ), (1, ))
    assert_size_stride(arg361_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg362_1, (480, ), (1, ))
    assert_size_stride(arg363_1, (480, ), (1, ))
    assert_size_stride(arg364_1, (480, ), (1, ))
    assert_size_stride(arg365_1, (480, ), (1, ))
    assert_size_stride(arg366_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg367_1, (240, ), (1, ))
    assert_size_stride(arg368_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg369_1, (960, ), (1, ))
    assert_size_stride(arg370_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg371_1, (80, ), (1, ))
    assert_size_stride(arg372_1, (80, ), (1, ))
    assert_size_stride(arg373_1, (80, ), (1, ))
    assert_size_stride(arg374_1, (80, ), (1, ))
    assert_size_stride(arg375_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg376_1, (80, ), (1, ))
    assert_size_stride(arg377_1, (80, ), (1, ))
    assert_size_stride(arg378_1, (80, ), (1, ))
    assert_size_stride(arg379_1, (80, ), (1, ))
    assert_size_stride(arg380_1, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg381_1, (480, ), (1, ))
    assert_size_stride(arg382_1, (480, ), (1, ))
    assert_size_stride(arg383_1, (480, ), (1, ))
    assert_size_stride(arg384_1, (480, ), (1, ))
    assert_size_stride(arg385_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg386_1, (480, ), (1, ))
    assert_size_stride(arg387_1, (480, ), (1, ))
    assert_size_stride(arg388_1, (480, ), (1, ))
    assert_size_stride(arg389_1, (480, ), (1, ))
    assert_size_stride(arg390_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg391_1, (80, ), (1, ))
    assert_size_stride(arg392_1, (80, ), (1, ))
    assert_size_stride(arg393_1, (80, ), (1, ))
    assert_size_stride(arg394_1, (80, ), (1, ))
    assert_size_stride(arg395_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg396_1, (80, ), (1, ))
    assert_size_stride(arg397_1, (80, ), (1, ))
    assert_size_stride(arg398_1, (80, ), (1, ))
    assert_size_stride(arg399_1, (80, ), (1, ))
    assert_size_stride(arg400_1, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg401_1, (480, ), (1, ))
    assert_size_stride(arg402_1, (480, ), (1, ))
    assert_size_stride(arg403_1, (480, ), (1, ))
    assert_size_stride(arg404_1, (480, ), (1, ))
    assert_size_stride(arg405_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg406_1, (480, ), (1, ))
    assert_size_stride(arg407_1, (480, ), (1, ))
    assert_size_stride(arg408_1, (480, ), (1, ))
    assert_size_stride(arg409_1, (480, ), (1, ))
    assert_size_stride(arg410_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg411_1, (240, ), (1, ))
    assert_size_stride(arg412_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg413_1, (960, ), (1, ))
    assert_size_stride(arg414_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg415_1, (80, ), (1, ))
    assert_size_stride(arg416_1, (80, ), (1, ))
    assert_size_stride(arg417_1, (80, ), (1, ))
    assert_size_stride(arg418_1, (80, ), (1, ))
    assert_size_stride(arg419_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg420_1, (80, ), (1, ))
    assert_size_stride(arg421_1, (80, ), (1, ))
    assert_size_stride(arg422_1, (80, ), (1, ))
    assert_size_stride(arg423_1, (80, ), (1, ))
    assert_size_stride(arg424_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg425_1, (960, ), (1, ))
    assert_size_stride(arg426_1, (960, ), (1, ))
    assert_size_stride(arg427_1, (960, ), (1, ))
    assert_size_stride(arg428_1, (960, ), (1, ))
    assert_size_stride(arg429_1, (1280, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg430_1, (1280, ), (1, ))
    assert_size_stride(arg431_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg432_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((16, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 48, 9, grid=grid(48, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_75, x_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        # Topologically Sorted Source Nodes: [input_181], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 8, 112, 112), (100352, 1, 896, 8))
        del arg6_1
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_182, input_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf5, arg7_1, arg8_1, arg9_1, arg10_1, 802816, grid=grid(802816), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [input_184], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg11_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf6, (8, 8, 112, 112), (100352, 1, 896, 8))
        del arg11_1
        buf7 = empty_strided_cuda((8, 16, 112, 112), (200704, 1, 1792, 16), torch.float32)
        # Topologically Sorted Source Nodes: [out_32], Original ATen: [aten.cat]
        triton_poi_fused_cat_4.run(buf5, buf6, arg12_1, arg13_1, arg14_1, arg15_1, buf7, 1605632, grid=grid(1605632), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del buf5
        del buf6
        # Topologically Sorted Source Nodes: [out_32, input_187], Original ATen: [aten.cat, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 8, 112, 112), (100352, 1, 896, 8))
        del arg16_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [input_188], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_5.run(buf9, arg17_1, arg18_1, arg19_1, arg20_1, 802816, grid=grid(802816), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        # Topologically Sorted Source Nodes: [input_189], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg21_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf10, (8, 8, 112, 112), (100352, 1, 896, 8))
        del arg21_1
        buf11 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [out_33, x_79], Original ATen: [aten.cat, aten.add]
        triton_poi_fused_add_cat_6.run(buf11, buf9, buf10, arg22_1, arg23_1, arg24_1, arg25_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del buf10
        del buf9
        # Topologically Sorted Source Nodes: [input_191], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 24, 112, 112), (301056, 1, 2688, 24))
        del arg26_1
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_192, input_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf13, arg27_1, arg28_1, arg29_1, arg30_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        # Topologically Sorted Source Nodes: [input_194], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg31_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf14, (8, 24, 112, 112), (301056, 1, 2688, 24))
        del arg31_1
        buf15 = empty_strided_cuda((8, 48, 112, 112), (602112, 1, 5376, 48), torch.float32)
        # Topologically Sorted Source Nodes: [out_34], Original ATen: [aten.cat]
        triton_poi_fused_cat_8.run(buf13, buf14, arg32_1, arg33_1, arg34_1, arg35_1, buf15, 4816896, grid=grid(4816896), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        del buf13
        del buf14
        # Topologically Sorted Source Nodes: [out_34, x_81], Original ATen: [aten.cat, aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg36_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf16, (8, 48, 56, 56), (150528, 1, 2688, 48))
        del arg36_1
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_9.run(buf17, arg37_1, arg38_1, arg39_1, arg40_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        # Topologically Sorted Source Nodes: [x_82, input_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 12, 56, 56), (37632, 1, 672, 12))
        del arg41_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_198], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf19, arg42_1, arg43_1, arg44_1, arg45_1, 301056, grid=grid(301056), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        # Topologically Sorted Source Nodes: [input_199], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg46_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=12, bias=None)
        assert_size_stride(buf20, (8, 12, 56, 56), (37632, 1, 672, 12))
        del arg46_1
        # Topologically Sorted Source Nodes: [input_201], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf11, arg51_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf21, (8, 16, 56, 56), (50176, 1, 896, 16))
        del arg51_1
        del buf11
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [input_202], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf22, arg52_1, arg53_1, arg54_1, arg55_1, 401408, grid=grid(401408), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        # Topologically Sorted Source Nodes: [input_202, input_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg56_1
        del buf22
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [out_35, input_204, x_84], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_12.run(buf24, buf19, buf20, arg47_1, arg48_1, arg49_1, arg50_1, arg57_1, arg58_1, arg59_1, arg60_1, 602112, grid=grid(602112), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf19
        del buf20
        # Topologically Sorted Source Nodes: [input_205], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (8, 36, 56, 56), (112896, 1, 2016, 36))
        del arg61_1
        buf26 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [input_206, input_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf26, arg62_1, arg63_1, arg64_1, arg65_1, 903168, grid=grid(903168), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        # Topologically Sorted Source Nodes: [input_208], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, arg66_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=36, bias=None)
        assert_size_stride(buf27, (8, 36, 56, 56), (112896, 1, 2016, 36))
        del arg66_1
        buf28 = empty_strided_cuda((8, 72, 56, 56), (225792, 1, 4032, 72), torch.float32)
        # Topologically Sorted Source Nodes: [out_36], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf26, buf27, arg67_1, arg68_1, arg69_1, arg70_1, buf28, 1806336, grid=grid(1806336), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        del buf26
        del buf27
        # Topologically Sorted Source Nodes: [out_36, input_211], Original ATen: [aten.cat, aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 12, 56, 56), (37632, 1, 672, 12))
        del arg71_1
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [input_212], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf30, arg72_1, arg73_1, arg74_1, arg75_1, 301056, grid=grid(301056), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        # Topologically Sorted Source Nodes: [input_213], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg76_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=12, bias=None)
        assert_size_stride(buf31, (8, 12, 56, 56), (37632, 1, 672, 12))
        del arg76_1
        buf32 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [out_37, x_87], Original ATen: [aten.cat, aten.add]
        triton_poi_fused_add_cat_15.run(buf32, buf30, buf31, arg77_1, arg78_1, arg79_1, arg80_1, 602112, grid=grid(602112), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        del buf30
        del buf31
        # Topologically Sorted Source Nodes: [input_215], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 36, 56, 56), (112896, 1, 2016, 36))
        del arg81_1
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [input_216, input_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf34, arg82_1, arg83_1, arg84_1, arg85_1, 903168, grid=grid(903168), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        # Topologically Sorted Source Nodes: [input_218], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg86_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=36, bias=None)
        assert_size_stride(buf35, (8, 36, 56, 56), (112896, 1, 2016, 36))
        del arg86_1
        buf36 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [out_38], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf34, buf35, arg87_1, arg88_1, arg89_1, arg90_1, buf36, 1806336, grid=grid(1806336), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        del buf34
        del buf35
        # Topologically Sorted Source Nodes: [out_38, x_89], Original ATen: [aten.cat, aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg91_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf37, (8, 72, 28, 28), (56448, 1, 2016, 72))
        del arg91_1
        del buf36
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf38, arg92_1, arg93_1, arg94_1, arg95_1, 451584, grid=grid(451584), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        buf39 = empty_strided_cuda((8, 72, 1, 1, 7), (504, 1, 4032, 4032, 72), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_28], Original ATen: [aten.mean]
        triton_red_fused_mean_17.run(buf38, buf39, 4032, 112, grid=grid(4032), stream=stream0)
        buf41 = empty_strided_cuda((8, 72, 1, 1), (72, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_28], Original ATen: [aten.mean]
        triton_per_fused_mean_18.run(buf39, buf41, 576, 7, grid=grid(576), stream=stream0)
        del buf39
        # Topologically Sorted Source Nodes: [x_se_28, x_se_29], Original ATen: [aten.mean, aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg96_1
        del buf41
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_se_28, x_se_29, x_se_30], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_19.run(buf43, arg97_1, 160, grid=grid(160), stream=stream0)
        del arg97_1
        # Topologically Sorted Source Nodes: [x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf44 = extern_kernels.convolution(buf43, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 72, 1, 1), (72, 1, 1, 1))
        del arg98_1
        del buf43
        buf45 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_se_28, x_se_29, x_se_30, x_se_31, hardsigmoid_7, x_91], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_20.run(buf45, buf44, arg99_1, 451584, grid=grid(451584), stream=stream0)
        del arg99_1
        del buf44
        # Topologically Sorted Source Nodes: [x_se_28, x_se_29, x_se_30, x_se_31, hardsigmoid_7, x_91, input_221], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf46 = extern_kernels.convolution(buf45, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 20, 28, 28), (15680, 1, 560, 20))
        del arg100_1
        del buf45
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [input_222], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf47, arg101_1, arg102_1, arg103_1, arg104_1, 125440, grid=grid(125440), stream=stream0)
        del arg101_1
        del arg102_1
        del arg103_1
        del arg104_1
        # Topologically Sorted Source Nodes: [input_223], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
        assert_size_stride(buf48, (8, 20, 28, 28), (15680, 1, 560, 20))
        del arg105_1
        # Topologically Sorted Source Nodes: [input_225], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf32, arg110_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf49, (8, 24, 28, 28), (18816, 1, 672, 24))
        del arg110_1
        del buf32
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [input_226], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf50, arg111_1, arg112_1, arg113_1, arg114_1, 150528, grid=grid(150528), stream=stream0)
        del arg111_1
        del arg112_1
        del arg113_1
        del arg114_1
        # Topologically Sorted Source Nodes: [input_226, input_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 40, 28, 28), (31360, 1, 1120, 40))
        del arg115_1
        del buf50
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [out_39, input_228, x_93], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_23.run(buf52, buf47, buf48, arg106_1, arg107_1, arg108_1, arg109_1, arg116_1, arg117_1, arg118_1, arg119_1, 250880, grid=grid(250880), stream=stream0)
        del arg106_1
        del arg107_1
        del arg108_1
        del arg109_1
        del arg116_1
        del arg117_1
        del arg118_1
        del arg119_1
        del buf47
        del buf48
        # Topologically Sorted Source Nodes: [input_229], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (8, 60, 28, 28), (47040, 1, 1680, 60))
        del arg120_1
        buf54 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [input_230, input_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf54, arg121_1, arg122_1, arg123_1, arg124_1, 376320, grid=grid(376320), stream=stream0)
        del arg121_1
        del arg122_1
        del arg123_1
        del arg124_1
        # Topologically Sorted Source Nodes: [input_232], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, arg125_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf55, (8, 60, 28, 28), (47040, 1, 1680, 60))
        del arg125_1
        buf56 = empty_strided_cuda((8, 120, 28, 28), (94080, 784, 28, 1), torch.float32)
        buf58 = empty_strided_cuda((8, 120, 1, 1), (120, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_40, x_se_32], Original ATen: [aten.cat, aten.mean]
        triton_red_fused_cat_mean_25.run(buf54, buf55, arg126_1, arg127_1, arg128_1, arg129_1, buf56, buf58, 960, 784, grid=grid(960), stream=stream0)
        del arg126_1
        del arg127_1
        del arg128_1
        del arg129_1
        del buf54
        del buf55
        # Topologically Sorted Source Nodes: [x_se_32, x_se_33], Original ATen: [aten.mean, aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg130_1
        del buf58
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_se_32, x_se_33, x_se_34], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_26.run(buf60, arg131_1, 256, grid=grid(256), stream=stream0)
        del arg131_1
        # Topologically Sorted Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf61 = extern_kernels.convolution(buf60, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg132_1
        del buf60
        buf62 = empty_strided_cuda((8, 120, 28, 28), (94080, 1, 3360, 120), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35, hardsigmoid_8, x_95], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_27.run(buf56, buf61, arg133_1, buf62, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg133_1
        del buf56
        del buf61
        # Topologically Sorted Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35, hardsigmoid_8, x_95, input_235], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf63 = extern_kernels.convolution(buf62, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 20, 28, 28), (15680, 1, 560, 20))
        del arg134_1
        del buf62
        buf64 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [input_236], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf64, arg135_1, arg136_1, arg137_1, arg138_1, 125440, grid=grid(125440), stream=stream0)
        del arg135_1
        del arg136_1
        del arg137_1
        del arg138_1
        # Topologically Sorted Source Nodes: [input_237], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, arg139_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
        assert_size_stride(buf65, (8, 20, 28, 28), (15680, 1, 560, 20))
        del arg139_1
        buf66 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [out_41, x_97], Original ATen: [aten.cat, aten.add]
        triton_poi_fused_add_cat_28.run(buf66, buf64, buf65, arg140_1, arg141_1, arg142_1, arg143_1, 250880, grid=grid(250880), stream=stream0)
        del arg140_1
        del arg141_1
        del arg142_1
        del arg143_1
        del buf64
        del buf65
        # Topologically Sorted Source Nodes: [input_239], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 120, 28, 28), (94080, 1, 3360, 120))
        del arg144_1
        buf68 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [input_240, input_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf68, arg145_1, arg146_1, arg147_1, arg148_1, 752640, grid=grid(752640), stream=stream0)
        del arg145_1
        del arg146_1
        del arg147_1
        del arg148_1
        # Topologically Sorted Source Nodes: [input_242], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, arg149_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf69, (8, 120, 28, 28), (94080, 1, 3360, 120))
        del arg149_1
        buf70 = empty_strided_cuda((8, 240, 28, 28), (188160, 1, 6720, 240), torch.float32)
        # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf68, buf69, arg150_1, arg151_1, arg152_1, arg153_1, buf70, 1505280, grid=grid(1505280), stream=stream0)
        del arg150_1
        del arg151_1
        del arg152_1
        del arg153_1
        # Topologically Sorted Source Nodes: [out_42, x_99], Original ATen: [aten.cat, aten.convolution]
        buf71 = extern_kernels.convolution(buf70, arg154_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf71, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg154_1
        del buf70
        buf72 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf72, arg155_1, arg156_1, arg157_1, arg158_1, 376320, grid=grid(376320), stream=stream0)
        del arg155_1
        del arg156_1
        del arg157_1
        del arg158_1
        # Topologically Sorted Source Nodes: [x_100, input_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 40, 14, 14), (7840, 1, 560, 40))
        del arg159_1
        del buf72
        buf74 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [input_246], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf74, arg160_1, arg161_1, arg162_1, arg163_1, 62720, grid=grid(62720), stream=stream0)
        del arg160_1
        del arg161_1
        del arg162_1
        del arg163_1
        # Topologically Sorted Source Nodes: [input_247], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg164_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf75, (8, 40, 14, 14), (7840, 1, 560, 40))
        del arg164_1
        # Topologically Sorted Source Nodes: [input_249], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf66, arg169_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf76, (8, 40, 14, 14), (7840, 1, 560, 40))
        del arg169_1
        del buf66
        buf77 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [input_250], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf77, arg170_1, arg171_1, arg172_1, arg173_1, 62720, grid=grid(62720), stream=stream0)
        del arg170_1
        del arg171_1
        del arg172_1
        del arg173_1
        # Topologically Sorted Source Nodes: [input_250, input_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg174_1
        del buf77
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [out_43, input_252, x_102], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_33.run(buf79, buf74, buf75, arg165_1, arg166_1, arg167_1, arg168_1, arg175_1, arg176_1, arg177_1, arg178_1, 125440, grid=grid(125440), stream=stream0)
        del arg165_1
        del arg166_1
        del arg167_1
        del arg168_1
        del arg175_1
        del arg176_1
        del arg177_1
        del arg178_1
        del buf74
        del buf75
        # Topologically Sorted Source Nodes: [input_253], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 100, 14, 14), (19600, 1, 1400, 100))
        del arg179_1
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [input_254, input_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf81, arg180_1, arg181_1, arg182_1, arg183_1, 156800, grid=grid(156800), stream=stream0)
        del arg180_1
        del arg181_1
        del arg182_1
        del arg183_1
        # Topologically Sorted Source Nodes: [input_256], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg184_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=100, bias=None)
        assert_size_stride(buf82, (8, 100, 14, 14), (19600, 1, 1400, 100))
        del arg184_1
        buf83 = empty_strided_cuda((8, 200, 14, 14), (39200, 1, 2800, 200), torch.float32)
        # Topologically Sorted Source Nodes: [out_44], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf81, buf82, arg185_1, arg186_1, arg187_1, arg188_1, buf83, 313600, grid=grid(313600), stream=stream0)
        del arg185_1
        del arg186_1
        del arg187_1
        del arg188_1
        del buf81
        del buf82
        # Topologically Sorted Source Nodes: [out_44, input_259], Original ATen: [aten.cat, aten.convolution]
        buf84 = extern_kernels.convolution(buf83, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 40, 14, 14), (7840, 1, 560, 40))
        del arg189_1
        del buf83
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [input_260], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf85, arg190_1, arg191_1, arg192_1, arg193_1, 62720, grid=grid(62720), stream=stream0)
        del arg190_1
        del arg191_1
        del arg192_1
        del arg193_1
        # Topologically Sorted Source Nodes: [input_261], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg194_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf86, (8, 40, 14, 14), (7840, 1, 560, 40))
        del arg194_1
        buf87 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [out_45, x_105], Original ATen: [aten.cat, aten.add]
        triton_poi_fused_add_cat_36.run(buf87, buf85, buf86, arg195_1, arg196_1, arg197_1, arg198_1, 125440, grid=grid(125440), stream=stream0)
        del arg195_1
        del arg196_1
        del arg197_1
        del arg198_1
        del buf85
        del buf86
        # Topologically Sorted Source Nodes: [input_263], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 92, 14, 14), (18032, 1, 1288, 92))
        del arg199_1
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [input_264, input_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf89, arg200_1, arg201_1, arg202_1, arg203_1, 144256, grid=grid(144256), stream=stream0)
        del arg200_1
        del arg201_1
        del arg202_1
        del arg203_1
        # Topologically Sorted Source Nodes: [input_266], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg204_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=92, bias=None)
        assert_size_stride(buf90, (8, 92, 14, 14), (18032, 1, 1288, 92))
        del arg204_1
        buf91 = empty_strided_cuda((8, 184, 14, 14), (36064, 1, 2576, 184), torch.float32)
        # Topologically Sorted Source Nodes: [out_46], Original ATen: [aten.cat]
        triton_poi_fused_cat_38.run(buf89, buf90, arg205_1, arg206_1, arg207_1, arg208_1, buf91, 288512, grid=grid(288512), stream=stream0)
        del arg205_1
        del arg206_1
        del arg207_1
        del arg208_1
        del buf89
        del buf90
        # Topologically Sorted Source Nodes: [out_46, input_269], Original ATen: [aten.cat, aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 40, 14, 14), (7840, 1, 560, 40))
        del arg209_1
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [input_270], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf93, arg210_1, arg211_1, arg212_1, arg213_1, 62720, grid=grid(62720), stream=stream0)
        del arg210_1
        del arg211_1
        del arg212_1
        del arg213_1
        # Topologically Sorted Source Nodes: [input_271], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg214_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf94, (8, 40, 14, 14), (7840, 1, 560, 40))
        del arg214_1
        buf95 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [out_47, x_108], Original ATen: [aten.cat, aten.add]
        triton_poi_fused_add_cat_36.run(buf95, buf93, buf94, arg215_1, arg216_1, arg217_1, arg218_1, 125440, grid=grid(125440), stream=stream0)
        del arg215_1
        del arg216_1
        del arg217_1
        del arg218_1
        del buf93
        del buf94
        # Topologically Sorted Source Nodes: [input_273], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 92, 14, 14), (18032, 1, 1288, 92))
        del arg219_1
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [input_274, input_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf97, arg220_1, arg221_1, arg222_1, arg223_1, 144256, grid=grid(144256), stream=stream0)
        del arg220_1
        del arg221_1
        del arg222_1
        del arg223_1
        # Topologically Sorted Source Nodes: [input_276], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, arg224_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=92, bias=None)
        assert_size_stride(buf98, (8, 92, 14, 14), (18032, 1, 1288, 92))
        del arg224_1
        buf99 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [out_48], Original ATen: [aten.cat]
        triton_poi_fused_cat_38.run(buf97, buf98, arg225_1, arg226_1, arg227_1, arg228_1, buf99, 288512, grid=grid(288512), stream=stream0)
        del arg225_1
        del arg226_1
        del arg227_1
        del arg228_1
        del buf97
        del buf98
        # Topologically Sorted Source Nodes: [out_48, input_279], Original ATen: [aten.cat, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg229_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 40, 14, 14), (7840, 1, 560, 40))
        del arg229_1
        del buf99
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [input_280], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf101, arg230_1, arg231_1, arg232_1, arg233_1, 62720, grid=grid(62720), stream=stream0)
        del arg230_1
        del arg231_1
        del arg232_1
        del arg233_1
        # Topologically Sorted Source Nodes: [input_281], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg234_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf102, (8, 40, 14, 14), (7840, 1, 560, 40))
        del arg234_1
        buf103 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [out_49, x_111], Original ATen: [aten.cat, aten.add]
        triton_poi_fused_add_cat_36.run(buf103, buf101, buf102, arg235_1, arg236_1, arg237_1, arg238_1, 125440, grid=grid(125440), stream=stream0)
        del arg235_1
        del arg236_1
        del arg237_1
        del arg238_1
        del buf101
        del buf102
        # Topologically Sorted Source Nodes: [input_283], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg239_1
        buf105 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [input_284, input_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf105, arg240_1, arg241_1, arg242_1, arg243_1, 376320, grid=grid(376320), stream=stream0)
        del arg240_1
        del arg241_1
        del arg242_1
        del arg243_1
        # Topologically Sorted Source Nodes: [input_286], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg244_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf106, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg244_1
        buf107 = reinterpret_tensor(buf69, (8, 480, 14, 14), (94080, 196, 14, 1), 0); del buf69  # reuse
        buf109 = empty_strided_cuda((8, 480, 1, 1), (480, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_50, x_se_36], Original ATen: [aten.cat, aten.mean]
        triton_red_fused_cat_mean_40.run(buf105, buf106, arg245_1, arg246_1, arg247_1, arg248_1, buf107, buf109, 3840, 196, grid=grid(3840), stream=stream0)
        del arg245_1
        del arg246_1
        del arg247_1
        del arg248_1
        # Topologically Sorted Source Nodes: [x_se_36, x_se_37], Original ATen: [aten.mean, aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg249_1
        del buf109
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_se_36, x_se_37, x_se_38], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_41.run(buf111, arg250_1, 960, grid=grid(960), stream=stream0)
        del arg250_1
        # Topologically Sorted Source Nodes: [x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf112 = extern_kernels.convolution(buf111, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg251_1
        del buf111
        buf113 = reinterpret_tensor(buf68, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_se_36, x_se_37, x_se_38, x_se_39, hardsigmoid_9, x_113], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_42.run(buf107, buf112, arg252_1, buf113, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg252_1
        del buf107
        del buf112
        # Topologically Sorted Source Nodes: [x_se_36, x_se_37, x_se_38, x_se_39, hardsigmoid_9, x_113, input_289], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf114 = extern_kernels.convolution(buf113, arg253_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 56, 14, 14), (10976, 1, 784, 56))
        del arg253_1
        del buf113
        buf115 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [input_290], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf115, arg254_1, arg255_1, arg256_1, arg257_1, 87808, grid=grid(87808), stream=stream0)
        del arg254_1
        del arg255_1
        del arg256_1
        del arg257_1
        # Topologically Sorted Source Nodes: [input_291], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, arg258_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
        assert_size_stride(buf116, (8, 56, 14, 14), (10976, 1, 784, 56))
        del arg258_1
        # Topologically Sorted Source Nodes: [input_293], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf103, arg263_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf117, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg263_1
        del buf103
        buf118 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [input_294], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_44.run(buf118, arg264_1, arg265_1, arg266_1, arg267_1, 125440, grid=grid(125440), stream=stream0)
        del arg264_1
        del arg265_1
        del arg266_1
        del arg267_1
        # Topologically Sorted Source Nodes: [input_294, input_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf119 = extern_kernels.convolution(buf118, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg268_1
        del buf118
        buf120 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [out_51, input_296, x_115], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_45.run(buf120, buf115, buf116, arg259_1, arg260_1, arg261_1, arg262_1, arg269_1, arg270_1, arg271_1, arg272_1, 175616, grid=grid(175616), stream=stream0)
        del arg259_1
        del arg260_1
        del arg261_1
        del arg262_1
        del arg269_1
        del arg270_1
        del arg271_1
        del arg272_1
        del buf115
        del buf116
        # Topologically Sorted Source Nodes: [input_297], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, arg273_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 336, 14, 14), (65856, 1, 4704, 336))
        del arg273_1
        buf122 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [input_298, input_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf122, arg274_1, arg275_1, arg276_1, arg277_1, 526848, grid=grid(526848), stream=stream0)
        del arg274_1
        del arg275_1
        del arg276_1
        del arg277_1
        # Topologically Sorted Source Nodes: [input_300], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, arg278_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
        assert_size_stride(buf123, (8, 336, 14, 14), (65856, 1, 4704, 336))
        del arg278_1
        buf124 = empty_strided_cuda((8, 672, 14, 14), (131712, 196, 14, 1), torch.float32)
        buf126 = empty_strided_cuda((8, 672, 1, 1), (672, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_52, x_se_40], Original ATen: [aten.cat, aten.mean]
        triton_red_fused_cat_mean_47.run(buf122, buf123, arg279_1, arg280_1, arg281_1, arg282_1, buf124, buf126, 5376, 196, grid=grid(5376), stream=stream0)
        del arg279_1
        del arg280_1
        del arg281_1
        del arg282_1
        del buf122
        del buf123
        # Topologically Sorted Source Nodes: [x_se_40, x_se_41], Original ATen: [aten.mean, aten.convolution]
        buf127 = extern_kernels.convolution(buf126, arg283_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 168, 1, 1), (168, 1, 1, 1))
        del arg283_1
        del buf126
        buf128 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_se_40, x_se_41, x_se_42], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_48.run(buf128, arg284_1, 1344, grid=grid(1344), stream=stream0)
        del arg284_1
        # Topologically Sorted Source Nodes: [x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf129 = extern_kernels.convolution(buf128, arg285_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg285_1
        del buf128
        buf130 = empty_strided_cuda((8, 672, 14, 14), (131712, 1, 9408, 672), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_40, x_se_41, x_se_42, x_se_43, hardsigmoid_10, x_117], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_49.run(buf124, buf129, arg286_1, buf130, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg286_1
        del buf124
        # Topologically Sorted Source Nodes: [x_se_40, x_se_41, x_se_42, x_se_43, hardsigmoid_10, x_117, input_303], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf131 = extern_kernels.convolution(buf130, arg287_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 56, 14, 14), (10976, 1, 784, 56))
        del arg287_1
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [input_304], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf132, arg288_1, arg289_1, arg290_1, arg291_1, 87808, grid=grid(87808), stream=stream0)
        del arg288_1
        del arg289_1
        del arg290_1
        del arg291_1
        # Topologically Sorted Source Nodes: [input_305], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, arg292_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
        assert_size_stride(buf133, (8, 56, 14, 14), (10976, 1, 784, 56))
        del arg292_1
        buf134 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [out_53, x_119], Original ATen: [aten.cat, aten.add]
        triton_poi_fused_add_cat_50.run(buf134, buf132, buf133, arg293_1, arg294_1, arg295_1, arg296_1, 175616, grid=grid(175616), stream=stream0)
        del arg293_1
        del arg294_1
        del arg295_1
        del arg296_1
        del buf132
        del buf133
        # Topologically Sorted Source Nodes: [input_307], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, arg297_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (8, 336, 14, 14), (65856, 1, 4704, 336))
        del arg297_1
        buf136 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [input_308, input_309], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf136, arg298_1, arg299_1, arg300_1, arg301_1, 526848, grid=grid(526848), stream=stream0)
        del arg298_1
        del arg299_1
        del arg300_1
        del arg301_1
        # Topologically Sorted Source Nodes: [input_310], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, arg302_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
        assert_size_stride(buf137, (8, 336, 14, 14), (65856, 1, 4704, 336))
        del arg302_1
        buf138 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [out_54], Original ATen: [aten.cat]
        triton_poi_fused_cat_51.run(buf136, buf137, arg303_1, arg304_1, arg305_1, arg306_1, buf138, 1053696, grid=grid(1053696), stream=stream0)
        del arg303_1
        del arg304_1
        del arg305_1
        del arg306_1
        del buf136
        del buf137
        # Topologically Sorted Source Nodes: [out_54, x_121], Original ATen: [aten.cat, aten.convolution]
        buf139 = extern_kernels.convolution(buf138, arg307_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf139, (8, 672, 7, 7), (32928, 1, 4704, 672))
        del arg307_1
        del buf138
        buf140 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_52.run(buf140, arg308_1, arg309_1, arg310_1, arg311_1, 263424, grid=grid(263424), stream=stream0)
        del arg308_1
        del arg309_1
        del arg310_1
        del arg311_1
        buf142 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [x_se_44], Original ATen: [aten.mean]
        triton_per_fused_mean_53.run(buf140, buf142, 5376, 49, grid=grid(5376), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_44, x_se_45], Original ATen: [aten.mean, aten.convolution]
        buf143 = extern_kernels.convolution(buf142, arg312_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 168, 1, 1), (168, 1, 1, 1))
        del arg312_1
        del buf142
        buf144 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_se_44, x_se_45, x_se_46], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_48.run(buf144, arg313_1, 1344, grid=grid(1344), stream=stream0)
        del arg313_1
        # Topologically Sorted Source Nodes: [x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf145 = extern_kernels.convolution(buf144, arg314_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg314_1
        del buf144
        buf146 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_se_44, x_se_45, x_se_46, x_se_47, hardsigmoid_11, x_123], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_54.run(buf146, buf145, arg315_1, 263424, grid=grid(263424), stream=stream0)
        del arg315_1
        del buf145
        # Topologically Sorted Source Nodes: [x_se_44, x_se_45, x_se_46, x_se_47, hardsigmoid_11, x_123, input_313], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf147 = extern_kernels.convolution(buf146, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (8, 80, 7, 7), (3920, 1, 560, 80))
        del arg316_1
        del buf146
        buf148 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [input_314], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_55.run(buf148, arg317_1, arg318_1, arg319_1, arg320_1, 31360, grid=grid(31360), stream=stream0)
        del arg317_1
        del arg318_1
        del arg319_1
        del arg320_1
        # Topologically Sorted Source Nodes: [input_315], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, arg321_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf149, (8, 80, 7, 7), (3920, 1, 560, 80))
        del arg321_1
        # Topologically Sorted Source Nodes: [input_317], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf134, arg326_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf150, (8, 112, 7, 7), (5488, 1, 784, 112))
        del arg326_1
        del buf134
        buf151 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [input_318], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_56.run(buf151, arg327_1, arg328_1, arg329_1, arg330_1, 43904, grid=grid(43904), stream=stream0)
        del arg327_1
        del arg328_1
        del arg329_1
        del arg330_1
        # Topologically Sorted Source Nodes: [input_318, input_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg331_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 160, 7, 7), (7840, 1, 1120, 160))
        del arg331_1
        del buf151
        buf153 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [out_55, input_320, x_125], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_57.run(buf153, buf148, buf149, arg322_1, arg323_1, arg324_1, arg325_1, arg332_1, arg333_1, arg334_1, arg335_1, 62720, grid=grid(62720), stream=stream0)
        del arg322_1
        del arg323_1
        del arg324_1
        del arg325_1
        del arg332_1
        del arg333_1
        del arg334_1
        del arg335_1
        del buf148
        del buf149
        # Topologically Sorted Source Nodes: [input_321], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, arg336_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 480, 7, 7), (23520, 1, 3360, 480))
        del arg336_1
        buf155 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [input_322, input_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf155, arg337_1, arg338_1, arg339_1, arg340_1, 188160, grid=grid(188160), stream=stream0)
        del arg337_1
        del arg338_1
        del arg339_1
        del arg340_1
        # Topologically Sorted Source Nodes: [input_324], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, arg341_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf156, (8, 480, 7, 7), (23520, 1, 3360, 480))
        del arg341_1
        buf157 = reinterpret_tensor(buf106, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [out_56], Original ATen: [aten.cat]
        triton_poi_fused_cat_59.run(buf155, buf156, arg342_1, arg343_1, arg344_1, arg345_1, buf157, 376320, grid=grid(376320), stream=stream0)
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        del buf155
        del buf156
        # Topologically Sorted Source Nodes: [out_56, input_327], Original ATen: [aten.cat, aten.convolution]
        buf158 = extern_kernels.convolution(buf157, arg346_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 80, 7, 7), (3920, 1, 560, 80))
        del arg346_1
        buf159 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [input_328], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_55.run(buf159, arg347_1, arg348_1, arg349_1, arg350_1, 31360, grid=grid(31360), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        del arg350_1
        # Topologically Sorted Source Nodes: [input_329], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, arg351_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf160, (8, 80, 7, 7), (3920, 1, 560, 80))
        del arg351_1
        buf161 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [out_57, x_128], Original ATen: [aten.cat, aten.add]
        triton_poi_fused_add_cat_60.run(buf161, buf159, buf160, arg352_1, arg353_1, arg354_1, arg355_1, 62720, grid=grid(62720), stream=stream0)
        del arg352_1
        del arg353_1
        del arg354_1
        del arg355_1
        del buf159
        del buf160
        # Topologically Sorted Source Nodes: [input_331], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, arg356_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (8, 480, 7, 7), (23520, 1, 3360, 480))
        del arg356_1
        buf163 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [input_332, input_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf163, arg357_1, arg358_1, arg359_1, arg360_1, 188160, grid=grid(188160), stream=stream0)
        del arg357_1
        del arg358_1
        del arg359_1
        del arg360_1
        # Topologically Sorted Source Nodes: [input_334], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, arg361_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf164, (8, 480, 7, 7), (23520, 1, 3360, 480))
        del arg361_1
        buf165 = reinterpret_tensor(buf157, (8, 960, 7, 7), (47040, 49, 7, 1), 0); del buf157  # reuse
        buf167 = empty_strided_cuda((8, 960, 1, 1), (960, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_58, x_se_48], Original ATen: [aten.cat, aten.mean]
        triton_per_fused_cat_mean_61.run(buf163, buf164, arg362_1, arg363_1, arg364_1, arg365_1, buf165, buf167, 7680, 49, grid=grid(7680), stream=stream0)
        del arg362_1
        del arg363_1
        del arg364_1
        del arg365_1
        del buf163
        del buf164
        # Topologically Sorted Source Nodes: [x_se_48, x_se_49], Original ATen: [aten.mean, aten.convolution]
        buf168 = extern_kernels.convolution(buf167, arg366_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg366_1
        del buf167
        buf169 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_se_48, x_se_49, x_se_50], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_62.run(buf169, arg367_1, 1920, grid=grid(1920), stream=stream0)
        del arg367_1
        # Topologically Sorted Source Nodes: [x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf170 = extern_kernels.convolution(buf169, arg368_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (8, 960, 1, 1), (960, 1, 1, 1))
        del arg368_1
        del buf169
        buf171 = reinterpret_tensor(buf105, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_se_48, x_se_49, x_se_50, x_se_51, hardsigmoid_12, x_130], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_63.run(buf165, buf170, arg369_1, buf171, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg369_1
        # Topologically Sorted Source Nodes: [x_se_48, x_se_49, x_se_50, x_se_51, hardsigmoid_12, x_130, input_337], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf172 = extern_kernels.convolution(buf171, arg370_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 80, 7, 7), (3920, 1, 560, 80))
        del arg370_1
        buf173 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [input_338], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_55.run(buf173, arg371_1, arg372_1, arg373_1, arg374_1, 31360, grid=grid(31360), stream=stream0)
        del arg371_1
        del arg372_1
        del arg373_1
        del arg374_1
        # Topologically Sorted Source Nodes: [input_339], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, arg375_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf174, (8, 80, 7, 7), (3920, 1, 560, 80))
        del arg375_1
        buf175 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [out_59, x_132], Original ATen: [aten.cat, aten.add]
        triton_poi_fused_add_cat_60.run(buf175, buf173, buf174, arg376_1, arg377_1, arg378_1, arg379_1, 62720, grid=grid(62720), stream=stream0)
        del arg376_1
        del arg377_1
        del arg378_1
        del arg379_1
        del buf173
        del buf174
        # Topologically Sorted Source Nodes: [input_341], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg380_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 480, 7, 7), (23520, 1, 3360, 480))
        del arg380_1
        buf177 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [input_342, input_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf177, arg381_1, arg382_1, arg383_1, arg384_1, 188160, grid=grid(188160), stream=stream0)
        del arg381_1
        del arg382_1
        del arg383_1
        del arg384_1
        # Topologically Sorted Source Nodes: [input_344], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, arg385_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf178, (8, 480, 7, 7), (23520, 1, 3360, 480))
        del arg385_1
        buf179 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [out_60], Original ATen: [aten.cat]
        triton_poi_fused_cat_59.run(buf177, buf178, arg386_1, arg387_1, arg388_1, arg389_1, buf179, 376320, grid=grid(376320), stream=stream0)
        del arg386_1
        del arg387_1
        del arg388_1
        del arg389_1
        del buf177
        del buf178
        # Topologically Sorted Source Nodes: [out_60, input_347], Original ATen: [aten.cat, aten.convolution]
        buf180 = extern_kernels.convolution(buf179, arg390_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 80, 7, 7), (3920, 1, 560, 80))
        del arg390_1
        buf181 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [input_348], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_55.run(buf181, arg391_1, arg392_1, arg393_1, arg394_1, 31360, grid=grid(31360), stream=stream0)
        del arg391_1
        del arg392_1
        del arg393_1
        del arg394_1
        # Topologically Sorted Source Nodes: [input_349], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, arg395_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf182, (8, 80, 7, 7), (3920, 1, 560, 80))
        del arg395_1
        buf183 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [out_61, x_135], Original ATen: [aten.cat, aten.add]
        triton_poi_fused_add_cat_60.run(buf183, buf181, buf182, arg396_1, arg397_1, arg398_1, arg399_1, 62720, grid=grid(62720), stream=stream0)
        del arg396_1
        del arg397_1
        del arg398_1
        del arg399_1
        del buf181
        del buf182
        # Topologically Sorted Source Nodes: [input_351], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, arg400_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 480, 7, 7), (23520, 1, 3360, 480))
        del arg400_1
        buf185 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [input_352, input_353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf185, arg401_1, arg402_1, arg403_1, arg404_1, 188160, grid=grid(188160), stream=stream0)
        del arg401_1
        del arg402_1
        del arg403_1
        del arg404_1
        # Topologically Sorted Source Nodes: [input_354], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, arg405_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf186, (8, 480, 7, 7), (23520, 1, 3360, 480))
        del arg405_1
        buf187 = reinterpret_tensor(buf179, (8, 960, 7, 7), (47040, 49, 7, 1), 0); del buf179  # reuse
        buf189 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [out_62, x_se_52], Original ATen: [aten.cat, aten.mean]
        triton_per_fused_cat_mean_61.run(buf185, buf186, arg406_1, arg407_1, arg408_1, arg409_1, buf187, buf189, 7680, 49, grid=grid(7680), stream=stream0)
        del arg406_1
        del arg407_1
        del arg408_1
        del arg409_1
        del buf185
        del buf186
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53], Original ATen: [aten.mean, aten.convolution]
        buf190 = extern_kernels.convolution(buf189, arg410_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg410_1
        del buf189
        buf191 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53, x_se_54], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_62.run(buf191, arg411_1, 1920, grid=grid(1920), stream=stream0)
        del arg411_1
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf192 = extern_kernels.convolution(buf191, arg412_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 960, 1, 1), (960, 1, 1, 1))
        del arg412_1
        del buf191
        buf193 = reinterpret_tensor(buf165, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53, x_se_54, x_se_55, hardsigmoid_13, x_137], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_63.run(buf187, buf192, arg413_1, buf193, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg413_1
        del buf187
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53, x_se_54, x_se_55, hardsigmoid_13, x_137, input_357], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf194 = extern_kernels.convolution(buf193, arg414_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 80, 7, 7), (3920, 1, 560, 80))
        del arg414_1
        del buf193
        buf195 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [input_358], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_55.run(buf195, arg415_1, arg416_1, arg417_1, arg418_1, 31360, grid=grid(31360), stream=stream0)
        del arg415_1
        del arg416_1
        del arg417_1
        del arg418_1
        # Topologically Sorted Source Nodes: [input_359], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, arg419_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf196, (8, 80, 7, 7), (3920, 1, 560, 80))
        del arg419_1
        buf197 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [out_63, x_139], Original ATen: [aten.cat, aten.add]
        triton_poi_fused_add_cat_60.run(buf197, buf195, buf196, arg420_1, arg421_1, arg422_1, arg423_1, 62720, grid=grid(62720), stream=stream0)
        del arg420_1
        del arg421_1
        del arg422_1
        del arg423_1
        del buf195
        del buf196
        # Topologically Sorted Source Nodes: [out_63, x_139, x_140], Original ATen: [aten.cat, aten.add, aten.convolution]
        buf198 = extern_kernels.convolution(buf197, arg424_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del arg424_1
        del buf197
        buf200 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [x_141, x_142, x_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_64.run(buf198, arg425_1, arg426_1, arg427_1, arg428_1, buf200, 7680, 49, grid=grid(7680), stream=stream0)
        del arg425_1
        del arg426_1
        del arg427_1
        del arg428_1
        del buf198
        # Topologically Sorted Source Nodes: [x_141, x_142, x_143, x_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean, aten.convolution]
        buf201 = extern_kernels.convolution(buf200, arg429_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 1280, 1, 1), (1280, 1, 1, 1))
        del arg429_1
        del buf200
        buf202 = reinterpret_tensor(buf201, (8, 1280, 1, 1), (1280, 1, 10240, 10240), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [x_141, x_142, x_143, x_144, x_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_65.run(buf202, arg430_1, 10240, grid=grid(10240), stream=stream0)
        del arg430_1
        buf203 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_147], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg432_1, reinterpret_tensor(buf202, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg431_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf203)
        del arg431_1
        del arg432_1
        del buf202
    return (buf203, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((24, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((12, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((12, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((24, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((36, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((12, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((12, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((36, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((20, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((72, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((20, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((20, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((40, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((60, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((20, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((20, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((80, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((100, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((100, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((40, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((92, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((92, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((40, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((92, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((92, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((40, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((56, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((112, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((56, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((80, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((160, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((1280, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ghostnet_100', benchmark_compiled_module)
