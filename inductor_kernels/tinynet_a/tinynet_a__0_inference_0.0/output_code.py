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


# kernel path: /tmp/torchinductor_sahanp/5q/c5q2u3gmorljio7pbiaw4ewevtihsxrgvzhp4vesdhmenzy3qunc.py
# Topologically Sorted Source Nodes: [x_189], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_189 => convolution_96
# Graph fragment:
#   %convolution_96 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    xnumel = 36864
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
    tmp0 = tl.load(in_ptr0 + (x2 + (36864*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (110592*y1)), tmp0, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fn/cfnqc5t7ruox32h42p5hkjg336a6auwfmlumtz3g3aru5iaouonc.py
# Topologically Sorted Source Nodes: [x_189], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_189 => convolution_96
# Graph fragment:
#   %convolution_96 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/f5/cf5woqca7c7n3qfbsv56xqunclhdt434omcexoexzrxedtlwbomn.py
# Topologically Sorted Source Nodes: [x_190, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_190 => add_129, mul_252, mul_253, sub_58
#   x_191 => mul_254, sigmoid_77
# Graph fragment:
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_96, %unsqueeze_465), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_467), kwargs = {})
#   %mul_253 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_252, %unsqueeze_469), kwargs = {})
#   %add_129 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_253, %unsqueeze_471), kwargs = {})
#   %sigmoid_77 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_129,), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_129, %sigmoid_77), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vb/cvbon4yiix4hr2vtidec5amk4fchjbalok56sugc462xvss35ass.py
# Topologically Sorted Source Nodes: [x_193], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_193 => add_131, mul_256, mul_257, sub_59
# Graph fragment:
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_97, %unsqueeze_473), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_475), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %unsqueeze_477), kwargs = {})
#   %add_131 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %unsqueeze_479), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xz/cxz3g57a6npogqfu6ytqr7l2yrpvcb5uxtm4uwintyo5x5372wu3.py
# Topologically Sorted Source Nodes: [x_194, x_se_76], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_194 => mul_258, sigmoid_78
#   x_se_76 => mean_20
# Graph fragment:
#   %sigmoid_78 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_131,), kwargs = {})
#   %mul_258 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_131, %sigmoid_78), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_258, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_4 = async_compile.triton('triton_red_fused_mean_silu_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_4(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yh/cyhwn2jqawy6xhsbmd6q464lg4wrrgduphmxpljqouls5ss2c4lq.py
# Topologically Sorted Source Nodes: [x_194, x_se_76], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_194 => mul_258, sigmoid_78
#   x_se_76 => mean_20
# Graph fragment:
#   %sigmoid_78 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_131,), kwargs = {})
#   %mul_258 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_131, %sigmoid_78), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_258, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_5 = async_compile.triton('triton_red_fused_mean_silu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_5(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (2304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = 9216.0
    tmp5 = tmp2 / tmp4
    tl.store(out_ptr1 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/s7/cs73aqtd4o47swivajt3nmevk5nxbdzsbm5zpz4ct6hzbt5ph6ap.py
# Topologically Sorted Source Nodes: [x_194, x_se_76, x_se_77, x_se_78], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_194 => mul_258, sigmoid_78
#   x_se_76 => mean_20
#   x_se_77 => convolution_98
#   x_se_78 => mul_259, sigmoid_79
# Graph fragment:
#   %sigmoid_78 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_131,), kwargs = {})
#   %mul_258 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_131, %sigmoid_78), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_258, [2, 3], True), kwargs = {})
#   %convolution_98 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_20, %arg11_1, %arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_79 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_98,), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_98, %sigmoid_79), kwargs = {})
triton_poi_fused_convolution_mean_silu_6 = async_compile.triton('triton_poi_fused_convolution_mean_silu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d4/cd4xfuoxxws3zoqlhfv3acx7b67d4atkb4kpj2oxwoqqfq5di7v5.py
# Topologically Sorted Source Nodes: [x_194, x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, x_195], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_19 => sigmoid_80
#   x_194 => mul_258, sigmoid_78
#   x_195 => mul_260
#   x_se_76 => mean_20
#   x_se_77 => convolution_98
#   x_se_78 => mul_259, sigmoid_79
#   x_se_79 => convolution_99
# Graph fragment:
#   %sigmoid_78 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_131,), kwargs = {})
#   %mul_258 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_131, %sigmoid_78), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_258, [2, 3], True), kwargs = {})
#   %convolution_98 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_20, %arg11_1, %arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_79 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_98,), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_98, %sigmoid_79), kwargs = {})
#   %convolution_99 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_259, %arg13_1, %arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_80 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_99,), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_258, %sigmoid_80), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_7 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 32
    x2 = (xindex // 294912)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wr/cwrhzojzxw5ssgcir74m6murpx355ohb7ehibwdlwab735xatlpu.py
# Topologically Sorted Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_197 => add_133, mul_262, mul_263, sub_60
# Graph fragment:
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_481), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_483), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %unsqueeze_485), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %unsqueeze_487), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
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


# kernel path: /tmp/torchinductor_sahanp/4p/c4pu7zmlami6mnowfixr74babcbhvovh67kmufv53oocajq7f2lk.py
# Topologically Sorted Source Nodes: [x_199, x_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_199 => add_135, mul_265, mul_266, sub_61
#   x_200 => mul_267, sigmoid_81
# Graph fragment:
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_101, %unsqueeze_489), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_265, %unsqueeze_493), kwargs = {})
#   %add_135 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_266, %unsqueeze_495), kwargs = {})
#   %sigmoid_81 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_135,), kwargs = {})
#   %mul_267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_135, %sigmoid_81), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7077888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 96
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/q3/cq3a2gocf5jq7aednhvkaonl3kv576wx36qclixnlkucgeovbuen.py
# Topologically Sorted Source Nodes: [x_202], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_202 => add_137, mul_269, mul_270, sub_62
# Graph fragment:
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_102, %unsqueeze_497), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %unsqueeze_499), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_269, %unsqueeze_501), kwargs = {})
#   %add_137 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_270, %unsqueeze_503), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 96
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


# kernel path: /tmp/torchinductor_sahanp/uq/cuqrdhridg3zcyqj572usynnazk5kx3t36m475yblvazuxfdnkki.py
# Topologically Sorted Source Nodes: [x_203, x_se_80], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_203 => mul_271, sigmoid_82
#   x_se_80 => mean_21
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_137,), kwargs = {})
#   %mul_271 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, %sigmoid_82), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_271, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_11 = async_compile.triton('triton_red_fused_mean_silu_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_11(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13824
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/47/c47kosjyps3syb7q7ssgmxqulgo6kifzot6ifu5dsdsbqo42fp6o.py
# Topologically Sorted Source Nodes: [x_203, x_se_80], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_203 => mul_271, sigmoid_82
#   x_se_80 => mean_21
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_137,), kwargs = {})
#   %mul_271 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, %sigmoid_82), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_271, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_12 = async_compile.triton('triton_per_fused_mean_silu_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_12(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 18
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 96
    x1 = (xindex // 96)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (1728*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 2304.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dm/cdmr333tfb6blrgz6utzynsex2rm5upfnzmcbrpz46yptap6bqj2.py
# Topologically Sorted Source Nodes: [x_203, x_se_80, x_se_81, x_se_82], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_203 => mul_271, sigmoid_82
#   x_se_80 => mean_21
#   x_se_81 => convolution_103
#   x_se_82 => mul_272, sigmoid_83
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_137,), kwargs = {})
#   %mul_271 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, %sigmoid_82), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_271, [2, 3], True), kwargs = {})
#   %convolution_103 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_21, %arg30_1, %arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_103,), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_103, %sigmoid_83), kwargs = {})
triton_poi_fused_convolution_mean_silu_13 = async_compile.triton('triton_poi_fused_convolution_mean_silu_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jg/cjgg5bz5h2izgnr3zghiq2fe2epyl34kd6ucsygfgjscjvu7uews.py
# Topologically Sorted Source Nodes: [x_203, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, x_204], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_20 => sigmoid_84
#   x_203 => mul_271, sigmoid_82
#   x_204 => mul_273
#   x_se_80 => mean_21
#   x_se_81 => convolution_103
#   x_se_82 => mul_272, sigmoid_83
#   x_se_83 => convolution_104
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_137,), kwargs = {})
#   %mul_271 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, %sigmoid_82), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_271, [2, 3], True), kwargs = {})
#   %convolution_103 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_21, %arg30_1, %arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_103,), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_103, %sigmoid_83), kwargs = {})
#   %convolution_104 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_272, %arg32_1, %arg33_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_84 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_104,), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %sigmoid_84), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_14 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_14(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 96
    x2 = (xindex // 221184)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (96*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wj/cwjchuvq5sxfgt4w4eiqolqw3p3msse3bn6u3o42226r6q5tq6tf.py
# Topologically Sorted Source Nodes: [x_206], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_206 => add_139, mul_275, mul_276, sub_63
# Graph fragment:
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_505), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_275, %unsqueeze_509), kwargs = {})
#   %add_139 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_276, %unsqueeze_511), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
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


# kernel path: /tmp/torchinductor_sahanp/xj/cxjqfdrdzxmny3rfbctkbapbcsbh5tbi6f6naropsideemr2y43v.py
# Topologically Sorted Source Nodes: [x_208, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_208 => add_141, mul_278, mul_279, sub_64
#   x_209 => mul_280, sigmoid_85
# Graph fragment:
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_106, %unsqueeze_513), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_515), kwargs = {})
#   %mul_279 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_278, %unsqueeze_517), kwargs = {})
#   %add_141 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_279, %unsqueeze_519), kwargs = {})
#   %sigmoid_85 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_141,), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_141, %sigmoid_85), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7n/c7nxicpvxbc7llcekz2x3dshoixggxocbkcaftf6gaibh2fcgb3a.py
# Topologically Sorted Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_211 => add_143, mul_282, mul_283, sub_65
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_107, %unsqueeze_521), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_282, %unsqueeze_525), kwargs = {})
#   %add_143 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_283, %unsqueeze_527), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/er/cereay3tgu63nxgvqb2ekfkgy3z7klba3i5bxxjfgc6hbvctjgil.py
# Topologically Sorted Source Nodes: [x_212, x_se_84], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_212 => mul_284, sigmoid_86
#   x_se_84 => mean_22
# Graph fragment:
#   %sigmoid_86 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_143,), kwargs = {})
#   %mul_284 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_143, %sigmoid_86), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_284, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_18 = async_compile.triton('triton_red_fused_mean_silu_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_18(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20736
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5u/c5uoqjhr26xhadaq3sif7q4ro6c7uq3i3udlgyz5seasfkpwaboj.py
# Topologically Sorted Source Nodes: [x_212, x_se_84], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_212 => mul_284, sigmoid_86
#   x_se_84 => mean_22
# Graph fragment:
#   %sigmoid_86 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_143,), kwargs = {})
#   %mul_284 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_143, %sigmoid_86), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_284, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_19 = async_compile.triton('triton_per_fused_mean_silu_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_19(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 18
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 144
    x1 = (xindex // 144)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (2592*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 2304.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cg/ccgqekzfhs2d5ojy4wqoy6llmdre55lvldigodmtuurul4464c3b.py
# Topologically Sorted Source Nodes: [x_212, x_se_84, x_se_85, x_se_86], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_212 => mul_284, sigmoid_86
#   x_se_84 => mean_22
#   x_se_85 => convolution_108
#   x_se_86 => mul_285, sigmoid_87
# Graph fragment:
#   %sigmoid_86 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_143,), kwargs = {})
#   %mul_284 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_143, %sigmoid_86), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_284, [2, 3], True), kwargs = {})
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_22, %arg49_1, %arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_87 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_108,), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, %sigmoid_87), kwargs = {})
triton_poi_fused_convolution_mean_silu_20 = async_compile.triton('triton_poi_fused_convolution_mean_silu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ie/cieuiyj2iymxmw5257put6xrvjgkwfu5t4r773mcz5abm7kikufp.py
# Topologically Sorted Source Nodes: [x_212, x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, x_213], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_21 => sigmoid_88
#   x_212 => mul_284, sigmoid_86
#   x_213 => mul_286
#   x_se_84 => mean_22
#   x_se_85 => convolution_108
#   x_se_86 => mul_285, sigmoid_87
#   x_se_87 => convolution_109
# Graph fragment:
#   %sigmoid_86 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_143,), kwargs = {})
#   %mul_284 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_143, %sigmoid_86), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_284, [2, 3], True), kwargs = {})
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_22, %arg49_1, %arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_87 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_108,), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, %sigmoid_87), kwargs = {})
#   %convolution_109 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_285, %arg51_1, %arg52_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_88 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_109,), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_284, %sigmoid_88), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_21 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_21(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 144
    x2 = (xindex // 331776)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (144*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fa/cfajhnfhs4s2wlm7e2qkvmefxik2abejrop34jqoqj5nwp2gn5fz.py
# Topologically Sorted Source Nodes: [x_215, x_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_215 => add_145, mul_288, mul_289, sub_66
#   x_216 => add_146
# Graph fragment:
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_110, %unsqueeze_529), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_288, %unsqueeze_533), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_289, %unsqueeze_535), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_145, %add_139), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
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


# kernel path: /tmp/torchinductor_sahanp/kj/ckjva7mpihcz7q6h6jr7qbae6dstdo5oiuodb24vhh3nr4s5p6cn.py
# Topologically Sorted Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_221 => add_150, mul_295, mul_296, sub_68
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_112, %unsqueeze_545), kwargs = {})
#   %mul_295 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_295, %unsqueeze_549), kwargs = {})
#   %add_150 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_296, %unsqueeze_551), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 663552
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cm/ccmxuh6rkwnq23l3pwulm2ckghizsxebjwojifdojd6kvjiigyfk.py
# Topologically Sorted Source Nodes: [x_222, x_se_88], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_222 => mul_297, sigmoid_90
#   x_se_88 => mean_23
# Graph fragment:
#   %sigmoid_90 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_150,), kwargs = {})
#   %mul_297 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_150, %sigmoid_90), kwargs = {})
#   %mean_23 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_297, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_24 = async_compile.triton('triton_red_fused_mean_silu_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_24(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5760
    rnumel = 116
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 144) % 5
    x0 = xindex % 144
    x2 = (xindex // 720)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (116*x1)
        tmp1 = tl.full([1, 1], 576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (144*((r3 + (116*x1)) % 576)) + (82944*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.sigmoid(tmp3)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yg/cyg6567sikymen7plwuxmqeptiaztsa3wmlwuthhe3ebarajuck6.py
# Topologically Sorted Source Nodes: [x_222, x_se_88], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_222 => mul_297, sigmoid_90
#   x_se_88 => mean_23
# Graph fragment:
#   %sigmoid_90 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_150,), kwargs = {})
#   %mul_297 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_150, %sigmoid_90), kwargs = {})
#   %mean_23 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_297, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_25 = async_compile.triton('triton_per_fused_mean_silu_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_25(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 144
    x1 = (xindex // 144)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 576.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7r/c7r4octmtedhuirouioopdcoxa55fbszh3pn5555zdbxptzh7fa7.py
# Topologically Sorted Source Nodes: [x_222, x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, x_223], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_22 => sigmoid_92
#   x_222 => mul_297, sigmoid_90
#   x_223 => mul_299
#   x_se_88 => mean_23
#   x_se_89 => convolution_113
#   x_se_90 => mul_298, sigmoid_91
#   x_se_91 => convolution_114
# Graph fragment:
#   %sigmoid_90 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_150,), kwargs = {})
#   %mul_297 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_150, %sigmoid_90), kwargs = {})
#   %mean_23 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_297, [2, 3], True), kwargs = {})
#   %convolution_113 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_23, %arg68_1, %arg69_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_91 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_113,), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_113, %sigmoid_91), kwargs = {})
#   %convolution_114 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_298, %arg70_1, %arg71_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_92 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_114,), kwargs = {})
#   %mul_299 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_297, %sigmoid_92), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_26 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_26(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 663552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 144
    x2 = (xindex // 82944)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (144*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5h/c5hsn7zvdkwme4avzpfc7j2r7c6lzgjc2jjkmgoizmxf42weyw7e.py
# Topologically Sorted Source Nodes: [x_225], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_225 => add_152, mul_301, mul_302, sub_69
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_553), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_301, %unsqueeze_557), kwargs = {})
#   %add_152 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_302, %unsqueeze_559), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 184320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 40
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


# kernel path: /tmp/torchinductor_sahanp/hl/chllurjfukillev35ouvds5cepxayccprxgxxy5sfr5b2au7alkd.py
# Topologically Sorted Source Nodes: [x_227, x_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_227 => add_154, mul_304, mul_305, sub_70
#   x_228 => mul_306, sigmoid_93
# Graph fragment:
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_116, %unsqueeze_561), kwargs = {})
#   %mul_304 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_563), kwargs = {})
#   %mul_305 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_304, %unsqueeze_565), kwargs = {})
#   %add_154 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_305, %unsqueeze_567), kwargs = {})
#   %sigmoid_93 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_154,), kwargs = {})
#   %mul_306 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_154, %sigmoid_93), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1105920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 240
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/25/c25f7v753bfscsgztfjjbtwinn7u6qee5ogpag72fekalyz5yjhh.py
# Topologically Sorted Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_230 => add_156, mul_308, mul_309, sub_71
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_117, %unsqueeze_569), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_571), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_308, %unsqueeze_573), kwargs = {})
#   %add_156 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_309, %unsqueeze_575), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1105920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 240
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


# kernel path: /tmp/torchinductor_sahanp/jh/cjhxr3akliu73olupbgrraxtlvp672pwp2zwcq5imdkhcnuspm7q.py
# Topologically Sorted Source Nodes: [x_231, x_se_92], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_231 => mul_310, sigmoid_94
#   x_se_92 => mean_24
# Graph fragment:
#   %sigmoid_94 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_156,), kwargs = {})
#   %mul_310 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_156, %sigmoid_94), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_310, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_30 = async_compile.triton('triton_red_fused_mean_silu_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_30(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9600
    rnumel = 116
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 240) % 5
    x0 = xindex % 240
    x2 = (xindex // 1200)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (116*x1)
        tmp1 = tl.full([1, 1], 576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (240*((r3 + (116*x1)) % 576)) + (138240*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.sigmoid(tmp3)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oq/coqq74e6k3noe57idi6pmfhnfkey3twrxgrfroqdubiobze2r6ph.py
# Topologically Sorted Source Nodes: [x_231, x_se_92], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_231 => mul_310, sigmoid_94
#   x_se_92 => mean_24
# Graph fragment:
#   %sigmoid_94 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_156,), kwargs = {})
#   %mul_310 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_156, %sigmoid_94), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_310, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_31 = async_compile.triton('triton_per_fused_mean_silu_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_31(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 240
    x1 = (xindex // 240)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (1200*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 576.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uc/cucwowg2ingp32lelfugxfd4ybiijtj6roo6fbebzz2fxdygrcxj.py
# Topologically Sorted Source Nodes: [x_231, x_se_92, x_se_93, x_se_94], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_231 => mul_310, sigmoid_94
#   x_se_92 => mean_24
#   x_se_93 => convolution_118
#   x_se_94 => mul_311, sigmoid_95
# Graph fragment:
#   %sigmoid_94 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_156,), kwargs = {})
#   %mul_310 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_156, %sigmoid_94), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_310, [2, 3], True), kwargs = {})
#   %convolution_118 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_24, %arg87_1, %arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_95 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_118,), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_118, %sigmoid_95), kwargs = {})
triton_poi_fused_convolution_mean_silu_32 = async_compile.triton('triton_poi_fused_convolution_mean_silu_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_32(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 10
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/27/c27k7v2xptpvd5dqmp3okfwqnq2h2yawiqtotbqiaf6gr7anqj2f.py
# Topologically Sorted Source Nodes: [x_231, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, x_232], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_23 => sigmoid_96
#   x_231 => mul_310, sigmoid_94
#   x_232 => mul_312
#   x_se_92 => mean_24
#   x_se_93 => convolution_118
#   x_se_94 => mul_311, sigmoid_95
#   x_se_95 => convolution_119
# Graph fragment:
#   %sigmoid_94 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_156,), kwargs = {})
#   %mul_310 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_156, %sigmoid_94), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_310, [2, 3], True), kwargs = {})
#   %convolution_118 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_24, %arg87_1, %arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_95 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_118,), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_118, %sigmoid_95), kwargs = {})
#   %convolution_119 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_311, %arg89_1, %arg90_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_96 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_119,), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, %sigmoid_96), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_33 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_33(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1105920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 240
    x2 = (xindex // 138240)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (240*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ek/cekojcbh5v5s3ytjtgc72cuosk7iqre4t3epjbyqxyd2a7gskzn3.py
# Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_234 => add_158, mul_314, mul_315, sub_72
#   x_235 => add_159
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_120, %unsqueeze_577), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_314, %unsqueeze_581), kwargs = {})
#   %add_158 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_315, %unsqueeze_583), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_158, %add_152), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 184320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 40
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


# kernel path: /tmp/torchinductor_sahanp/w5/cw5ehkunq674obrewaaqfaum7bt5egzkzv6wvlc2isqur7qgltpd.py
# Topologically Sorted Source Nodes: [x_240], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_240 => add_163, mul_321, mul_322, sub_74
# Graph fragment:
#   %sub_74 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_122, %unsqueeze_593), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_74, %unsqueeze_595), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_321, %unsqueeze_597), kwargs = {})
#   %add_163 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_322, %unsqueeze_599), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 276480
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


# kernel path: /tmp/torchinductor_sahanp/fe/cfepdrabeiuov2pwoo3kz6wzdtzlhwao34tgvswwbbhziuue3qo5.py
# Topologically Sorted Source Nodes: [x_241, x_se_96], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_241 => mul_323, sigmoid_98
#   x_se_96 => mean_25
# Graph fragment:
#   %sigmoid_98 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_323 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_98), kwargs = {})
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_323, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_36 = async_compile.triton('triton_red_fused_mean_silu_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_36(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (17280*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rd/crdg5om5tvykf5gwsipfomugnwy6oq77f5vfjgec7lgzrrrc4ir2.py
# Topologically Sorted Source Nodes: [x_241, x_se_96], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_241 => mul_323, sigmoid_98
#   x_se_96 => mean_25
# Graph fragment:
#   %sigmoid_98 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_323 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_98), kwargs = {})
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_323, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_37 = async_compile.triton('triton_per_fused_mean_silu_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_37(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 240
    x1 = (xindex // 240)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (480*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 144.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rd/crdoshl7zhw22mebp5qffnu5fwh2iks7stdjnji6ezs5dvgi4zni.py
# Topologically Sorted Source Nodes: [x_241, x_se_96, x_se_97, x_se_98, x_se_99, sigmoid_24, x_242], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_24 => sigmoid_100
#   x_241 => mul_323, sigmoid_98
#   x_242 => mul_325
#   x_se_96 => mean_25
#   x_se_97 => convolution_123
#   x_se_98 => mul_324, sigmoid_99
#   x_se_99 => convolution_124
# Graph fragment:
#   %sigmoid_98 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_323 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_98), kwargs = {})
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_323, [2, 3], True), kwargs = {})
#   %convolution_123 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_25, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_99 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_123,), kwargs = {})
#   %mul_324 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_123, %sigmoid_99), kwargs = {})
#   %convolution_124 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_324, %arg108_1, %arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_100 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_124,), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_323, %sigmoid_100), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_38 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_38(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 276480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 240
    x2 = (xindex // 34560)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + (240*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2b/c2brfph3qurxkyb5azqvxhne6tzzb5ek6i33afuyrv4z7vi3cyps.py
# Topologically Sorted Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_244 => add_165, mul_327, mul_328, sub_75
# Graph fragment:
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_125, %unsqueeze_601), kwargs = {})
#   %mul_327 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_75, %unsqueeze_603), kwargs = {})
#   %mul_328 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_327, %unsqueeze_605), kwargs = {})
#   %add_165 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_328, %unsqueeze_607), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 92160
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


# kernel path: /tmp/torchinductor_sahanp/46/c465o3kfmgtiot62dtat44juraihtz5hnllvwfyylpca56pjexsy.py
# Topologically Sorted Source Nodes: [x_246, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_246 => add_167, mul_330, mul_331, sub_76
#   x_247 => mul_332, sigmoid_101
# Graph fragment:
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_126, %unsqueeze_609), kwargs = {})
#   %mul_330 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %unsqueeze_611), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_330, %unsqueeze_613), kwargs = {})
#   %add_167 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_331, %unsqueeze_615), kwargs = {})
#   %sigmoid_101 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_167,), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_167, %sigmoid_101), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 552960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 480
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ap/capo6xiq5vpdcn4hgker2jsqdyuzzorfhpu6tiulfey25fna5jbv.py
# Topologically Sorted Source Nodes: [x_249], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_249 => add_169, mul_334, mul_335, sub_77
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_127, %unsqueeze_617), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_334, %unsqueeze_621), kwargs = {})
#   %add_169 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_335, %unsqueeze_623), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 552960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 480
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


# kernel path: /tmp/torchinductor_sahanp/un/cunswnqfqe4br642auvyyn46rqico2d4gthkmadw2vjrwtps344o.py
# Topologically Sorted Source Nodes: [x_250, x_se_100], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_250 => mul_336, sigmoid_102
#   x_se_100 => mean_26
# Graph fragment:
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_169,), kwargs = {})
#   %mul_336 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_169, %sigmoid_102), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_336, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_42 = async_compile.triton('triton_red_fused_mean_silu_42', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_42(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (34560*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ik/cikp3gscvkcxwmgeirwbqwnmyemdy3w3ttjz7cb34udll2naegdw.py
# Topologically Sorted Source Nodes: [x_250, x_se_100], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_250 => mul_336, sigmoid_102
#   x_se_100 => mean_26
# Graph fragment:
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_169,), kwargs = {})
#   %mul_336 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_169, %sigmoid_102), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_336, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_43 = async_compile.triton('triton_per_fused_mean_silu_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_43(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 480
    x1 = (xindex // 480)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (960*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 144.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/t7/ct7xraccv64xo3vvt3dyqxwgnsm4e3bxzpzxuh27out2cpehidmj.py
# Topologically Sorted Source Nodes: [x_250, x_se_100, x_se_101, x_se_102], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_250 => mul_336, sigmoid_102
#   x_se_100 => mean_26
#   x_se_101 => convolution_128
#   x_se_102 => mul_337, sigmoid_103
# Graph fragment:
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_169,), kwargs = {})
#   %mul_336 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_169, %sigmoid_102), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_336, [2, 3], True), kwargs = {})
#   %convolution_128 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_26, %arg125_1, %arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_103 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_128,), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_128, %sigmoid_103), kwargs = {})
triton_poi_fused_convolution_mean_silu_44 = async_compile.triton('triton_poi_fused_convolution_mean_silu_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_44(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 20
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/du/cduibkfyqrczeqs7f47e3hk2i6el6mxiq5svb2mz7zqfyo5weihz.py
# Topologically Sorted Source Nodes: [x_250, x_se_100, x_se_101, x_se_102, x_se_103, sigmoid_25, x_251], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_25 => sigmoid_104
#   x_250 => mul_336, sigmoid_102
#   x_251 => mul_338
#   x_se_100 => mean_26
#   x_se_101 => convolution_128
#   x_se_102 => mul_337, sigmoid_103
#   x_se_103 => convolution_129
# Graph fragment:
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_169,), kwargs = {})
#   %mul_336 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_169, %sigmoid_102), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_336, [2, 3], True), kwargs = {})
#   %convolution_128 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_26, %arg125_1, %arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_103 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_128,), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_128, %sigmoid_103), kwargs = {})
#   %convolution_129 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_337, %arg127_1, %arg128_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_104 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_129,), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_336, %sigmoid_104), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_45 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_45(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 552960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 480
    x2 = (xindex // 69120)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (480*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qx/cqxc3ljnx2blajyjywrolen44qov7gvhfh5efqpxnu22jkpkhosa.py
# Topologically Sorted Source Nodes: [x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_253 => add_171, mul_340, mul_341, sub_78
#   x_254 => add_172
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_130, %unsqueeze_625), kwargs = {})
#   %mul_340 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_341 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_340, %unsqueeze_629), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_341, %unsqueeze_631), kwargs = {})
#   %add_172 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_171, %add_165), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 92160
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


# kernel path: /tmp/torchinductor_sahanp/wd/cwd74ubfg6ejd757ugmosyzkdecyk53bscq5o6whr4xxhvg32bxf.py
# Topologically Sorted Source Nodes: [x_263, x_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_263 => add_178, mul_353, mul_354, sub_81
#   x_264 => add_179
# Graph fragment:
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_135, %unsqueeze_649), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_651), kwargs = {})
#   %mul_354 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_353, %unsqueeze_653), kwargs = {})
#   %add_178 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_354, %unsqueeze_655), kwargs = {})
#   %add_179 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_178, %add_172), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 92160
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sg/csgvqmpuwbbyzgkjzngbxs5buac7a6v65kchwe5l6y6x23qtg6ny.py
# Topologically Sorted Source Nodes: [x_283], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_283 => add_192, mul_379, mul_380, sub_87
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_145, %unsqueeze_697), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_379, %unsqueeze_701), kwargs = {})
#   %add_192 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_380, %unsqueeze_703), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129024
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


# kernel path: /tmp/torchinductor_sahanp/ih/cih7rsywig3umvvfle3xn32lbscuuzvgv3e7pir42bkgvwwghoqx.py
# Topologically Sorted Source Nodes: [x_285, x_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_285 => add_194, mul_382, mul_383, sub_88
#   x_286 => mul_384, sigmoid_117
# Graph fragment:
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_146, %unsqueeze_705), kwargs = {})
#   %mul_382 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_707), kwargs = {})
#   %mul_383 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_382, %unsqueeze_709), kwargs = {})
#   %add_194 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_383, %unsqueeze_711), kwargs = {})
#   %sigmoid_117 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_194,), kwargs = {})
#   %mul_384 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_194, %sigmoid_117), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 774144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 672
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hu/chuqljzyl4ndg5jgsm4f5dzmkwifea6ctljaozn7gb7rdtcloy5t.py
# Topologically Sorted Source Nodes: [x_288], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_288 => add_196, mul_386, mul_387, sub_89
# Graph fragment:
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_147, %unsqueeze_713), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %unsqueeze_715), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_386, %unsqueeze_717), kwargs = {})
#   %add_196 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_387, %unsqueeze_719), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 774144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 672
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


# kernel path: /tmp/torchinductor_sahanp/lo/clohzherfc4knvaognpfbd3pefges3aewg24lha6inuj2w5vzlwq.py
# Topologically Sorted Source Nodes: [x_289, x_se_116], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_289 => mul_388, sigmoid_118
#   x_se_116 => mean_30
# Graph fragment:
#   %sigmoid_118 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_196,), kwargs = {})
#   %mul_388 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_196, %sigmoid_118), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_388, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_51 = async_compile.triton('triton_red_fused_mean_silu_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_51(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (48384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n2/cn2mmp62qtr44jigqpja7bdphxsk5zek63hzme3cdu4i4abaebuw.py
# Topologically Sorted Source Nodes: [x_289, x_se_116], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_289 => mul_388, sigmoid_118
#   x_se_116 => mean_30
# Graph fragment:
#   %sigmoid_118 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_196,), kwargs = {})
#   %mul_388 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_196, %sigmoid_118), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_388, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_52 = async_compile.triton('triton_per_fused_mean_silu_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_52(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (1344*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 144.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/r2/cr2v7yepjcdx6zrvwpy2gdqwpmxltvcdtylfqtlxm7d7to4tjmop.py
# Topologically Sorted Source Nodes: [x_289, x_se_116, x_se_117, x_se_118], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_289 => mul_388, sigmoid_118
#   x_se_116 => mean_30
#   x_se_117 => convolution_148
#   x_se_118 => mul_389, sigmoid_119
# Graph fragment:
#   %sigmoid_118 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_196,), kwargs = {})
#   %mul_388 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_196, %sigmoid_118), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_388, [2, 3], True), kwargs = {})
#   %convolution_148 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_30, %arg201_1, %arg202_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_119 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_148,), kwargs = {})
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_148, %sigmoid_119), kwargs = {})
triton_poi_fused_convolution_mean_silu_53 = async_compile.triton('triton_poi_fused_convolution_mean_silu_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_53(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 28
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/a5/ca5gel7jbrgdvsyd2nzyniumuoc5efonz2wvorefc4w54coar5fi.py
# Topologically Sorted Source Nodes: [x_289, x_se_116, x_se_117, x_se_118, x_se_119, sigmoid_29, x_290], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_29 => sigmoid_120
#   x_289 => mul_388, sigmoid_118
#   x_290 => mul_390
#   x_se_116 => mean_30
#   x_se_117 => convolution_148
#   x_se_118 => mul_389, sigmoid_119
#   x_se_119 => convolution_149
# Graph fragment:
#   %sigmoid_118 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_196,), kwargs = {})
#   %mul_388 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_196, %sigmoid_118), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_388, [2, 3], True), kwargs = {})
#   %convolution_148 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_30, %arg201_1, %arg202_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_119 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_148,), kwargs = {})
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_148, %sigmoid_119), kwargs = {})
#   %convolution_149 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_389, %arg203_1, %arg204_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_120 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_149,), kwargs = {})
#   %mul_390 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_388, %sigmoid_120), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_54 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_54', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_54(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 774144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 96768)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (672*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pw/cpwolfgz5zpe7gkccxgatbzn5za7a57qfm6zze4uohmtk5czu52t.py
# Topologically Sorted Source Nodes: [x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_292 => add_198, mul_392, mul_393, sub_90
#   x_293 => add_199
# Graph fragment:
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_150, %unsqueeze_721), kwargs = {})
#   %mul_392 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_723), kwargs = {})
#   %mul_393 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_392, %unsqueeze_725), kwargs = {})
#   %add_198 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_393, %unsqueeze_727), kwargs = {})
#   %add_199 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_198, %add_192), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_55(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
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


# kernel path: /tmp/torchinductor_sahanp/rx/crxxrig3asbzgptmvbcyuvd63zgxdi4xjfpi7pyshdg3racflojc.py
# Topologically Sorted Source Nodes: [x_318], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_318 => add_217, mul_425, mul_426, sub_98
# Graph fragment:
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_162, %unsqueeze_785), kwargs = {})
#   %mul_425 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_98, %unsqueeze_787), kwargs = {})
#   %mul_426 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_425, %unsqueeze_789), kwargs = {})
#   %add_217 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_426, %unsqueeze_791), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_56', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_56(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 193536
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


# kernel path: /tmp/torchinductor_sahanp/b2/cb2vcbd7aijavy4pk4e2vf3howpmx43bjmbxbbsve5k2ksrtcqgb.py
# Topologically Sorted Source Nodes: [x_319, x_se_128], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_319 => mul_427, sigmoid_130
#   x_se_128 => mean_33
# Graph fragment:
#   %sigmoid_130 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_217,), kwargs = {})
#   %mul_427 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_217, %sigmoid_130), kwargs = {})
#   %mean_33 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_427, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_57 = async_compile.triton('triton_per_fused_mean_silu_57', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_57(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 36
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
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (24192*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 36.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ny/cny6xum63h3ldeef47jhvlnouzp6e575xb5oknrvx2hrckccoogn.py
# Topologically Sorted Source Nodes: [x_319, x_se_128, x_se_129, x_se_130, x_se_131, sigmoid_32, x_320], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_32 => sigmoid_132
#   x_319 => mul_427, sigmoid_130
#   x_320 => mul_429
#   x_se_128 => mean_33
#   x_se_129 => convolution_163
#   x_se_130 => mul_428, sigmoid_131
#   x_se_131 => convolution_164
# Graph fragment:
#   %sigmoid_130 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_217,), kwargs = {})
#   %mul_427 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_217, %sigmoid_130), kwargs = {})
#   %mean_33 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_427, [2, 3], True), kwargs = {})
#   %convolution_163 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_33, %arg258_1, %arg259_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_131 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_163,), kwargs = {})
#   %mul_428 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_163, %sigmoid_131), kwargs = {})
#   %convolution_164 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_428, %arg260_1, %arg261_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_132 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_164,), kwargs = {})
#   %mul_429 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_427, %sigmoid_132), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_58 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_58', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_58(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 193536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 24192)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ip/cip4leq7hrsa2cg6a2uoerstglz6t7auvsixsbyqg65yv6iw3pcq.py
# Topologically Sorted Source Nodes: [x_322], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_322 => add_219, mul_431, mul_432, sub_99
# Graph fragment:
#   %sub_99 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_165, %unsqueeze_793), kwargs = {})
#   %mul_431 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_99, %unsqueeze_795), kwargs = {})
#   %mul_432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_431, %unsqueeze_797), kwargs = {})
#   %add_219 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_432, %unsqueeze_799), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_59 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_59', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 55296
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


# kernel path: /tmp/torchinductor_sahanp/ky/ckycqlft2zyzl5vi3vb5shibsnuwmdzoywauzdhkupo5mtl3znur.py
# Topologically Sorted Source Nodes: [x_324, x_325], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_324 => add_221, mul_434, mul_435, sub_100
#   x_325 => mul_436, sigmoid_133
# Graph fragment:
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_166, %unsqueeze_801), kwargs = {})
#   %mul_434 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_803), kwargs = {})
#   %mul_435 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_434, %unsqueeze_805), kwargs = {})
#   %add_221 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_435, %unsqueeze_807), kwargs = {})
#   %sigmoid_133 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_221,), kwargs = {})
#   %mul_436 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_221, %sigmoid_133), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_60 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_60', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_60', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_60(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 331776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1152
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3s/c3sttr5moohaywc66k77ctqpodtnlhuuotnh2wk2rzndzodi3w3d.py
# Topologically Sorted Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_327 => add_223, mul_438, mul_439, sub_101
# Graph fragment:
#   %sub_101 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_167, %unsqueeze_809), kwargs = {})
#   %mul_438 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_101, %unsqueeze_811), kwargs = {})
#   %mul_439 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_438, %unsqueeze_813), kwargs = {})
#   %add_223 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_439, %unsqueeze_815), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_61 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_61', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_61', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_61(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 331776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1152
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


# kernel path: /tmp/torchinductor_sahanp/2g/c2ga7bgypvfvl3mkmqrz36obwsswls3ncedzehmjourtxncccckz.py
# Topologically Sorted Source Nodes: [x_328, x_se_132], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_328 => mul_440, sigmoid_134
#   x_se_132 => mean_34
# Graph fragment:
#   %sigmoid_134 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_223,), kwargs = {})
#   %mul_440 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_223, %sigmoid_134), kwargs = {})
#   %mean_34 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_440, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_62 = async_compile.triton('triton_per_fused_mean_silu_62', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_62(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1152
    x1 = (xindex // 1152)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r2) + (41472*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 36.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cr/ccrzlzdtaot33boojfyh6s2imulyfjcjtckuet4yj3gllord5hs7.py
# Topologically Sorted Source Nodes: [x_328, x_se_132, x_se_133, x_se_134], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_328 => mul_440, sigmoid_134
#   x_se_132 => mean_34
#   x_se_133 => convolution_168
#   x_se_134 => mul_441, sigmoid_135
# Graph fragment:
#   %sigmoid_134 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_223,), kwargs = {})
#   %mul_440 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_223, %sigmoid_134), kwargs = {})
#   %mean_34 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_440, [2, 3], True), kwargs = {})
#   %convolution_168 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_34, %arg277_1, %arg278_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_135 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_168,), kwargs = {})
#   %mul_441 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_168, %sigmoid_135), kwargs = {})
triton_poi_fused_convolution_mean_silu_63 = async_compile.triton('triton_poi_fused_convolution_mean_silu_63', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_63', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_63(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/in/cin7yswc3surcqlwi22p342fn35z23oj4zfultg5hcmqzubyiyun.py
# Topologically Sorted Source Nodes: [x_328, x_se_132, x_se_133, x_se_134, x_se_135, sigmoid_33, x_329], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_33 => sigmoid_136
#   x_328 => mul_440, sigmoid_134
#   x_329 => mul_442
#   x_se_132 => mean_34
#   x_se_133 => convolution_168
#   x_se_134 => mul_441, sigmoid_135
#   x_se_135 => convolution_169
# Graph fragment:
#   %sigmoid_134 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_223,), kwargs = {})
#   %mul_440 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_223, %sigmoid_134), kwargs = {})
#   %mean_34 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_440, [2, 3], True), kwargs = {})
#   %convolution_168 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_34, %arg277_1, %arg278_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_135 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_168,), kwargs = {})
#   %mul_441 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_168, %sigmoid_135), kwargs = {})
#   %convolution_169 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_441, %arg279_1, %arg280_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_136 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_169,), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_440, %sigmoid_136), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_64 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_64', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_64', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_64(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 331776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 1152
    x2 = (xindex // 41472)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (1152*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kz/ckz5p56antxavoas556vxvh2ys5h6ceh5at25u6omdbxvhd4samm.py
# Topologically Sorted Source Nodes: [x_331, x_332], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_331 => add_225, mul_444, mul_445, sub_102
#   x_332 => add_226
# Graph fragment:
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_170, %unsqueeze_817), kwargs = {})
#   %mul_444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %unsqueeze_819), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_444, %unsqueeze_821), kwargs = {})
#   %add_225 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_445, %unsqueeze_823), kwargs = {})
#   %add_226 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_225, %add_219), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_65 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_65', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_65(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 55296
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


# kernel path: /tmp/torchinductor_sahanp/as/cass7hqhuett433dkunp7scjuybluwbanve7oow2t22o5zxcovxn.py
# Topologically Sorted Source Nodes: [x_371], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_371 => add_253, mul_496, mul_497, sub_114
# Graph fragment:
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_190, %unsqueeze_913), kwargs = {})
#   %mul_496 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_915), kwargs = {})
#   %mul_497 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_496, %unsqueeze_917), kwargs = {})
#   %add_253 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_497, %unsqueeze_919), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_66 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_66', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_66', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_66(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 92160
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


# kernel path: /tmp/torchinductor_sahanp/ok/cok6gifbosv3zmdqwob4dcidw7ftqbvbbuxnaaxtpohgjfm6qu3h.py
# Topologically Sorted Source Nodes: [x_373], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_373 => add_255, mul_499, mul_500, sub_115
# Graph fragment:
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_191, %unsqueeze_921), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %unsqueeze_923), kwargs = {})
#   %mul_500 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_499, %unsqueeze_925), kwargs = {})
#   %add_255 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_500, %unsqueeze_927), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_67 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_67', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_67', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_67(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 368640
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/r3/cr3zqluh6zd75r3l6534jwn5nttcqhkiptqbzxforvqd4rut4ydz.py
# Topologically Sorted Source Nodes: [x_374, x_375], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_374 => mul_501, sigmoid_153
#   x_375 => mean_39
# Graph fragment:
#   %sigmoid_153 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_255,), kwargs = {})
#   %mul_501 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_255, %sigmoid_153), kwargs = {})
#   %mean_39 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_501, [-1, -2], True), kwargs = {})
triton_per_fused_mean_silu_68 = async_compile.triton('triton_per_fused_mean_silu_68', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_68(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 36
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
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (46080*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 36.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 192, 192), (110592, 36864, 192, 1))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (32, ), (1, ))
    assert_size_stride(arg9_1, (32, ), (1, ))
    assert_size_stride(arg10_1, (32, ), (1, ))
    assert_size_stride(arg11_1, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg12_1, (8, ), (1, ))
    assert_size_stride(arg13_1, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg14_1, (32, ), (1, ))
    assert_size_stride(arg15_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg16_1, (16, ), (1, ))
    assert_size_stride(arg17_1, (16, ), (1, ))
    assert_size_stride(arg18_1, (16, ), (1, ))
    assert_size_stride(arg19_1, (16, ), (1, ))
    assert_size_stride(arg20_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg21_1, (96, ), (1, ))
    assert_size_stride(arg22_1, (96, ), (1, ))
    assert_size_stride(arg23_1, (96, ), (1, ))
    assert_size_stride(arg24_1, (96, ), (1, ))
    assert_size_stride(arg25_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg26_1, (96, ), (1, ))
    assert_size_stride(arg27_1, (96, ), (1, ))
    assert_size_stride(arg28_1, (96, ), (1, ))
    assert_size_stride(arg29_1, (96, ), (1, ))
    assert_size_stride(arg30_1, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg31_1, (4, ), (1, ))
    assert_size_stride(arg32_1, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(arg33_1, (96, ), (1, ))
    assert_size_stride(arg34_1, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg35_1, (24, ), (1, ))
    assert_size_stride(arg36_1, (24, ), (1, ))
    assert_size_stride(arg37_1, (24, ), (1, ))
    assert_size_stride(arg38_1, (24, ), (1, ))
    assert_size_stride(arg39_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg40_1, (144, ), (1, ))
    assert_size_stride(arg41_1, (144, ), (1, ))
    assert_size_stride(arg42_1, (144, ), (1, ))
    assert_size_stride(arg43_1, (144, ), (1, ))
    assert_size_stride(arg44_1, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg45_1, (144, ), (1, ))
    assert_size_stride(arg46_1, (144, ), (1, ))
    assert_size_stride(arg47_1, (144, ), (1, ))
    assert_size_stride(arg48_1, (144, ), (1, ))
    assert_size_stride(arg49_1, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg50_1, (6, ), (1, ))
    assert_size_stride(arg51_1, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg52_1, (144, ), (1, ))
    assert_size_stride(arg53_1, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg54_1, (24, ), (1, ))
    assert_size_stride(arg55_1, (24, ), (1, ))
    assert_size_stride(arg56_1, (24, ), (1, ))
    assert_size_stride(arg57_1, (24, ), (1, ))
    assert_size_stride(arg58_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg59_1, (144, ), (1, ))
    assert_size_stride(arg60_1, (144, ), (1, ))
    assert_size_stride(arg61_1, (144, ), (1, ))
    assert_size_stride(arg62_1, (144, ), (1, ))
    assert_size_stride(arg63_1, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg64_1, (144, ), (1, ))
    assert_size_stride(arg65_1, (144, ), (1, ))
    assert_size_stride(arg66_1, (144, ), (1, ))
    assert_size_stride(arg67_1, (144, ), (1, ))
    assert_size_stride(arg68_1, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg69_1, (6, ), (1, ))
    assert_size_stride(arg70_1, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg71_1, (144, ), (1, ))
    assert_size_stride(arg72_1, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg73_1, (40, ), (1, ))
    assert_size_stride(arg74_1, (40, ), (1, ))
    assert_size_stride(arg75_1, (40, ), (1, ))
    assert_size_stride(arg76_1, (40, ), (1, ))
    assert_size_stride(arg77_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg78_1, (240, ), (1, ))
    assert_size_stride(arg79_1, (240, ), (1, ))
    assert_size_stride(arg80_1, (240, ), (1, ))
    assert_size_stride(arg81_1, (240, ), (1, ))
    assert_size_stride(arg82_1, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg83_1, (240, ), (1, ))
    assert_size_stride(arg84_1, (240, ), (1, ))
    assert_size_stride(arg85_1, (240, ), (1, ))
    assert_size_stride(arg86_1, (240, ), (1, ))
    assert_size_stride(arg87_1, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg88_1, (10, ), (1, ))
    assert_size_stride(arg89_1, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(arg90_1, (240, ), (1, ))
    assert_size_stride(arg91_1, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg92_1, (40, ), (1, ))
    assert_size_stride(arg93_1, (40, ), (1, ))
    assert_size_stride(arg94_1, (40, ), (1, ))
    assert_size_stride(arg95_1, (40, ), (1, ))
    assert_size_stride(arg96_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg97_1, (240, ), (1, ))
    assert_size_stride(arg98_1, (240, ), (1, ))
    assert_size_stride(arg99_1, (240, ), (1, ))
    assert_size_stride(arg100_1, (240, ), (1, ))
    assert_size_stride(arg101_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg102_1, (240, ), (1, ))
    assert_size_stride(arg103_1, (240, ), (1, ))
    assert_size_stride(arg104_1, (240, ), (1, ))
    assert_size_stride(arg105_1, (240, ), (1, ))
    assert_size_stride(arg106_1, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg107_1, (10, ), (1, ))
    assert_size_stride(arg108_1, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(arg109_1, (240, ), (1, ))
    assert_size_stride(arg110_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg111_1, (80, ), (1, ))
    assert_size_stride(arg112_1, (80, ), (1, ))
    assert_size_stride(arg113_1, (80, ), (1, ))
    assert_size_stride(arg114_1, (80, ), (1, ))
    assert_size_stride(arg115_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg116_1, (480, ), (1, ))
    assert_size_stride(arg117_1, (480, ), (1, ))
    assert_size_stride(arg118_1, (480, ), (1, ))
    assert_size_stride(arg119_1, (480, ), (1, ))
    assert_size_stride(arg120_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg121_1, (480, ), (1, ))
    assert_size_stride(arg122_1, (480, ), (1, ))
    assert_size_stride(arg123_1, (480, ), (1, ))
    assert_size_stride(arg124_1, (480, ), (1, ))
    assert_size_stride(arg125_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg126_1, (20, ), (1, ))
    assert_size_stride(arg127_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg128_1, (480, ), (1, ))
    assert_size_stride(arg129_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg130_1, (80, ), (1, ))
    assert_size_stride(arg131_1, (80, ), (1, ))
    assert_size_stride(arg132_1, (80, ), (1, ))
    assert_size_stride(arg133_1, (80, ), (1, ))
    assert_size_stride(arg134_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg135_1, (480, ), (1, ))
    assert_size_stride(arg136_1, (480, ), (1, ))
    assert_size_stride(arg137_1, (480, ), (1, ))
    assert_size_stride(arg138_1, (480, ), (1, ))
    assert_size_stride(arg139_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg140_1, (480, ), (1, ))
    assert_size_stride(arg141_1, (480, ), (1, ))
    assert_size_stride(arg142_1, (480, ), (1, ))
    assert_size_stride(arg143_1, (480, ), (1, ))
    assert_size_stride(arg144_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg145_1, (20, ), (1, ))
    assert_size_stride(arg146_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg147_1, (480, ), (1, ))
    assert_size_stride(arg148_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg149_1, (80, ), (1, ))
    assert_size_stride(arg150_1, (80, ), (1, ))
    assert_size_stride(arg151_1, (80, ), (1, ))
    assert_size_stride(arg152_1, (80, ), (1, ))
    assert_size_stride(arg153_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg154_1, (480, ), (1, ))
    assert_size_stride(arg155_1, (480, ), (1, ))
    assert_size_stride(arg156_1, (480, ), (1, ))
    assert_size_stride(arg157_1, (480, ), (1, ))
    assert_size_stride(arg158_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg159_1, (480, ), (1, ))
    assert_size_stride(arg160_1, (480, ), (1, ))
    assert_size_stride(arg161_1, (480, ), (1, ))
    assert_size_stride(arg162_1, (480, ), (1, ))
    assert_size_stride(arg163_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg164_1, (20, ), (1, ))
    assert_size_stride(arg165_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg166_1, (480, ), (1, ))
    assert_size_stride(arg167_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg168_1, (80, ), (1, ))
    assert_size_stride(arg169_1, (80, ), (1, ))
    assert_size_stride(arg170_1, (80, ), (1, ))
    assert_size_stride(arg171_1, (80, ), (1, ))
    assert_size_stride(arg172_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg173_1, (480, ), (1, ))
    assert_size_stride(arg174_1, (480, ), (1, ))
    assert_size_stride(arg175_1, (480, ), (1, ))
    assert_size_stride(arg176_1, (480, ), (1, ))
    assert_size_stride(arg177_1, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg178_1, (480, ), (1, ))
    assert_size_stride(arg179_1, (480, ), (1, ))
    assert_size_stride(arg180_1, (480, ), (1, ))
    assert_size_stride(arg181_1, (480, ), (1, ))
    assert_size_stride(arg182_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg183_1, (20, ), (1, ))
    assert_size_stride(arg184_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg185_1, (480, ), (1, ))
    assert_size_stride(arg186_1, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg187_1, (112, ), (1, ))
    assert_size_stride(arg188_1, (112, ), (1, ))
    assert_size_stride(arg189_1, (112, ), (1, ))
    assert_size_stride(arg190_1, (112, ), (1, ))
    assert_size_stride(arg191_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg192_1, (672, ), (1, ))
    assert_size_stride(arg193_1, (672, ), (1, ))
    assert_size_stride(arg194_1, (672, ), (1, ))
    assert_size_stride(arg195_1, (672, ), (1, ))
    assert_size_stride(arg196_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg197_1, (672, ), (1, ))
    assert_size_stride(arg198_1, (672, ), (1, ))
    assert_size_stride(arg199_1, (672, ), (1, ))
    assert_size_stride(arg200_1, (672, ), (1, ))
    assert_size_stride(arg201_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg202_1, (28, ), (1, ))
    assert_size_stride(arg203_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg204_1, (672, ), (1, ))
    assert_size_stride(arg205_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg206_1, (112, ), (1, ))
    assert_size_stride(arg207_1, (112, ), (1, ))
    assert_size_stride(arg208_1, (112, ), (1, ))
    assert_size_stride(arg209_1, (112, ), (1, ))
    assert_size_stride(arg210_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg211_1, (672, ), (1, ))
    assert_size_stride(arg212_1, (672, ), (1, ))
    assert_size_stride(arg213_1, (672, ), (1, ))
    assert_size_stride(arg214_1, (672, ), (1, ))
    assert_size_stride(arg215_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg216_1, (672, ), (1, ))
    assert_size_stride(arg217_1, (672, ), (1, ))
    assert_size_stride(arg218_1, (672, ), (1, ))
    assert_size_stride(arg219_1, (672, ), (1, ))
    assert_size_stride(arg220_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg221_1, (28, ), (1, ))
    assert_size_stride(arg222_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg223_1, (672, ), (1, ))
    assert_size_stride(arg224_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg225_1, (112, ), (1, ))
    assert_size_stride(arg226_1, (112, ), (1, ))
    assert_size_stride(arg227_1, (112, ), (1, ))
    assert_size_stride(arg228_1, (112, ), (1, ))
    assert_size_stride(arg229_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg230_1, (672, ), (1, ))
    assert_size_stride(arg231_1, (672, ), (1, ))
    assert_size_stride(arg232_1, (672, ), (1, ))
    assert_size_stride(arg233_1, (672, ), (1, ))
    assert_size_stride(arg234_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg235_1, (672, ), (1, ))
    assert_size_stride(arg236_1, (672, ), (1, ))
    assert_size_stride(arg237_1, (672, ), (1, ))
    assert_size_stride(arg238_1, (672, ), (1, ))
    assert_size_stride(arg239_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg240_1, (28, ), (1, ))
    assert_size_stride(arg241_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg242_1, (672, ), (1, ))
    assert_size_stride(arg243_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg244_1, (112, ), (1, ))
    assert_size_stride(arg245_1, (112, ), (1, ))
    assert_size_stride(arg246_1, (112, ), (1, ))
    assert_size_stride(arg247_1, (112, ), (1, ))
    assert_size_stride(arg248_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg249_1, (672, ), (1, ))
    assert_size_stride(arg250_1, (672, ), (1, ))
    assert_size_stride(arg251_1, (672, ), (1, ))
    assert_size_stride(arg252_1, (672, ), (1, ))
    assert_size_stride(arg253_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg254_1, (672, ), (1, ))
    assert_size_stride(arg255_1, (672, ), (1, ))
    assert_size_stride(arg256_1, (672, ), (1, ))
    assert_size_stride(arg257_1, (672, ), (1, ))
    assert_size_stride(arg258_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg259_1, (28, ), (1, ))
    assert_size_stride(arg260_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg261_1, (672, ), (1, ))
    assert_size_stride(arg262_1, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg263_1, (192, ), (1, ))
    assert_size_stride(arg264_1, (192, ), (1, ))
    assert_size_stride(arg265_1, (192, ), (1, ))
    assert_size_stride(arg266_1, (192, ), (1, ))
    assert_size_stride(arg267_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg268_1, (1152, ), (1, ))
    assert_size_stride(arg269_1, (1152, ), (1, ))
    assert_size_stride(arg270_1, (1152, ), (1, ))
    assert_size_stride(arg271_1, (1152, ), (1, ))
    assert_size_stride(arg272_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg273_1, (1152, ), (1, ))
    assert_size_stride(arg274_1, (1152, ), (1, ))
    assert_size_stride(arg275_1, (1152, ), (1, ))
    assert_size_stride(arg276_1, (1152, ), (1, ))
    assert_size_stride(arg277_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg278_1, (48, ), (1, ))
    assert_size_stride(arg279_1, (1152, 48, 1, 1), (48, 1, 1, 1))
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
    assert_size_stride(arg296_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg297_1, (48, ), (1, ))
    assert_size_stride(arg298_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg299_1, (1152, ), (1, ))
    assert_size_stride(arg300_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg301_1, (192, ), (1, ))
    assert_size_stride(arg302_1, (192, ), (1, ))
    assert_size_stride(arg303_1, (192, ), (1, ))
    assert_size_stride(arg304_1, (192, ), (1, ))
    assert_size_stride(arg305_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg306_1, (1152, ), (1, ))
    assert_size_stride(arg307_1, (1152, ), (1, ))
    assert_size_stride(arg308_1, (1152, ), (1, ))
    assert_size_stride(arg309_1, (1152, ), (1, ))
    assert_size_stride(arg310_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg311_1, (1152, ), (1, ))
    assert_size_stride(arg312_1, (1152, ), (1, ))
    assert_size_stride(arg313_1, (1152, ), (1, ))
    assert_size_stride(arg314_1, (1152, ), (1, ))
    assert_size_stride(arg315_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg316_1, (48, ), (1, ))
    assert_size_stride(arg317_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg318_1, (1152, ), (1, ))
    assert_size_stride(arg319_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg320_1, (192, ), (1, ))
    assert_size_stride(arg321_1, (192, ), (1, ))
    assert_size_stride(arg322_1, (192, ), (1, ))
    assert_size_stride(arg323_1, (192, ), (1, ))
    assert_size_stride(arg324_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg325_1, (1152, ), (1, ))
    assert_size_stride(arg326_1, (1152, ), (1, ))
    assert_size_stride(arg327_1, (1152, ), (1, ))
    assert_size_stride(arg328_1, (1152, ), (1, ))
    assert_size_stride(arg329_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg330_1, (1152, ), (1, ))
    assert_size_stride(arg331_1, (1152, ), (1, ))
    assert_size_stride(arg332_1, (1152, ), (1, ))
    assert_size_stride(arg333_1, (1152, ), (1, ))
    assert_size_stride(arg334_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg335_1, (48, ), (1, ))
    assert_size_stride(arg336_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg337_1, (1152, ), (1, ))
    assert_size_stride(arg338_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg339_1, (192, ), (1, ))
    assert_size_stride(arg340_1, (192, ), (1, ))
    assert_size_stride(arg341_1, (192, ), (1, ))
    assert_size_stride(arg342_1, (192, ), (1, ))
    assert_size_stride(arg343_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg344_1, (1152, ), (1, ))
    assert_size_stride(arg345_1, (1152, ), (1, ))
    assert_size_stride(arg346_1, (1152, ), (1, ))
    assert_size_stride(arg347_1, (1152, ), (1, ))
    assert_size_stride(arg348_1, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg349_1, (1152, ), (1, ))
    assert_size_stride(arg350_1, (1152, ), (1, ))
    assert_size_stride(arg351_1, (1152, ), (1, ))
    assert_size_stride(arg352_1, (1152, ), (1, ))
    assert_size_stride(arg353_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg354_1, (48, ), (1, ))
    assert_size_stride(arg355_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg356_1, (1152, ), (1, ))
    assert_size_stride(arg357_1, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg358_1, (320, ), (1, ))
    assert_size_stride(arg359_1, (320, ), (1, ))
    assert_size_stride(arg360_1, (320, ), (1, ))
    assert_size_stride(arg361_1, (320, ), (1, ))
    assert_size_stride(arg362_1, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg363_1, (1280, ), (1, ))
    assert_size_stride(arg364_1, (1280, ), (1, ))
    assert_size_stride(arg365_1, (1280, ), (1, ))
    assert_size_stride(arg366_1, (1280, ), (1, ))
    assert_size_stride(arg367_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg368_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 192, 192), (110592, 1, 576, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_189], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 36864, grid=grid(24, 36864), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_189], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 96, 96), (294912, 1, 3072, 32))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((8, 32, 96, 96), (294912, 1, 3072, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_190, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, buf4, 2359296, grid=grid(2359296), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf3
        # Topologically Sorted Source Nodes: [x_191, x_192], Original ATen: [aten.silu, aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf5, (8, 32, 96, 96), (294912, 1, 3072, 32))
        del arg6_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_193], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf7 = empty_strided_cuda((8, 32, 1, 1, 72), (2304, 1, 18432, 18432, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_194, x_se_76], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_4.run(buf6, buf7, 18432, 128, grid=grid(18432), stream=stream0)
        buf9 = empty_strided_cuda((8, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_194, x_se_76], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_5.run(buf7, buf9, 256, 72, grid=grid(256), stream=stream0)
        del buf7
        # Topologically Sorted Source Nodes: [x_194, x_se_76, x_se_77], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg11_1
        del buf9
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_194, x_se_76, x_se_77, x_se_78], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_6.run(buf11, arg12_1, 64, grid=grid(64), stream=stream0)
        del arg12_1
        # Topologically Sorted Source Nodes: [x_194, x_se_76, x_se_77, x_se_78, x_se_79], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg13_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg13_1
        del buf11
        buf13 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_194, x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, x_195], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_7.run(buf13, buf12, arg14_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg14_1
        del buf12
        # Topologically Sorted Source Nodes: [x_194, x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, x_195, x_196], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf14 = extern_kernels.convolution(buf13, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 16, 96, 96), (147456, 1, 1536, 16))
        del arg15_1
        del buf13
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf15, arg16_1, arg17_1, arg18_1, arg19_1, 1179648, grid=grid(1179648), stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        del arg19_1
        # Topologically Sorted Source Nodes: [x_197, x_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg20_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 96, 96, 96), (884736, 1, 9216, 96))
        del arg20_1
        del buf15
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided_cuda((8, 96, 96, 96), (884736, 1, 9216, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_199, x_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_9.run(buf17, arg21_1, arg22_1, arg23_1, arg24_1, buf18, 7077888, grid=grid(7077888), stream=stream0)
        del arg21_1
        del arg22_1
        del arg23_1
        del arg24_1
        del buf17
        # Topologically Sorted Source Nodes: [x_200, x_201], Original ATen: [aten.silu, aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg25_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf19, (8, 96, 48, 48), (221184, 1, 4608, 96))
        del arg25_1
        del buf18
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_202], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf20, arg26_1, arg27_1, arg28_1, arg29_1, 1769472, grid=grid(1769472), stream=stream0)
        del arg26_1
        del arg27_1
        del arg28_1
        del arg29_1
        buf21 = empty_strided_cuda((8, 96, 1, 1, 18), (1728, 1, 13824, 13824, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_203, x_se_80], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_11.run(buf20, buf21, 13824, 128, grid=grid(13824), stream=stream0)
        buf23 = empty_strided_cuda((8, 96, 1, 1), (96, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_203, x_se_80], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_12.run(buf21, buf23, 768, 18, grid=grid(768), stream=stream0)
        del buf21
        # Topologically Sorted Source Nodes: [x_203, x_se_80, x_se_81], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 4, 1, 1), (4, 1, 1, 1))
        del arg30_1
        del buf23
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_203, x_se_80, x_se_81, x_se_82], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_13.run(buf25, arg31_1, 32, grid=grid(32), stream=stream0)
        del arg31_1
        # Topologically Sorted Source Nodes: [x_203, x_se_80, x_se_81, x_se_82, x_se_83], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg32_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 96, 1, 1), (96, 1, 1, 1))
        del arg32_1
        del buf25
        buf27 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_203, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, x_204], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_14.run(buf27, buf26, arg33_1, 1769472, grid=grid(1769472), stream=stream0)
        del arg33_1
        del buf26
        # Topologically Sorted Source Nodes: [x_203, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, x_204, x_205], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf28 = extern_kernels.convolution(buf27, arg34_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 24, 48, 48), (55296, 1, 1152, 24))
        del arg34_1
        del buf27
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_206], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf29, arg35_1, arg36_1, arg37_1, arg38_1, 442368, grid=grid(442368), stream=stream0)
        del arg35_1
        del arg36_1
        del arg37_1
        del arg38_1
        # Topologically Sorted Source Nodes: [x_207], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 144, 48, 48), (331776, 1, 6912, 144))
        del arg39_1
        buf31 = buf30; del buf30  # reuse
        buf32 = empty_strided_cuda((8, 144, 48, 48), (331776, 1, 6912, 144), torch.float32)
        # Topologically Sorted Source Nodes: [x_208, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf31, arg40_1, arg41_1, arg42_1, arg43_1, buf32, 2654208, grid=grid(2654208), stream=stream0)
        del arg40_1
        del arg41_1
        del arg42_1
        del arg43_1
        del buf31
        # Topologically Sorted Source Nodes: [x_209, x_210], Original ATen: [aten.silu, aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg44_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf33, (8, 144, 48, 48), (331776, 1, 6912, 144))
        del arg44_1
        del buf32
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_17.run(buf34, arg45_1, arg46_1, arg47_1, arg48_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg45_1
        del arg46_1
        del arg47_1
        del arg48_1
        buf35 = empty_strided_cuda((8, 144, 1, 1, 18), (2592, 1, 20736, 20736, 144), torch.float32)
        # Topologically Sorted Source Nodes: [x_212, x_se_84], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_18.run(buf34, buf35, 20736, 128, grid=grid(20736), stream=stream0)
        buf37 = empty_strided_cuda((8, 144, 1, 1), (144, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_212, x_se_84], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_19.run(buf35, buf37, 1152, 18, grid=grid(1152), stream=stream0)
        del buf35
        # Topologically Sorted Source Nodes: [x_212, x_se_84, x_se_85], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg49_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 6, 1, 1), (6, 1, 1, 1))
        del arg49_1
        del buf37
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_212, x_se_84, x_se_85, x_se_86], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_20.run(buf39, arg50_1, 48, grid=grid(48), stream=stream0)
        del arg50_1
        # Topologically Sorted Source Nodes: [x_212, x_se_84, x_se_85, x_se_86, x_se_87], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 144, 1, 1), (144, 1, 1, 1))
        del arg51_1
        del buf39
        buf41 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_212, x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, x_213], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_21.run(buf41, buf40, arg52_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg52_1
        # Topologically Sorted Source Nodes: [x_212, x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, x_213, x_214], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf42 = extern_kernels.convolution(buf41, arg53_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 24, 48, 48), (55296, 1, 1152, 24))
        del arg53_1
        buf43 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_215, x_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_22.run(buf43, buf42, arg54_1, arg55_1, arg56_1, arg57_1, 442368, grid=grid(442368), stream=stream0)
        del arg54_1
        del arg55_1
        del arg56_1
        del arg57_1
        del buf42
        # Topologically Sorted Source Nodes: [x_215, x_216, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg58_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 144, 48, 48), (331776, 1, 6912, 144))
        del arg58_1
        del buf43
        buf45 = buf44; del buf44  # reuse
        buf46 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_218, x_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_16.run(buf45, arg59_1, arg60_1, arg61_1, arg62_1, buf46, 2654208, grid=grid(2654208), stream=stream0)
        del arg59_1
        del arg60_1
        del arg61_1
        del arg62_1
        del buf45
        # Topologically Sorted Source Nodes: [x_219, x_220], Original ATen: [aten.silu, aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg63_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf47, (8, 144, 24, 24), (82944, 1, 3456, 144))
        del arg63_1
        del buf46
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_23.run(buf48, arg64_1, arg65_1, arg66_1, arg67_1, 663552, grid=grid(663552), stream=stream0)
        del arg64_1
        del arg65_1
        del arg66_1
        del arg67_1
        buf49 = empty_strided_cuda((8, 144, 1, 1, 5), (720, 1, 5760, 5760, 144), torch.float32)
        # Topologically Sorted Source Nodes: [x_222, x_se_88], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_24.run(buf48, buf49, 5760, 116, grid=grid(5760), stream=stream0)
        buf51 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_222, x_se_88], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_25.run(buf49, buf51, 1152, 5, grid=grid(1152), stream=stream0)
        del buf49
        # Topologically Sorted Source Nodes: [x_222, x_se_88, x_se_89], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 6, 1, 1), (6, 1, 1, 1))
        del arg68_1
        del buf51
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_222, x_se_88, x_se_89, x_se_90], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_20.run(buf53, arg69_1, 48, grid=grid(48), stream=stream0)
        del arg69_1
        # Topologically Sorted Source Nodes: [x_222, x_se_88, x_se_89, x_se_90, x_se_91], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg70_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 144, 1, 1), (144, 1, 1, 1))
        del arg70_1
        del buf53
        buf55 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_222, x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, x_223], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_26.run(buf55, buf54, arg71_1, 663552, grid=grid(663552), stream=stream0)
        del arg71_1
        del buf54
        # Topologically Sorted Source Nodes: [x_222, x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, x_223, x_224], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf56 = extern_kernels.convolution(buf55, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 40, 24, 24), (23040, 1, 960, 40))
        del arg72_1
        del buf55
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_225], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf57, arg73_1, arg74_1, arg75_1, arg76_1, 184320, grid=grid(184320), stream=stream0)
        del arg73_1
        del arg74_1
        del arg75_1
        del arg76_1
        # Topologically Sorted Source Nodes: [x_226], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg77_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 240, 24, 24), (138240, 1, 5760, 240))
        del arg77_1
        buf59 = buf58; del buf58  # reuse
        buf60 = empty_strided_cuda((8, 240, 24, 24), (138240, 1, 5760, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_227, x_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf59, arg78_1, arg79_1, arg80_1, arg81_1, buf60, 1105920, grid=grid(1105920), stream=stream0)
        del arg78_1
        del arg79_1
        del arg80_1
        del arg81_1
        del buf59
        # Topologically Sorted Source Nodes: [x_228, x_229], Original ATen: [aten.silu, aten.convolution]
        buf61 = extern_kernels.convolution(buf60, arg82_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf61, (8, 240, 24, 24), (138240, 1, 5760, 240))
        del arg82_1
        del buf60
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_29.run(buf62, arg83_1, arg84_1, arg85_1, arg86_1, 1105920, grid=grid(1105920), stream=stream0)
        del arg83_1
        del arg84_1
        del arg85_1
        del arg86_1
        buf63 = empty_strided_cuda((8, 240, 1, 1, 5), (1200, 1, 9600, 9600, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_231, x_se_92], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_30.run(buf62, buf63, 9600, 116, grid=grid(9600), stream=stream0)
        buf65 = empty_strided_cuda((8, 240, 1, 1), (240, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_231, x_se_92], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_31.run(buf63, buf65, 1920, 5, grid=grid(1920), stream=stream0)
        del buf63
        # Topologically Sorted Source Nodes: [x_231, x_se_92, x_se_93], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 10, 1, 1), (10, 1, 1, 1))
        del arg87_1
        del buf65
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_231, x_se_92, x_se_93, x_se_94], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_32.run(buf67, arg88_1, 80, grid=grid(80), stream=stream0)
        del arg88_1
        # Topologically Sorted Source Nodes: [x_231, x_se_92, x_se_93, x_se_94, x_se_95], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg89_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg89_1
        del buf67
        buf69 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_231, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, x_232], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_33.run(buf69, buf68, arg90_1, 1105920, grid=grid(1105920), stream=stream0)
        del arg90_1
        # Topologically Sorted Source Nodes: [x_231, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, x_232, x_233], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf70 = extern_kernels.convolution(buf69, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 40, 24, 24), (23040, 1, 960, 40))
        del arg91_1
        buf71 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_34.run(buf71, buf70, arg92_1, arg93_1, arg94_1, arg95_1, 184320, grid=grid(184320), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf70
        # Topologically Sorted Source Nodes: [x_234, x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 240, 24, 24), (138240, 1, 5760, 240))
        del arg96_1
        del buf71
        buf73 = buf72; del buf72  # reuse
        buf74 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf73, arg97_1, arg98_1, arg99_1, arg100_1, buf74, 1105920, grid=grid(1105920), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf73
        # Topologically Sorted Source Nodes: [x_238, x_239], Original ATen: [aten.silu, aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg101_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf75, (8, 240, 12, 12), (34560, 1, 2880, 240))
        del arg101_1
        del buf74
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_240], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_35.run(buf76, arg102_1, arg103_1, arg104_1, arg105_1, 276480, grid=grid(276480), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        buf77 = empty_strided_cuda((8, 240, 1, 1, 2), (480, 1, 3840, 3840, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_241, x_se_96], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_36.run(buf76, buf77, 3840, 72, grid=grid(3840), stream=stream0)
        buf79 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_se_96], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_37.run(buf77, buf79, 1920, 2, grid=grid(1920), stream=stream0)
        # Topologically Sorted Source Nodes: [x_241, x_se_96, x_se_97], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 10, 1, 1), (10, 1, 1, 1))
        del arg106_1
        del buf79
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_se_96, x_se_97, x_se_98], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_32.run(buf81, arg107_1, 80, grid=grid(80), stream=stream0)
        del arg107_1
        # Topologically Sorted Source Nodes: [x_241, x_se_96, x_se_97, x_se_98, x_se_99], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg108_1
        del buf81
        buf83 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_se_96, x_se_97, x_se_98, x_se_99, sigmoid_24, x_242], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_38.run(buf83, buf82, arg109_1, 276480, grid=grid(276480), stream=stream0)
        del arg109_1
        del buf82
        # Topologically Sorted Source Nodes: [x_241, x_se_96, x_se_97, x_se_98, x_se_99, sigmoid_24, x_242, x_243], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf84 = extern_kernels.convolution(buf83, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 80, 12, 12), (11520, 1, 960, 80))
        del arg110_1
        del buf83
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_39.run(buf85, arg111_1, arg112_1, arg113_1, arg114_1, 92160, grid=grid(92160), stream=stream0)
        del arg111_1
        del arg112_1
        del arg113_1
        del arg114_1
        # Topologically Sorted Source Nodes: [x_245], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 480, 12, 12), (69120, 1, 5760, 480))
        del arg115_1
        buf87 = buf86; del buf86  # reuse
        buf88 = empty_strided_cuda((8, 480, 12, 12), (69120, 1, 5760, 480), torch.float32)
        # Topologically Sorted Source Nodes: [x_246, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_40.run(buf87, arg116_1, arg117_1, arg118_1, arg119_1, buf88, 552960, grid=grid(552960), stream=stream0)
        del arg116_1
        del arg117_1
        del arg118_1
        del arg119_1
        del buf87
        # Topologically Sorted Source Nodes: [x_247, x_248], Original ATen: [aten.silu, aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg120_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf89, (8, 480, 12, 12), (69120, 1, 5760, 480))
        del arg120_1
        del buf88
        buf90 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [x_249], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf90, arg121_1, arg122_1, arg123_1, arg124_1, 552960, grid=grid(552960), stream=stream0)
        del arg121_1
        del arg122_1
        del arg123_1
        del arg124_1
        buf91 = empty_strided_cuda((8, 480, 1, 1, 2), (960, 1, 7680, 7680, 480), torch.float32)
        # Topologically Sorted Source Nodes: [x_250, x_se_100], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_42.run(buf90, buf91, 7680, 72, grid=grid(7680), stream=stream0)
        buf93 = reinterpret_tensor(buf77, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [x_250, x_se_100], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_43.run(buf91, buf93, 3840, 2, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [x_250, x_se_100, x_se_101], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg125_1
        del buf93
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_250, x_se_100, x_se_101, x_se_102], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_44.run(buf95, arg126_1, 160, grid=grid(160), stream=stream0)
        del arg126_1
        # Topologically Sorted Source Nodes: [x_250, x_se_100, x_se_101, x_se_102, x_se_103], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg127_1
        del buf95
        buf97 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_250, x_se_100, x_se_101, x_se_102, x_se_103, sigmoid_25, x_251], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_45.run(buf97, buf96, arg128_1, 552960, grid=grid(552960), stream=stream0)
        del arg128_1
        # Topologically Sorted Source Nodes: [x_250, x_se_100, x_se_101, x_se_102, x_se_103, sigmoid_25, x_251, x_252], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf98 = extern_kernels.convolution(buf97, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 80, 12, 12), (11520, 1, 960, 80))
        del arg129_1
        buf99 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_46.run(buf99, buf98, arg130_1, arg131_1, arg132_1, arg133_1, 92160, grid=grid(92160), stream=stream0)
        del arg130_1
        del arg131_1
        del arg132_1
        del arg133_1
        del buf98
        # Topologically Sorted Source Nodes: [x_255], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 480, 12, 12), (69120, 1, 5760, 480))
        del arg134_1
        buf101 = buf100; del buf100  # reuse
        buf102 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_40.run(buf101, arg135_1, arg136_1, arg137_1, arg138_1, buf102, 552960, grid=grid(552960), stream=stream0)
        del arg135_1
        del arg136_1
        del arg137_1
        del arg138_1
        del buf101
        # Topologically Sorted Source Nodes: [x_257, x_258], Original ATen: [aten.silu, aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg139_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf103, (8, 480, 12, 12), (69120, 1, 5760, 480))
        del arg139_1
        del buf102
        buf104 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf104, arg140_1, arg141_1, arg142_1, arg143_1, 552960, grid=grid(552960), stream=stream0)
        del arg140_1
        del arg141_1
        del arg142_1
        del arg143_1
        buf105 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_260, x_se_104], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_42.run(buf104, buf105, 7680, 72, grid=grid(7680), stream=stream0)
        buf107 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_260, x_se_104], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_43.run(buf105, buf107, 3840, 2, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [x_260, x_se_104, x_se_105], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg144_1
        del buf107
        buf109 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_260, x_se_104, x_se_105, x_se_106], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_44.run(buf109, arg145_1, 160, grid=grid(160), stream=stream0)
        del arg145_1
        # Topologically Sorted Source Nodes: [x_260, x_se_104, x_se_105, x_se_106, x_se_107], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg146_1
        del buf109
        buf111 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_260, x_se_104, x_se_105, x_se_106, x_se_107, sigmoid_26, x_261], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_45.run(buf111, buf110, arg147_1, 552960, grid=grid(552960), stream=stream0)
        del arg147_1
        # Topologically Sorted Source Nodes: [x_260, x_se_104, x_se_105, x_se_106, x_se_107, sigmoid_26, x_261, x_262], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf112 = extern_kernels.convolution(buf111, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 80, 12, 12), (11520, 1, 960, 80))
        del arg148_1
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_263, x_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf113, arg149_1, arg150_1, arg151_1, arg152_1, buf99, 92160, grid=grid(92160), stream=stream0)
        del arg149_1
        del arg150_1
        del arg151_1
        del arg152_1
        del buf99
        # Topologically Sorted Source Nodes: [x_265], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 480, 12, 12), (69120, 1, 5760, 480))
        del arg153_1
        buf115 = buf114; del buf114  # reuse
        buf116 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [x_266, x_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_40.run(buf115, arg154_1, arg155_1, arg156_1, arg157_1, buf116, 552960, grid=grid(552960), stream=stream0)
        del arg154_1
        del arg155_1
        del arg156_1
        del arg157_1
        del buf115
        # Topologically Sorted Source Nodes: [x_267, x_268], Original ATen: [aten.silu, aten.convolution]
        buf117 = extern_kernels.convolution(buf116, arg158_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf117, (8, 480, 12, 12), (69120, 1, 5760, 480))
        del arg158_1
        del buf116
        buf118 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [x_269], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf118, arg159_1, arg160_1, arg161_1, arg162_1, 552960, grid=grid(552960), stream=stream0)
        del arg159_1
        del arg160_1
        del arg161_1
        del arg162_1
        buf119 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_270, x_se_108], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_42.run(buf118, buf119, 7680, 72, grid=grid(7680), stream=stream0)
        buf121 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_270, x_se_108], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_43.run(buf119, buf121, 3840, 2, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [x_270, x_se_108, x_se_109], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf122 = extern_kernels.convolution(buf121, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg163_1
        del buf121
        buf123 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_270, x_se_108, x_se_109, x_se_110], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_44.run(buf123, arg164_1, 160, grid=grid(160), stream=stream0)
        del arg164_1
        # Topologically Sorted Source Nodes: [x_270, x_se_108, x_se_109, x_se_110, x_se_111], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg165_1
        del buf123
        buf125 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_270, x_se_108, x_se_109, x_se_110, x_se_111, sigmoid_27, x_271], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_45.run(buf125, buf124, arg166_1, 552960, grid=grid(552960), stream=stream0)
        del arg166_1
        # Topologically Sorted Source Nodes: [x_270, x_se_108, x_se_109, x_se_110, x_se_111, sigmoid_27, x_271, x_272], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf126 = extern_kernels.convolution(buf125, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 80, 12, 12), (11520, 1, 960, 80))
        del arg167_1
        buf127 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [x_273, x_274], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_46.run(buf127, buf126, arg168_1, arg169_1, arg170_1, arg171_1, 92160, grid=grid(92160), stream=stream0)
        del arg168_1
        del arg169_1
        del arg170_1
        del arg171_1
        del buf126
        # Topologically Sorted Source Nodes: [x_273, x_274, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 480, 12, 12), (69120, 1, 5760, 480))
        del arg172_1
        del buf127
        buf129 = buf128; del buf128  # reuse
        buf130 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [x_276, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_40.run(buf129, arg173_1, arg174_1, arg175_1, arg176_1, buf130, 552960, grid=grid(552960), stream=stream0)
        del arg173_1
        del arg174_1
        del arg175_1
        del arg176_1
        del buf129
        # Topologically Sorted Source Nodes: [x_277, x_278], Original ATen: [aten.silu, aten.convolution]
        buf131 = extern_kernels.convolution(buf130, arg177_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf131, (8, 480, 12, 12), (69120, 1, 5760, 480))
        del arg177_1
        del buf130
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf132, arg178_1, arg179_1, arg180_1, arg181_1, 552960, grid=grid(552960), stream=stream0)
        del arg178_1
        del arg179_1
        del arg180_1
        del arg181_1
        buf133 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [x_280, x_se_112], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_42.run(buf132, buf133, 7680, 72, grid=grid(7680), stream=stream0)
        buf135 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_280, x_se_112], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_43.run(buf133, buf135, 3840, 2, grid=grid(3840), stream=stream0)
        del buf133
        # Topologically Sorted Source Nodes: [x_280, x_se_112, x_se_113], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg182_1
        del buf135
        buf137 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_280, x_se_112, x_se_113, x_se_114], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_44.run(buf137, arg183_1, 160, grid=grid(160), stream=stream0)
        del arg183_1
        # Topologically Sorted Source Nodes: [x_280, x_se_112, x_se_113, x_se_114, x_se_115], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf138 = extern_kernels.convolution(buf137, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg184_1
        del buf137
        buf139 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [x_280, x_se_112, x_se_113, x_se_114, x_se_115, sigmoid_28, x_281], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_45.run(buf139, buf138, arg185_1, 552960, grid=grid(552960), stream=stream0)
        del arg185_1
        del buf138
        # Topologically Sorted Source Nodes: [x_280, x_se_112, x_se_113, x_se_114, x_se_115, sigmoid_28, x_281, x_282], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf140 = extern_kernels.convolution(buf139, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 112, 12, 12), (16128, 1, 1344, 112))
        del arg186_1
        del buf139
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_283], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_48.run(buf141, arg187_1, arg188_1, arg189_1, arg190_1, 129024, grid=grid(129024), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        # Topologically Sorted Source Nodes: [x_284], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 672, 12, 12), (96768, 1, 8064, 672))
        del arg191_1
        buf143 = buf142; del buf142  # reuse
        buf144 = empty_strided_cuda((8, 672, 12, 12), (96768, 1, 8064, 672), torch.float32)
        # Topologically Sorted Source Nodes: [x_285, x_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_49.run(buf143, arg192_1, arg193_1, arg194_1, arg195_1, buf144, 774144, grid=grid(774144), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        del buf143
        # Topologically Sorted Source Nodes: [x_286, x_287], Original ATen: [aten.silu, aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg196_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf145, (8, 672, 12, 12), (96768, 1, 8064, 672))
        del arg196_1
        del buf144
        buf146 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [x_288], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_50.run(buf146, arg197_1, arg198_1, arg199_1, arg200_1, 774144, grid=grid(774144), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        buf147 = empty_strided_cuda((8, 672, 1, 1, 2), (1344, 1, 10752, 10752, 672), torch.float32)
        # Topologically Sorted Source Nodes: [x_289, x_se_116], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_51.run(buf146, buf147, 10752, 72, grid=grid(10752), stream=stream0)
        buf149 = empty_strided_cuda((8, 672, 1, 1), (672, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_289, x_se_116], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_52.run(buf147, buf149, 5376, 2, grid=grid(5376), stream=stream0)
        # Topologically Sorted Source Nodes: [x_289, x_se_116, x_se_117], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf150 = extern_kernels.convolution(buf149, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg201_1
        del buf149
        buf151 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [x_289, x_se_116, x_se_117, x_se_118], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_53.run(buf151, arg202_1, 224, grid=grid(224), stream=stream0)
        del arg202_1
        # Topologically Sorted Source Nodes: [x_289, x_se_116, x_se_117, x_se_118, x_se_119], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg203_1
        del buf151
        buf153 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_289, x_se_116, x_se_117, x_se_118, x_se_119, sigmoid_29, x_290], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_54.run(buf153, buf152, arg204_1, 774144, grid=grid(774144), stream=stream0)
        del arg204_1
        # Topologically Sorted Source Nodes: [x_289, x_se_116, x_se_117, x_se_118, x_se_119, sigmoid_29, x_290, x_291], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf154 = extern_kernels.convolution(buf153, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 112, 12, 12), (16128, 1, 1344, 112))
        del arg205_1
        buf155 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_55.run(buf155, buf154, arg206_1, arg207_1, arg208_1, arg209_1, 129024, grid=grid(129024), stream=stream0)
        del arg206_1
        del arg207_1
        del arg208_1
        del arg209_1
        del buf154
        # Topologically Sorted Source Nodes: [x_294], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 672, 12, 12), (96768, 1, 8064, 672))
        del arg210_1
        buf157 = buf156; del buf156  # reuse
        buf158 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [x_295, x_296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_49.run(buf157, arg211_1, arg212_1, arg213_1, arg214_1, buf158, 774144, grid=grid(774144), stream=stream0)
        del arg211_1
        del arg212_1
        del arg213_1
        del arg214_1
        del buf157
        # Topologically Sorted Source Nodes: [x_296, x_297], Original ATen: [aten.silu, aten.convolution]
        buf159 = extern_kernels.convolution(buf158, arg215_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf159, (8, 672, 12, 12), (96768, 1, 8064, 672))
        del arg215_1
        del buf158
        buf160 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_298], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_50.run(buf160, arg216_1, arg217_1, arg218_1, arg219_1, 774144, grid=grid(774144), stream=stream0)
        del arg216_1
        del arg217_1
        del arg218_1
        del arg219_1
        buf161 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_299, x_se_120], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_51.run(buf160, buf161, 10752, 72, grid=grid(10752), stream=stream0)
        buf163 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [x_299, x_se_120], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_52.run(buf161, buf163, 5376, 2, grid=grid(5376), stream=stream0)
        # Topologically Sorted Source Nodes: [x_299, x_se_120, x_se_121], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf164 = extern_kernels.convolution(buf163, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg220_1
        del buf163
        buf165 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [x_299, x_se_120, x_se_121, x_se_122], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_53.run(buf165, arg221_1, 224, grid=grid(224), stream=stream0)
        del arg221_1
        # Topologically Sorted Source Nodes: [x_299, x_se_120, x_se_121, x_se_122, x_se_123], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf166 = extern_kernels.convolution(buf165, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg222_1
        del buf165
        buf167 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_299, x_se_120, x_se_121, x_se_122, x_se_123, sigmoid_30, x_300], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_54.run(buf167, buf166, arg223_1, 774144, grid=grid(774144), stream=stream0)
        del arg223_1
        # Topologically Sorted Source Nodes: [x_299, x_se_120, x_se_121, x_se_122, x_se_123, sigmoid_30, x_300, x_301], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf168 = extern_kernels.convolution(buf167, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 112, 12, 12), (16128, 1, 1344, 112))
        del arg224_1
        buf169 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_302, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_55.run(buf169, buf168, arg225_1, arg226_1, arg227_1, arg228_1, 129024, grid=grid(129024), stream=stream0)
        del arg225_1
        del arg226_1
        del arg227_1
        del arg228_1
        del buf168
        # Topologically Sorted Source Nodes: [x_304], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, arg229_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (8, 672, 12, 12), (96768, 1, 8064, 672))
        del arg229_1
        buf171 = buf170; del buf170  # reuse
        buf172 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_305, x_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_49.run(buf171, arg230_1, arg231_1, arg232_1, arg233_1, buf172, 774144, grid=grid(774144), stream=stream0)
        del arg230_1
        del arg231_1
        del arg232_1
        del arg233_1
        del buf171
        # Topologically Sorted Source Nodes: [x_306, x_307], Original ATen: [aten.silu, aten.convolution]
        buf173 = extern_kernels.convolution(buf172, arg234_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf173, (8, 672, 12, 12), (96768, 1, 8064, 672))
        del arg234_1
        del buf172
        buf174 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [x_308], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_50.run(buf174, arg235_1, arg236_1, arg237_1, arg238_1, 774144, grid=grid(774144), stream=stream0)
        del arg235_1
        del arg236_1
        del arg237_1
        del arg238_1
        buf175 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_309, x_se_124], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_51.run(buf174, buf175, 10752, 72, grid=grid(10752), stream=stream0)
        buf177 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [x_309, x_se_124], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_52.run(buf175, buf177, 5376, 2, grid=grid(5376), stream=stream0)
        del buf175
        # Topologically Sorted Source Nodes: [x_309, x_se_124, x_se_125], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf178 = extern_kernels.convolution(buf177, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg239_1
        del buf177
        buf179 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_309, x_se_124, x_se_125, x_se_126], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_53.run(buf179, arg240_1, 224, grid=grid(224), stream=stream0)
        del arg240_1
        # Topologically Sorted Source Nodes: [x_309, x_se_124, x_se_125, x_se_126, x_se_127], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf180 = extern_kernels.convolution(buf179, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg241_1
        del buf179
        buf181 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [x_309, x_se_124, x_se_125, x_se_126, x_se_127, sigmoid_31, x_310], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_54.run(buf181, buf180, arg242_1, 774144, grid=grid(774144), stream=stream0)
        del arg242_1
        # Topologically Sorted Source Nodes: [x_309, x_se_124, x_se_125, x_se_126, x_se_127, sigmoid_31, x_310, x_311], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf182 = extern_kernels.convolution(buf181, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 112, 12, 12), (16128, 1, 1344, 112))
        del arg243_1
        buf183 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_312, x_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_55.run(buf183, buf182, arg244_1, arg245_1, arg246_1, arg247_1, 129024, grid=grid(129024), stream=stream0)
        del arg244_1
        del arg245_1
        del arg246_1
        del arg247_1
        del buf182
        # Topologically Sorted Source Nodes: [x_312, x_313, x_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf184 = extern_kernels.convolution(buf183, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 672, 12, 12), (96768, 1, 8064, 672))
        del arg248_1
        del buf183
        buf185 = buf184; del buf184  # reuse
        buf186 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [x_315, x_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_49.run(buf185, arg249_1, arg250_1, arg251_1, arg252_1, buf186, 774144, grid=grid(774144), stream=stream0)
        del arg249_1
        del arg250_1
        del arg251_1
        del arg252_1
        del buf185
        # Topologically Sorted Source Nodes: [x_316, x_317], Original ATen: [aten.silu, aten.convolution]
        buf187 = extern_kernels.convolution(buf186, arg253_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf187, (8, 672, 6, 6), (24192, 1, 4032, 672))
        del arg253_1
        del buf186
        buf188 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [x_318], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_56.run(buf188, arg254_1, arg255_1, arg256_1, arg257_1, 193536, grid=grid(193536), stream=stream0)
        del arg254_1
        del arg255_1
        del arg256_1
        del arg257_1
        buf190 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [x_319, x_se_128], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_57.run(buf188, buf190, 5376, 36, grid=grid(5376), stream=stream0)
        # Topologically Sorted Source Nodes: [x_319, x_se_128, x_se_129], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf191 = extern_kernels.convolution(buf190, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg258_1
        del buf190
        buf192 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [x_319, x_se_128, x_se_129, x_se_130], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_53.run(buf192, arg259_1, 224, grid=grid(224), stream=stream0)
        del arg259_1
        # Topologically Sorted Source Nodes: [x_319, x_se_128, x_se_129, x_se_130, x_se_131], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf193 = extern_kernels.convolution(buf192, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg260_1
        del buf192
        buf194 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_319, x_se_128, x_se_129, x_se_130, x_se_131, sigmoid_32, x_320], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_58.run(buf194, buf193, arg261_1, 193536, grid=grid(193536), stream=stream0)
        del arg261_1
        del buf193
        # Topologically Sorted Source Nodes: [x_319, x_se_128, x_se_129, x_se_130, x_se_131, sigmoid_32, x_320, x_321], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf195 = extern_kernels.convolution(buf194, arg262_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 192, 6, 6), (6912, 1, 1152, 192))
        del arg262_1
        del buf194
        buf196 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_322], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_59.run(buf196, arg263_1, arg264_1, arg265_1, arg266_1, 55296, grid=grid(55296), stream=stream0)
        del arg263_1
        del arg264_1
        del arg265_1
        del arg266_1
        # Topologically Sorted Source Nodes: [x_323], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, arg267_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
        del arg267_1
        buf198 = buf197; del buf197  # reuse
        buf199 = empty_strided_cuda((8, 1152, 6, 6), (41472, 1, 6912, 1152), torch.float32)
        # Topologically Sorted Source Nodes: [x_324, x_325], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_60.run(buf198, arg268_1, arg269_1, arg270_1, arg271_1, buf199, 331776, grid=grid(331776), stream=stream0)
        del arg268_1
        del arg269_1
        del arg270_1
        del arg271_1
        del buf198
        # Topologically Sorted Source Nodes: [x_325, x_326], Original ATen: [aten.silu, aten.convolution]
        buf200 = extern_kernels.convolution(buf199, arg272_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf200, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
        del arg272_1
        del buf199
        buf201 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_61.run(buf201, arg273_1, arg274_1, arg275_1, arg276_1, 331776, grid=grid(331776), stream=stream0)
        del arg273_1
        del arg274_1
        del arg275_1
        del arg276_1
        buf203 = empty_strided_cuda((8, 1152, 1, 1), (1152, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_328, x_se_132], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_62.run(buf201, buf203, 9216, 36, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_328, x_se_132, x_se_133], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf204 = extern_kernels.convolution(buf203, arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg277_1
        del buf203
        buf205 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_328, x_se_132, x_se_133, x_se_134], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_63.run(buf205, arg278_1, 384, grid=grid(384), stream=stream0)
        del arg278_1
        # Topologically Sorted Source Nodes: [x_328, x_se_132, x_se_133, x_se_134, x_se_135], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf206 = extern_kernels.convolution(buf205, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg279_1
        del buf205
        buf207 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [x_328, x_se_132, x_se_133, x_se_134, x_se_135, sigmoid_33, x_329], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_64.run(buf207, buf206, arg280_1, 331776, grid=grid(331776), stream=stream0)
        del arg280_1
        # Topologically Sorted Source Nodes: [x_328, x_se_132, x_se_133, x_se_134, x_se_135, sigmoid_33, x_329, x_330], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf208 = extern_kernels.convolution(buf207, arg281_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 192, 6, 6), (6912, 1, 1152, 192))
        del arg281_1
        buf209 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [x_331, x_332], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_65.run(buf209, buf208, arg282_1, arg283_1, arg284_1, arg285_1, 55296, grid=grid(55296), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        del buf208
        # Topologically Sorted Source Nodes: [x_333], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
        del arg286_1
        buf211 = buf210; del buf210  # reuse
        buf212 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_334, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_60.run(buf211, arg287_1, arg288_1, arg289_1, arg290_1, buf212, 331776, grid=grid(331776), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        del buf211
        # Topologically Sorted Source Nodes: [x_335, x_336], Original ATen: [aten.silu, aten.convolution]
        buf213 = extern_kernels.convolution(buf212, arg291_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf213, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
        del arg291_1
        del buf212
        buf214 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [x_337], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_61.run(buf214, arg292_1, arg293_1, arg294_1, arg295_1, 331776, grid=grid(331776), stream=stream0)
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        buf216 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [x_338, x_se_136], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_62.run(buf214, buf216, 9216, 36, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_338, x_se_136, x_se_137], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf217 = extern_kernels.convolution(buf216, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg296_1
        del buf216
        buf218 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_338, x_se_136, x_se_137, x_se_138], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_63.run(buf218, arg297_1, 384, grid=grid(384), stream=stream0)
        del arg297_1
        # Topologically Sorted Source Nodes: [x_338, x_se_136, x_se_137, x_se_138, x_se_139], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf219 = extern_kernels.convolution(buf218, arg298_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg298_1
        del buf218
        buf220 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [x_338, x_se_136, x_se_137, x_se_138, x_se_139, sigmoid_34, x_339], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_64.run(buf220, buf219, arg299_1, 331776, grid=grid(331776), stream=stream0)
        del arg299_1
        # Topologically Sorted Source Nodes: [x_338, x_se_136, x_se_137, x_se_138, x_se_139, sigmoid_34, x_339, x_340], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf221 = extern_kernels.convolution(buf220, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 192, 6, 6), (6912, 1, 1152, 192))
        del arg300_1
        buf222 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [x_341, x_342], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_65.run(buf222, buf221, arg301_1, arg302_1, arg303_1, arg304_1, 55296, grid=grid(55296), stream=stream0)
        del arg301_1
        del arg302_1
        del arg303_1
        del arg304_1
        del buf221
        # Topologically Sorted Source Nodes: [x_343], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, arg305_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
        del arg305_1
        buf224 = buf223; del buf223  # reuse
        buf225 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [x_344, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_60.run(buf224, arg306_1, arg307_1, arg308_1, arg309_1, buf225, 331776, grid=grid(331776), stream=stream0)
        del arg306_1
        del arg307_1
        del arg308_1
        del arg309_1
        del buf224
        # Topologically Sorted Source Nodes: [x_345, x_346], Original ATen: [aten.silu, aten.convolution]
        buf226 = extern_kernels.convolution(buf225, arg310_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf226, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
        del arg310_1
        del buf225
        buf227 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [x_347], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_61.run(buf227, arg311_1, arg312_1, arg313_1, arg314_1, 331776, grid=grid(331776), stream=stream0)
        del arg311_1
        del arg312_1
        del arg313_1
        del arg314_1
        buf229 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_348, x_se_140], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_62.run(buf227, buf229, 9216, 36, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_348, x_se_140, x_se_141], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf230 = extern_kernels.convolution(buf229, arg315_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg315_1
        del buf229
        buf231 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [x_348, x_se_140, x_se_141, x_se_142], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_63.run(buf231, arg316_1, 384, grid=grid(384), stream=stream0)
        del arg316_1
        # Topologically Sorted Source Nodes: [x_348, x_se_140, x_se_141, x_se_142, x_se_143], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf232 = extern_kernels.convolution(buf231, arg317_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg317_1
        del buf231
        buf233 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [x_348, x_se_140, x_se_141, x_se_142, x_se_143, sigmoid_35, x_349], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_64.run(buf233, buf232, arg318_1, 331776, grid=grid(331776), stream=stream0)
        del arg318_1
        # Topologically Sorted Source Nodes: [x_348, x_se_140, x_se_141, x_se_142, x_se_143, sigmoid_35, x_349, x_350], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf234 = extern_kernels.convolution(buf233, arg319_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (8, 192, 6, 6), (6912, 1, 1152, 192))
        del arg319_1
        buf235 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [x_351, x_352], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_65.run(buf235, buf234, arg320_1, arg321_1, arg322_1, arg323_1, 55296, grid=grid(55296), stream=stream0)
        del arg320_1
        del arg321_1
        del arg322_1
        del arg323_1
        del buf234
        # Topologically Sorted Source Nodes: [x_353], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, arg324_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
        del arg324_1
        buf237 = buf236; del buf236  # reuse
        buf238 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_354, x_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_60.run(buf237, arg325_1, arg326_1, arg327_1, arg328_1, buf238, 331776, grid=grid(331776), stream=stream0)
        del arg325_1
        del arg326_1
        del arg327_1
        del arg328_1
        del buf237
        # Topologically Sorted Source Nodes: [x_355, x_356], Original ATen: [aten.silu, aten.convolution]
        buf239 = extern_kernels.convolution(buf238, arg329_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf239, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
        del arg329_1
        del buf238
        buf240 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [x_357], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_61.run(buf240, arg330_1, arg331_1, arg332_1, arg333_1, 331776, grid=grid(331776), stream=stream0)
        del arg330_1
        del arg331_1
        del arg332_1
        del arg333_1
        buf242 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_358, x_se_144], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_62.run(buf240, buf242, 9216, 36, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_358, x_se_144, x_se_145], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf243 = extern_kernels.convolution(buf242, arg334_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg334_1
        del buf242
        buf244 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [x_358, x_se_144, x_se_145, x_se_146], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_63.run(buf244, arg335_1, 384, grid=grid(384), stream=stream0)
        del arg335_1
        # Topologically Sorted Source Nodes: [x_358, x_se_144, x_se_145, x_se_146, x_se_147], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf245 = extern_kernels.convolution(buf244, arg336_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg336_1
        del buf244
        buf246 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [x_358, x_se_144, x_se_145, x_se_146, x_se_147, sigmoid_36, x_359], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_64.run(buf246, buf245, arg337_1, 331776, grid=grid(331776), stream=stream0)
        del arg337_1
        # Topologically Sorted Source Nodes: [x_358, x_se_144, x_se_145, x_se_146, x_se_147, sigmoid_36, x_359, x_360], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf247 = extern_kernels.convolution(buf246, arg338_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (8, 192, 6, 6), (6912, 1, 1152, 192))
        del arg338_1
        buf248 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [x_361, x_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_65.run(buf248, buf247, arg339_1, arg340_1, arg341_1, arg342_1, 55296, grid=grid(55296), stream=stream0)
        del arg339_1
        del arg340_1
        del arg341_1
        del arg342_1
        del buf247
        # Topologically Sorted Source Nodes: [x_361, x_362, x_363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf249 = extern_kernels.convolution(buf248, arg343_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
        del arg343_1
        del buf248
        buf250 = buf249; del buf249  # reuse
        buf251 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [x_364, x_365], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_60.run(buf250, arg344_1, arg345_1, arg346_1, arg347_1, buf251, 331776, grid=grid(331776), stream=stream0)
        del arg344_1
        del arg345_1
        del arg346_1
        del arg347_1
        del buf250
        # Topologically Sorted Source Nodes: [x_365, x_366], Original ATen: [aten.silu, aten.convolution]
        buf252 = extern_kernels.convolution(buf251, arg348_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf252, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
        del arg348_1
        del buf251
        buf253 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [x_367], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_61.run(buf253, arg349_1, arg350_1, arg351_1, arg352_1, 331776, grid=grid(331776), stream=stream0)
        del arg349_1
        del arg350_1
        del arg351_1
        del arg352_1
        buf255 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [x_368, x_se_148], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_62.run(buf253, buf255, 9216, 36, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_368, x_se_148, x_se_149], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf256 = extern_kernels.convolution(buf255, arg353_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg353_1
        del buf255
        buf257 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [x_368, x_se_148, x_se_149, x_se_150], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_63.run(buf257, arg354_1, 384, grid=grid(384), stream=stream0)
        del arg354_1
        # Topologically Sorted Source Nodes: [x_368, x_se_148, x_se_149, x_se_150, x_se_151], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf258 = extern_kernels.convolution(buf257, arg355_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg355_1
        del buf257
        buf259 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [x_368, x_se_148, x_se_149, x_se_150, x_se_151, sigmoid_37, x_369], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_64.run(buf259, buf258, arg356_1, 331776, grid=grid(331776), stream=stream0)
        del arg356_1
        del buf258
        # Topologically Sorted Source Nodes: [x_368, x_se_148, x_se_149, x_se_150, x_se_151, sigmoid_37, x_369, x_370], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf260 = extern_kernels.convolution(buf259, arg357_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (8, 320, 6, 6), (11520, 1, 1920, 320))
        del arg357_1
        del buf259
        buf261 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [x_371], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_66.run(buf261, arg358_1, arg359_1, arg360_1, arg361_1, 92160, grid=grid(92160), stream=stream0)
        del arg358_1
        del arg359_1
        del arg360_1
        del arg361_1
        # Topologically Sorted Source Nodes: [x_371, x_372], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf262 = extern_kernels.convolution(buf261, arg362_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 1280, 6, 6), (46080, 1, 7680, 1280))
        del arg362_1
        del buf261
        buf263 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [x_373], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_67.run(buf263, arg363_1, arg364_1, arg365_1, arg366_1, 368640, grid=grid(368640), stream=stream0)
        del arg363_1
        del arg364_1
        del arg365_1
        del arg366_1
        buf265 = empty_strided_cuda((8, 1280, 1, 1), (1280, 1, 10240, 10240), torch.float32)
        # Topologically Sorted Source Nodes: [x_374, x_375], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_68.run(buf263, buf265, 10240, 36, grid=grid(10240), stream=stream0)
        del buf263
        buf266 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_377], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg368_1, reinterpret_tensor(buf265, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg367_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf266)
        del arg367_1
        del arg368_1
        del buf265
    return (buf266, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 192, 192), (110592, 36864, 192, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    arg296_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tinynet_a', benchmark_compiled_module)
