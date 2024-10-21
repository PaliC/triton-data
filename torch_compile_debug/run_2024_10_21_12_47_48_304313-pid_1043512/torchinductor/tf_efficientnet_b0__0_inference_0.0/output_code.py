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


# kernel path: /tmp/torchinductor_sahanp/pk/cpkhbgeg2kcqjjmkgqixokomy6bkct57mjcapnwcdeogg2c252d7.py
# Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_164 => constant_pad_nd_5
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg1_1, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_0 = async_compile.triton('triton_poi_fused_constant_pad_nd_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 50625
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 225)
    x2 = xindex % 225
    y4 = yindex
    x5 = xindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = x3
    tmp1 = tl.full([1, 1], 224, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (224*x3) + (50176*y4)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (y0 + (3*x5) + (151875*y1)), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jf/cjfodmtfuxyzhxsvkyg4ks5pht5fymfjilyro6kqrtfpo5pydiqi.py
# Topologically Sorted Source Nodes: [x_164, x_165], Original ATen: [aten.constant_pad_nd, aten.convolution]
# Source node to ATen node mapping:
#   x_164 => constant_pad_nd_5
#   x_165 => convolution_81
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg1_1, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_81 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_5, %arg0_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_1 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ip/cip4vzptwj4l6dfin6mv7gfl5lkbtcwwq52ueryo2aufryqmmzh7.py
# Topologically Sorted Source Nodes: [x_166, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_166 => add_108, mul_213, mul_214, sub_49
#   x_167 => mul_215, sigmoid_65
# Graph fragment:
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_81, %unsqueeze_393), kwargs = {})
#   %mul_213 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_213, %unsqueeze_397), kwargs = {})
#   %add_108 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_214, %unsqueeze_399), kwargs = {})
#   %sigmoid_65 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_108,), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_108, %sigmoid_65), kwargs = {})
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/7j/c7jodfbqvuqqhdfnjsdtto4gyquslhmzsnj5pgmjlxq2f6tenipt.py
# Topologically Sorted Source Nodes: [x_169], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_169 => add_110, mul_217, mul_218, sub_50
# Graph fragment:
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_82, %unsqueeze_401), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_403), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_217, %unsqueeze_405), kwargs = {})
#   %add_110 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_218, %unsqueeze_407), kwargs = {})
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/s6/cs6fqcvmjls4whqeytkt64xrnk356bxbirdhshxxghvp5ljw2c5i.py
# Topologically Sorted Source Nodes: [x_170, x_se_64], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_170 => mul_219, sigmoid_66
#   x_se_64 => mean_17
# Graph fragment:
#   %sigmoid_66 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_110,), kwargs = {})
#   %mul_219 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_110, %sigmoid_66), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_219, [2, 3], True), kwargs = {})
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
    xnumel = 25088
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


# kernel path: /tmp/torchinductor_sahanp/nm/cnmizzg5xyjgxdqouk3qiaimzpgpdsa7gqjohgtpr7o7ys57rr5k.py
# Topologically Sorted Source Nodes: [x_170, x_se_64], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_170 => mul_219, sigmoid_66
#   x_se_64 => mean_17
# Graph fragment:
#   %sigmoid_66 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_110,), kwargs = {})
#   %mul_219 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_110, %sigmoid_66), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_219, [2, 3], True), kwargs = {})
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
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (3136*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = 12544.0
    tmp5 = tmp2 / tmp4
    tl.store(out_ptr1 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/s7/cs73aqtd4o47swivajt3nmevk5nxbdzsbm5zpz4ct6hzbt5ph6ap.py
# Topologically Sorted Source Nodes: [x_170, x_se_64, x_se_65, x_se_66], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_170 => mul_219, sigmoid_66
#   x_se_64 => mean_17
#   x_se_65 => convolution_83
#   x_se_66 => mul_220, sigmoid_67
# Graph fragment:
#   %sigmoid_66 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_110,), kwargs = {})
#   %mul_219 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_110, %sigmoid_66), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_219, [2, 3], True), kwargs = {})
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_17, %arg11_1, %arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_67 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_83,), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_83, %sigmoid_67), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/pz/cpzczuvu6toyvy4jdx7dnmu7w2hg3qx36srbdp4antkqeute6ruc.py
# Topologically Sorted Source Nodes: [x_170, x_se_64, x_se_65, x_se_66, x_se_67, sigmoid_16, x_171], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_16 => sigmoid_68
#   x_170 => mul_219, sigmoid_66
#   x_171 => mul_221
#   x_se_64 => mean_17
#   x_se_65 => convolution_83
#   x_se_66 => mul_220, sigmoid_67
#   x_se_67 => convolution_84
# Graph fragment:
#   %sigmoid_66 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_110,), kwargs = {})
#   %mul_219 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_110, %sigmoid_66), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_219, [2, 3], True), kwargs = {})
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_17, %arg11_1, %arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_67 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_83,), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_83, %sigmoid_67), kwargs = {})
#   %convolution_84 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_220, %arg13_1, %arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_68 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_84,), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_219, %sigmoid_68), kwargs = {})
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
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 32
    x2 = (xindex // 401408)
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


# kernel path: /tmp/torchinductor_sahanp/gf/cgflmo26spmakrgz25zfh3mmxynjjsk3xdk3cmang3datyexpywi.py
# Topologically Sorted Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_173 => add_112, mul_223, mul_224, sub_51
# Graph fragment:
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_85, %unsqueeze_409), kwargs = {})
#   %mul_223 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_411), kwargs = {})
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_223, %unsqueeze_413), kwargs = {})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_224, %unsqueeze_415), kwargs = {})
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/ms/cmsavxvwgszitx6qg2ggzxr27ayt4v247q2ppjkvcd4ur6cj3eco.py
# Topologically Sorted Source Nodes: [x_175], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_175 => add_114, mul_226, mul_227, sub_52
# Graph fragment:
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_86, %unsqueeze_417), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_419), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_226, %unsqueeze_421), kwargs = {})
#   %add_114 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_227, %unsqueeze_423), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9633792
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/4x/c4xzpme5fet4y4rovjspibvgphtclxm3wkiqsh5og3citm2xlmfo.py
# Topologically Sorted Source Nodes: [x_176, x_177], Original ATen: [aten.silu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_176 => mul_228, sigmoid_69
#   x_177 => constant_pad_nd_6
# Graph fragment:
#   %sigmoid_69 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_114,), kwargs = {})
#   %mul_228 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_114, %sigmoid_69), kwargs = {})
#   %constant_pad_nd_6 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_228, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_silu_10 = async_compile.triton('triton_poi_fused_constant_pad_nd_silu_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_silu_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9806592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 10848) % 113
    x1 = (xindex // 96) % 113
    x3 = (xindex // 1225824)
    x4 = xindex % 10848
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 112, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (10752*x2) + (1204224*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x5), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xy/cxyxy4yauvr2rkypawjb7uhhr32xanqlccgyl472x7wy5s5bri23.py
# Topologically Sorted Source Nodes: [x_179], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_179 => add_116, mul_230, mul_231, sub_53
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_425), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_230, %unsqueeze_429), kwargs = {})
#   %add_116 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_231, %unsqueeze_431), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/sb/csb7q6boyatdgvkwzu2kq37ny3upitttbxq4ugxumn6f5s4l2msk.py
# Topologically Sorted Source Nodes: [x_180, x_se_68], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_180 => mul_232, sigmoid_70
#   x_se_68 => mean_18
# Graph fragment:
#   %sigmoid_70 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_116,), kwargs = {})
#   %mul_232 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_116, %sigmoid_70), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_232, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_12 = async_compile.triton('triton_red_fused_mean_silu_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_12(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96) % 25
    x0 = xindex % 96
    x2 = (xindex // 2400)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r3 + (126*x1)) % 3136)) + (301056*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/wi/cwizejjofn2jxzeppqyihi5kdgpnbgch6tlpxaxkhkmjwh7ldzvx.py
# Topologically Sorted Source Nodes: [x_180, x_se_68], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_180 => mul_232, sigmoid_70
#   x_se_68 => mean_18
# Graph fragment:
#   %sigmoid_70 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_116,), kwargs = {})
#   %mul_232 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_116, %sigmoid_70), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_232, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_13 = async_compile.triton('triton_per_fused_mean_silu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_13(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (2400*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 3136.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ub/cub63pioebybtrd6hg76pbu3arsx5l2xpfziu74ktqri7rpowyom.py
# Topologically Sorted Source Nodes: [x_180, x_se_68, x_se_69, x_se_70], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_180 => mul_232, sigmoid_70
#   x_se_68 => mean_18
#   x_se_69 => convolution_88
#   x_se_70 => mul_233, sigmoid_71
# Graph fragment:
#   %sigmoid_70 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_116,), kwargs = {})
#   %mul_232 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_116, %sigmoid_70), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_232, [2, 3], True), kwargs = {})
#   %convolution_88 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_18, %arg30_1, %arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_71 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_88,), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, %sigmoid_71), kwargs = {})
triton_poi_fused_convolution_mean_silu_14 = async_compile.triton('triton_poi_fused_convolution_mean_silu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/cs/ccsrcedqg3qqmv7hrx5ng7vpw2fxy4ndemadjr4gkhqkgfe565ym.py
# Topologically Sorted Source Nodes: [x_180, x_se_68, x_se_69, x_se_70, x_se_71, sigmoid_17, x_181], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_17 => sigmoid_72
#   x_180 => mul_232, sigmoid_70
#   x_181 => mul_234
#   x_se_68 => mean_18
#   x_se_69 => convolution_88
#   x_se_70 => mul_233, sigmoid_71
#   x_se_71 => convolution_89
# Graph fragment:
#   %sigmoid_70 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_116,), kwargs = {})
#   %mul_232 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_116, %sigmoid_70), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_232, [2, 3], True), kwargs = {})
#   %convolution_88 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_18, %arg30_1, %arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_71 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_88,), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, %sigmoid_71), kwargs = {})
#   %convolution_89 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_233, %arg32_1, %arg33_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_72 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_89,), kwargs = {})
#   %mul_234 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_232, %sigmoid_72), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_15 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_15(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 96
    x2 = (xindex // 301056)
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


# kernel path: /tmp/torchinductor_sahanp/44/c44cikxaz46fvkfjbb3s5nkzzrw3k5tjlg6amcobi2fenjccm5pq.py
# Topologically Sorted Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_183 => add_118, mul_236, mul_237, sub_54
# Graph fragment:
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_90, %unsqueeze_433), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_435), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_236, %unsqueeze_437), kwargs = {})
#   %add_118 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_237, %unsqueeze_439), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/35/c35ohf7ae63q6vizdm5zzetc4dtywpowbqirpbx4dkl7rjyilto7.py
# Topologically Sorted Source Nodes: [x_185, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_185 => add_120, mul_239, mul_240, sub_55
#   x_186 => mul_241, sigmoid_73
# Graph fragment:
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_91, %unsqueeze_441), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_443), kwargs = {})
#   %mul_240 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_239, %unsqueeze_445), kwargs = {})
#   %add_120 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_240, %unsqueeze_447), kwargs = {})
#   %sigmoid_73 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_120,), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_120, %sigmoid_73), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/w4/cw4lkj3qfn2ohjllq5m5uxdesghdjkbbtxxub7rwsb4xl2j6epj4.py
# Topologically Sorted Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_188 => add_122, mul_243, mul_244, sub_56
# Graph fragment:
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_92, %unsqueeze_449), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %unsqueeze_451), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_243, %unsqueeze_453), kwargs = {})
#   %add_122 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_244, %unsqueeze_455), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/5g/c5gav77f5yqwg2smlk56rmwtyaalpjgpyfqkgmn2dcczjy76vlrq.py
# Topologically Sorted Source Nodes: [x_189, x_se_72], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_189 => mul_245, sigmoid_74
#   x_se_72 => mean_19
# Graph fragment:
#   %sigmoid_74 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_122,), kwargs = {})
#   %mul_245 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_122, %sigmoid_74), kwargs = {})
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_245, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_19 = async_compile.triton('triton_red_fused_mean_silu_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_19(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 144) % 25
    x0 = xindex % 144
    x2 = (xindex // 3600)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (144*((r3 + (126*x1)) % 3136)) + (451584*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/4p/c4pu6iaeam4tudcjvgxav5kfmctkujiuzppfhtd6hm5rc7na2mqq.py
# Topologically Sorted Source Nodes: [x_189, x_se_72], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_189 => mul_245, sigmoid_74
#   x_se_72 => mean_19
# Graph fragment:
#   %sigmoid_74 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_122,), kwargs = {})
#   %mul_245 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_122, %sigmoid_74), kwargs = {})
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_245, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_20 = async_compile.triton('triton_per_fused_mean_silu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_20(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (3600*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 3136.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ly/clythjxopmmvi5h3kzvauxkedtlfklxcar7xd6pzkrfziclanycz.py
# Topologically Sorted Source Nodes: [x_189, x_se_72, x_se_73, x_se_74], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_189 => mul_245, sigmoid_74
#   x_se_72 => mean_19
#   x_se_73 => convolution_93
#   x_se_74 => mul_246, sigmoid_75
# Graph fragment:
#   %sigmoid_74 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_122,), kwargs = {})
#   %mul_245 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_122, %sigmoid_74), kwargs = {})
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_245, [2, 3], True), kwargs = {})
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_19, %arg49_1, %arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_75 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_93,), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, %sigmoid_75), kwargs = {})
triton_poi_fused_convolution_mean_silu_21 = async_compile.triton('triton_poi_fused_convolution_mean_silu_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/sp/cspwp54h6ql5mbkary6rtj6hfrztxvrv24uaufsfumoq4z46pgwq.py
# Topologically Sorted Source Nodes: [x_189, x_se_72, x_se_73, x_se_74, x_se_75, sigmoid_18, x_190], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_18 => sigmoid_76
#   x_189 => mul_245, sigmoid_74
#   x_190 => mul_247
#   x_se_72 => mean_19
#   x_se_73 => convolution_93
#   x_se_74 => mul_246, sigmoid_75
#   x_se_75 => convolution_94
# Graph fragment:
#   %sigmoid_74 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_122,), kwargs = {})
#   %mul_245 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_122, %sigmoid_74), kwargs = {})
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_245, [2, 3], True), kwargs = {})
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_19, %arg49_1, %arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_75 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_93,), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, %sigmoid_75), kwargs = {})
#   %convolution_94 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_246, %arg51_1, %arg52_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_76 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_94,), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_245, %sigmoid_76), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_22 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_22(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 144
    x2 = (xindex // 451584)
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


# kernel path: /tmp/torchinductor_sahanp/7j/c7jwg75lorzilgnsfm262vxutbx4wj34woskiu4lepq4wkauykck.py
# Topologically Sorted Source Nodes: [x_192, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_192 => add_124, mul_249, mul_250, sub_57
#   x_193 => add_125
# Graph fragment:
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_95, %unsqueeze_457), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_459), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_249, %unsqueeze_461), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_250, %unsqueeze_463), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_124, %add_118), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/b2/cb2iyfnbmz4skokv3ywckddcuqpb7xu63ldjjv5ectbx3t2yanzc.py
# Topologically Sorted Source Nodes: [x_196, x_197], Original ATen: [aten.silu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_196 => mul_254, sigmoid_77
#   x_197 => constant_pad_nd_7
# Graph fragment:
#   %sigmoid_77 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_127,), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_127, %sigmoid_77), kwargs = {})
#   %constant_pad_nd_7 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_254, [1, 2, 1, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_silu_24 = async_compile.triton('triton_poi_fused_constant_pad_nd_silu_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_silu_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4010112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 8496) % 59
    x1 = (xindex // 144) % 59
    x3 = (xindex // 501264)
    x4 = xindex % 8496
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-8208) + x4 + (8064*x2) + (451584*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fo/cfo4lsxrov6r4rgjnuirfg4y5flig2l6rtynaumqtegxvso262q2.py
# Topologically Sorted Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_199 => add_129, mul_256, mul_257, sub_59
# Graph fragment:
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_97, %unsqueeze_473), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_475), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %unsqueeze_477), kwargs = {})
#   %add_129 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %unsqueeze_479), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/rm/crm54r7hgmots2k5i66m6zeyb3at5h272i52bvcnnqhjvi5uj65j.py
# Topologically Sorted Source Nodes: [x_200, x_se_76], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_200 => mul_258, sigmoid_78
#   x_se_76 => mean_20
# Graph fragment:
#   %sigmoid_78 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_129,), kwargs = {})
#   %mul_258 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_129, %sigmoid_78), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_258, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_26 = async_compile.triton('triton_red_fused_mean_silu_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_26(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8064
    rnumel = 112
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
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (16128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/c2/cc23r23vwumgt63x73oh34m4k3gun7cckhmg6gx7fkwosahzvjkr.py
# Topologically Sorted Source Nodes: [x_200, x_se_76], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_200 => mul_258, sigmoid_78
#   x_se_76 => mean_20
# Graph fragment:
#   %sigmoid_78 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_129,), kwargs = {})
#   %mul_258 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_129, %sigmoid_78), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_258, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_27 = async_compile.triton('triton_per_fused_mean_silu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_27(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 7
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
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (1008*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wh/cwhvnonl2dvqi3bq3f4ctih7nyzaxrc5macsdcgtdt7fnhf35swr.py
# Topologically Sorted Source Nodes: [x_200, x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, x_201], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_19 => sigmoid_80
#   x_200 => mul_258, sigmoid_78
#   x_201 => mul_260
#   x_se_76 => mean_20
#   x_se_77 => convolution_98
#   x_se_78 => mul_259, sigmoid_79
#   x_se_79 => convolution_99
# Graph fragment:
#   %sigmoid_78 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_129,), kwargs = {})
#   %mul_258 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_129, %sigmoid_78), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_258, [2, 3], True), kwargs = {})
#   %convolution_98 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_20, %arg68_1, %arg69_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_79 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_98,), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_98, %sigmoid_79), kwargs = {})
#   %convolution_99 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_259, %arg70_1, %arg71_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_80 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_99,), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_258, %sigmoid_80), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_28 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_28(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 144
    x2 = (xindex // 112896)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + (144*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kw/ckwdi2rrw5u7snj4udrfyk4tv535jtiq3qxhfm4efnvcji4yrkzp.py
# Topologically Sorted Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_203 => add_131, mul_262, mul_263, sub_60
# Graph fragment:
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_481), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_483), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %unsqueeze_485), kwargs = {})
#   %add_131 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %unsqueeze_487), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/te/cteqbkjvqirumsrihvsnxy7t7p6ptisqbvfpapgn4c5mfav4hl7f.py
# Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_205 => add_133, mul_265, mul_266, sub_61
#   x_206 => mul_267, sigmoid_81
# Graph fragment:
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_101, %unsqueeze_489), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_265, %unsqueeze_493), kwargs = {})
#   %add_133 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_266, %unsqueeze_495), kwargs = {})
#   %sigmoid_81 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_133,), kwargs = {})
#   %mul_267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_133, %sigmoid_81), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hz/chzgemohtl6s3w7t2i5l54xlkzafnopaqmuejq73sjuvh63zgdiw.py
# Topologically Sorted Source Nodes: [x_208], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_208 => add_135, mul_269, mul_270, sub_62
# Graph fragment:
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_102, %unsqueeze_497), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %unsqueeze_499), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_269, %unsqueeze_501), kwargs = {})
#   %add_135 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_270, %unsqueeze_503), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/5m/c5mf4edhk7bbjdunndonnv2ymgri6z6vpsrrf3baq7ak2xy74oqc.py
# Topologically Sorted Source Nodes: [x_209, x_se_80], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_209 => mul_271, sigmoid_82
#   x_se_80 => mean_21
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_135,), kwargs = {})
#   %mul_271 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_135, %sigmoid_82), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_271, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_32 = async_compile.triton('triton_red_fused_mean_silu_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_32(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13440
    rnumel = 112
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
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (26880*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bz/cbzl4dpafwmdqzisvsm7y7ummigpoxmulpnqqlh7f6ruxec45df4.py
# Topologically Sorted Source Nodes: [x_209, x_se_80], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_209 => mul_271, sigmoid_82
#   x_se_80 => mean_21
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_135,), kwargs = {})
#   %mul_271 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_135, %sigmoid_82), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_271, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_33 = async_compile.triton('triton_per_fused_mean_silu_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_33(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 7
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
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (1680*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/o3/co3wmfcbtlmqsyok52h2obggx2drzxuc6z5qe67mwafe6znmpqan.py
# Topologically Sorted Source Nodes: [x_209, x_se_80, x_se_81, x_se_82], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_209 => mul_271, sigmoid_82
#   x_se_80 => mean_21
#   x_se_81 => convolution_103
#   x_se_82 => mul_272, sigmoid_83
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_135,), kwargs = {})
#   %mul_271 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_135, %sigmoid_82), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_271, [2, 3], True), kwargs = {})
#   %convolution_103 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_21, %arg87_1, %arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_103,), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_103, %sigmoid_83), kwargs = {})
triton_poi_fused_convolution_mean_silu_34 = async_compile.triton('triton_poi_fused_convolution_mean_silu_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_34(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ms/cmstxxsqtiqioona4bbrple4s2psdwiojxrhpsdvv4rvnsyppy7a.py
# Topologically Sorted Source Nodes: [x_209, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, x_210], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_20 => sigmoid_84
#   x_209 => mul_271, sigmoid_82
#   x_210 => mul_273
#   x_se_80 => mean_21
#   x_se_81 => convolution_103
#   x_se_82 => mul_272, sigmoid_83
#   x_se_83 => convolution_104
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_135,), kwargs = {})
#   %mul_271 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_135, %sigmoid_82), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_271, [2, 3], True), kwargs = {})
#   %convolution_103 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_21, %arg87_1, %arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_103,), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_103, %sigmoid_83), kwargs = {})
#   %convolution_104 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_272, %arg89_1, %arg90_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_84 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_104,), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %sigmoid_84), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_35 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_35(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 240
    x2 = (xindex // 188160)
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


# kernel path: /tmp/torchinductor_sahanp/pw/cpwtbvesqt3w3hqrbzhde5wg2v5d5zjzxyk4sffv5hpxx2vivmdz.py
# Topologically Sorted Source Nodes: [x_212, x_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_212 => add_137, mul_275, mul_276, sub_63
#   x_213 => add_138
# Graph fragment:
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_505), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_275, %unsqueeze_509), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_276, %unsqueeze_511), kwargs = {})
#   %add_138 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_137, %add_131), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/ci/cciptasib45p3pfhaipp5hhxxapouar4syxcyzpssvtynwb6v7rc.py
# Topologically Sorted Source Nodes: [x_216, x_217], Original ATen: [aten.silu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_216 => mul_280, sigmoid_85
#   x_217 => constant_pad_nd_8
# Graph fragment:
#   %sigmoid_85 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_140,), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_140, %sigmoid_85), kwargs = {})
#   %constant_pad_nd_8 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_280, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_silu_37 = async_compile.triton('triton_poi_fused_constant_pad_nd_silu_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_silu_37(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1614720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 6960) % 29
    x1 = (xindex // 240) % 29
    x3 = (xindex // 201840)
    x4 = xindex % 6960
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 28, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (6720*x2) + (188160*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x5), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ym/cymc2sdmcjx5fp5cble56ykhzb75qu2xb6d2c5d7o4obbg7zr26t.py
# Topologically Sorted Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_219 => add_142, mul_282, mul_283, sub_65
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_107, %unsqueeze_521), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_282, %unsqueeze_525), kwargs = {})
#   %add_142 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_283, %unsqueeze_527), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/kl/cklfp4pvro2f4nvsq2qyg77nz5hlqc4temboin3skpiqwaczdlvv.py
# Topologically Sorted Source Nodes: [x_220, x_se_84], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_220 => mul_284, sigmoid_86
#   x_se_84 => mean_22
# Graph fragment:
#   %sigmoid_86 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_142,), kwargs = {})
#   %mul_284 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_142, %sigmoid_86), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_284, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_39 = async_compile.triton('triton_red_fused_mean_silu_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_39(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (23520*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d5/cd5epryeulqwrnglrw6a6n5gmhkd2r54xgpnzxkjcrscwmpdzc2u.py
# Topologically Sorted Source Nodes: [x_220, x_se_84], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_220 => mul_284, sigmoid_86
#   x_se_84 => mean_22
# Graph fragment:
#   %sigmoid_86 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_142,), kwargs = {})
#   %mul_284 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_142, %sigmoid_86), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_284, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_40 = async_compile.triton('triton_per_fused_mean_silu_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_40(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/x2/cx2e5wbqow5gnwaclyr3moxep2zxxk4fscvv2pb7vl3lf675q6np.py
# Topologically Sorted Source Nodes: [x_220, x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, x_221], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_21 => sigmoid_88
#   x_220 => mul_284, sigmoid_86
#   x_221 => mul_286
#   x_se_84 => mean_22
#   x_se_85 => convolution_108
#   x_se_86 => mul_285, sigmoid_87
#   x_se_87 => convolution_109
# Graph fragment:
#   %sigmoid_86 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_142,), kwargs = {})
#   %mul_284 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_142, %sigmoid_86), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_284, [2, 3], True), kwargs = {})
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_22, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_87 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_108,), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, %sigmoid_87), kwargs = {})
#   %convolution_109 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_285, %arg108_1, %arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_88 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_109,), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_284, %sigmoid_88), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_41 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_41(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 240
    x2 = (xindex // 47040)
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


# kernel path: /tmp/torchinductor_sahanp/dn/cdn3xrmnvdr7kqbfdifeaoaxpj5y5ahqkn5zu6gbwhbf6mp37zl5.py
# Topologically Sorted Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_223 => add_144, mul_288, mul_289, sub_66
# Graph fragment:
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_110, %unsqueeze_529), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_288, %unsqueeze_533), kwargs = {})
#   %add_144 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_289, %unsqueeze_535), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_42', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/65/c65euxso5gjr5kxr47hzm3pso2wc43cadzs7ubdqpgyyr4q6wtxz.py
# Topologically Sorted Source Nodes: [x_225, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_225 => add_146, mul_291, mul_292, sub_67
#   x_226 => mul_293, sigmoid_89
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_111, %unsqueeze_537), kwargs = {})
#   %mul_291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_292 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_291, %unsqueeze_541), kwargs = {})
#   %add_146 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_292, %unsqueeze_543), kwargs = {})
#   %sigmoid_89 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_146,), kwargs = {})
#   %mul_293 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_146, %sigmoid_89), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yq/cyqluicsknzeqk2fzwxk23d3tdkta53nmqcu73xva5watdeghar7.py
# Topologically Sorted Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_228 => add_148, mul_295, mul_296, sub_68
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_112, %unsqueeze_545), kwargs = {})
#   %mul_295 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_295, %unsqueeze_549), kwargs = {})
#   %add_148 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_296, %unsqueeze_551), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/tn/ctn2rrj3nbyg547viqpziczaf5mbapev5xthzdbincfvu4jbdaus.py
# Topologically Sorted Source Nodes: [x_229, x_se_88], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_229 => mul_297, sigmoid_90
#   x_se_88 => mean_23
# Graph fragment:
#   %sigmoid_90 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_148,), kwargs = {})
#   %mul_297 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_148, %sigmoid_90), kwargs = {})
#   %mean_23 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_297, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_45 = async_compile.triton('triton_red_fused_mean_silu_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_45(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (47040*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/a4/ca4ukyd2dccw2capdipjfyn4uc4pgsz22ngwba45dyvo5zf75g3u.py
# Topologically Sorted Source Nodes: [x_229, x_se_88], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_229 => mul_297, sigmoid_90
#   x_se_88 => mean_23
# Graph fragment:
#   %sigmoid_90 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_148,), kwargs = {})
#   %mul_297 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_148, %sigmoid_90), kwargs = {})
#   %mean_23 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_297, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_46 = async_compile.triton('triton_per_fused_mean_silu_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_46(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vg/cvgfp6rn2534tn3knbadjjmtjyvepshp6yyxp6u5o437jnjfglks.py
# Topologically Sorted Source Nodes: [x_229, x_se_88, x_se_89, x_se_90], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_229 => mul_297, sigmoid_90
#   x_se_88 => mean_23
#   x_se_89 => convolution_113
#   x_se_90 => mul_298, sigmoid_91
# Graph fragment:
#   %sigmoid_90 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_148,), kwargs = {})
#   %mul_297 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_148, %sigmoid_90), kwargs = {})
#   %mean_23 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_297, [2, 3], True), kwargs = {})
#   %convolution_113 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_23, %arg125_1, %arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_91 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_113,), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_113, %sigmoid_91), kwargs = {})
triton_poi_fused_convolution_mean_silu_47 = async_compile.triton('triton_poi_fused_convolution_mean_silu_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_47(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/o4/co4omc26ihbu4bb3tkkmvcvbszkgvaovdfpzbrv37x5qibduxhtp.py
# Topologically Sorted Source Nodes: [x_229, x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, x_230], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_22 => sigmoid_92
#   x_229 => mul_297, sigmoid_90
#   x_230 => mul_299
#   x_se_88 => mean_23
#   x_se_89 => convolution_113
#   x_se_90 => mul_298, sigmoid_91
#   x_se_91 => convolution_114
# Graph fragment:
#   %sigmoid_90 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_148,), kwargs = {})
#   %mul_297 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_148, %sigmoid_90), kwargs = {})
#   %mean_23 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_297, [2, 3], True), kwargs = {})
#   %convolution_113 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_23, %arg125_1, %arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_91 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_113,), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_113, %sigmoid_91), kwargs = {})
#   %convolution_114 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_298, %arg127_1, %arg128_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_92 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_114,), kwargs = {})
#   %mul_299 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_297, %sigmoid_92), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_48 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_48(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 480
    x2 = (xindex // 94080)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + (480*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bz/cbzrbuewu7lq7qvfido42yy5nt5je2ks2dgqqphpcpgrtyld2cfo.py
# Topologically Sorted Source Nodes: [x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_232 => add_150, mul_301, mul_302, sub_69
#   x_233 => add_151
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_553), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_301, %unsqueeze_557), kwargs = {})
#   %add_150 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_302, %unsqueeze_559), kwargs = {})
#   %add_151 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_150, %add_144), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/sb/csbk7gctp5qdrao23rdaxcbk3xp6bcdz6ttwmfo2hpuqvbht6z42.py
# Topologically Sorted Source Nodes: [x_242, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_242 => add_157, mul_314, mul_315, sub_72
#   x_243 => add_158
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_120, %unsqueeze_577), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_314, %unsqueeze_581), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_315, %unsqueeze_583), kwargs = {})
#   %add_158 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_157, %add_151), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/i6/ci6fy7n7nx6epccqbt77xxu2a7tunqioyokk2rbjwce5oejpkhi2.py
# Topologically Sorted Source Nodes: [x_252], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_252 => add_164, mul_327, mul_328, sub_75
# Graph fragment:
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_125, %unsqueeze_601), kwargs = {})
#   %mul_327 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_75, %unsqueeze_603), kwargs = {})
#   %mul_328 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_327, %unsqueeze_605), kwargs = {})
#   %add_164 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_328, %unsqueeze_607), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/gy/cgy7pcnuf4i5idknfai2axdmc3gsmklae3adsceqazazmjqbm7qg.py
# Topologically Sorted Source Nodes: [x_254, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_254 => add_166, mul_330, mul_331, sub_76
#   x_255 => mul_332, sigmoid_101
# Graph fragment:
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_126, %unsqueeze_609), kwargs = {})
#   %mul_330 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %unsqueeze_611), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_330, %unsqueeze_613), kwargs = {})
#   %add_166 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_331, %unsqueeze_615), kwargs = {})
#   %sigmoid_101 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_166,), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_166, %sigmoid_101), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_52(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
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
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/le/clevenlhguawtjhn4lglxw3juucamxy36nsd3ajqfwjdhwq6lsj4.py
# Topologically Sorted Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_257 => add_168, mul_334, mul_335, sub_77
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_127, %unsqueeze_617), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_334, %unsqueeze_621), kwargs = {})
#   %add_168 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_335, %unsqueeze_623), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_53 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_53(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/7u/c7ujon2xknuzkf6xasy65yae6zdlvxnc6ob4kzait55qsshnjqkf.py
# Topologically Sorted Source Nodes: [x_258, x_se_100], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_258 => mul_336, sigmoid_102
#   x_se_100 => mean_26
# Graph fragment:
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_168,), kwargs = {})
#   %mul_336 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_168, %sigmoid_102), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_336, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_54 = async_compile.triton('triton_red_fused_mean_silu_54', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_54(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kr/ckr5exxirwk22afnkirnll3dn5eoio5el4wl6ptpnq263gtos65a.py
# Topologically Sorted Source Nodes: [x_258, x_se_100], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_258 => mul_336, sigmoid_102
#   x_se_100 => mean_26
# Graph fragment:
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_168,), kwargs = {})
#   %mul_336 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_168, %sigmoid_102), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_336, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_55 = async_compile.triton('triton_per_fused_mean_silu_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_55(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g3/cg3j7lvfm55lus4ve2hxdgxnl4ru4iecp7tk6v5hgoypuhpye7dm.py
# Topologically Sorted Source Nodes: [x_258, x_se_100, x_se_101, x_se_102], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_258 => mul_336, sigmoid_102
#   x_se_100 => mean_26
#   x_se_101 => convolution_128
#   x_se_102 => mul_337, sigmoid_103
# Graph fragment:
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_168,), kwargs = {})
#   %mul_336 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_168, %sigmoid_102), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_336, [2, 3], True), kwargs = {})
#   %convolution_128 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_26, %arg182_1, %arg183_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_103 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_128,), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_128, %sigmoid_103), kwargs = {})
triton_poi_fused_convolution_mean_silu_56 = async_compile.triton('triton_poi_fused_convolution_mean_silu_56', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_56(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/hw/chwax3sewkthjx5ullrl3vmkhuuluengbi4uotdddix53hsbd5ms.py
# Topologically Sorted Source Nodes: [x_258, x_se_100, x_se_101, x_se_102, x_se_103, sigmoid_25, x_259], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_25 => sigmoid_104
#   x_258 => mul_336, sigmoid_102
#   x_259 => mul_338
#   x_se_100 => mean_26
#   x_se_101 => convolution_128
#   x_se_102 => mul_337, sigmoid_103
#   x_se_103 => convolution_129
# Graph fragment:
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_168,), kwargs = {})
#   %mul_336 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_168, %sigmoid_102), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_336, [2, 3], True), kwargs = {})
#   %convolution_128 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_26, %arg182_1, %arg183_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_103 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_128,), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_128, %sigmoid_103), kwargs = {})
#   %convolution_129 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_337, %arg184_1, %arg185_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_104 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_129,), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_336, %sigmoid_104), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_57 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_57', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_57', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_57(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 131712)
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


# kernel path: /tmp/torchinductor_sahanp/c3/cc3gzqinjjcwobywtr3fmgblmloj3aetkj57u3k5c2blwrun7uqd.py
# Topologically Sorted Source Nodes: [x_261, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_261 => add_170, mul_340, mul_341, sub_78
#   x_262 => add_171
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_130, %unsqueeze_625), kwargs = {})
#   %mul_340 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_341 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_340, %unsqueeze_629), kwargs = {})
#   %add_170 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_341, %unsqueeze_631), kwargs = {})
#   %add_171 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_170, %add_164), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_58 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_58', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/wm/cwmblam57yuqq6mtovprkxwqo6yfuk6aha6dmxukgx4uk3z6ka6p.py
# Topologically Sorted Source Nodes: [x_275, x_276], Original ATen: [aten.silu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_275 => mul_358, sigmoid_109
#   x_276 => constant_pad_nd_9
# Graph fragment:
#   %sigmoid_109 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_180,), kwargs = {})
#   %mul_358 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_180, %sigmoid_109), kwargs = {})
#   %constant_pad_nd_9 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_358, [1, 2, 1, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_silu_59 = async_compile.triton('triton_poi_fused_constant_pad_nd_silu_59', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_silu_59(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1553664
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 11424) % 17
    x1 = (xindex // 672) % 17
    x3 = (xindex // 194208)
    x4 = xindex % 11424
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-10080) + x4 + (9408*x2) + (131712*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/u6/cu6v3ahbgwdytvbvun7dpsivxjleldsten25xs6ra6stos2z2xpw.py
# Topologically Sorted Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_278 => add_182, mul_360, mul_361, sub_83
# Graph fragment:
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_137, %unsqueeze_665), kwargs = {})
#   %mul_360 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_667), kwargs = {})
#   %mul_361 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_360, %unsqueeze_669), kwargs = {})
#   %add_182 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_361, %unsqueeze_671), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_60 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_60', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_60', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_60(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/ko/cko53m7uqfw4tm2zkzh55y6ott3r4ierrsalazqhmgdas6jauwmj.py
# Topologically Sorted Source Nodes: [x_279, x_se_108], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_279 => mul_362, sigmoid_110
#   x_se_108 => mean_28
# Graph fragment:
#   %sigmoid_110 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_182,), kwargs = {})
#   %mul_362 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_182, %sigmoid_110), kwargs = {})
#   %mean_28 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_362, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_61 = async_compile.triton('triton_per_fused_mean_silu_61', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_61(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tk/ctkl7iwituh7ta4u25b6o7t6juuewfkvnnme7z3sievylctjtq26.py
# Topologically Sorted Source Nodes: [x_279, x_se_108, x_se_109, x_se_110, x_se_111, sigmoid_27, x_280], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_27 => sigmoid_112
#   x_279 => mul_362, sigmoid_110
#   x_280 => mul_364
#   x_se_108 => mean_28
#   x_se_109 => convolution_138
#   x_se_110 => mul_363, sigmoid_111
#   x_se_111 => convolution_139
# Graph fragment:
#   %sigmoid_110 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_182,), kwargs = {})
#   %mul_362 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_182, %sigmoid_110), kwargs = {})
#   %mean_28 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_362, [2, 3], True), kwargs = {})
#   %convolution_138 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_28, %arg220_1, %arg221_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_111 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_138,), kwargs = {})
#   %mul_363 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_138, %sigmoid_111), kwargs = {})
#   %convolution_139 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_363, %arg222_1, %arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_112 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_139,), kwargs = {})
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_362, %sigmoid_112), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_62 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_62', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_62(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 263424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 32928)
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


# kernel path: /tmp/torchinductor_sahanp/eo/ceoke64b7fqgio5kxw7bo72ar6u37rvohlhlz75xl2uelnty7g7g.py
# Topologically Sorted Source Nodes: [x_282], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_282 => add_184, mul_366, mul_367, sub_84
# Graph fragment:
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_140, %unsqueeze_673), kwargs = {})
#   %mul_366 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_675), kwargs = {})
#   %mul_367 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_366, %unsqueeze_677), kwargs = {})
#   %add_184 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_367, %unsqueeze_679), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_63 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_63', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_63', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_63(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/4b/c4b332dvsjlkyybt2l4tqsdfwyxkhaiwsbiekwbuvkc4mdpeonlm.py
# Topologically Sorted Source Nodes: [x_284, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_284 => add_186, mul_369, mul_370, sub_85
#   x_285 => mul_371, sigmoid_113
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_141, %unsqueeze_681), kwargs = {})
#   %mul_369 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_369, %unsqueeze_685), kwargs = {})
#   %add_186 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_370, %unsqueeze_687), kwargs = {})
#   %sigmoid_113 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_186,), kwargs = {})
#   %mul_371 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_186, %sigmoid_113), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_64 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_64', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_64', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_64(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wb/cwbyhteqrn3ud45avtckthcq53r2rctmbmnxnuhlmfgvfvzzj7st.py
# Topologically Sorted Source Nodes: [x_287], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_287 => add_188, mul_373, mul_374, sub_86
# Graph fragment:
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_142, %unsqueeze_689), kwargs = {})
#   %mul_373 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_86, %unsqueeze_691), kwargs = {})
#   %mul_374 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_373, %unsqueeze_693), kwargs = {})
#   %add_188 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_374, %unsqueeze_695), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_65 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_65', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_65(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/qh/cqhaaxucrovuqctupgdhecopmdf7cycfor573g7wtpcjovcubxg6.py
# Topologically Sorted Source Nodes: [x_288, x_se_112], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_288 => mul_375, sigmoid_114
#   x_se_112 => mean_29
# Graph fragment:
#   %sigmoid_114 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_188,), kwargs = {})
#   %mul_375 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_188, %sigmoid_114), kwargs = {})
#   %mean_29 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_375, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_66 = async_compile.triton('triton_per_fused_mean_silu_66', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_66(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r2) + (56448*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n6/cn63wpuiyb4cw6vb5oilgwnwwrtwsax6jvrldjr6k3ek6klmhplj.py
# Topologically Sorted Source Nodes: [x_288, x_se_112, x_se_113, x_se_114], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_288 => mul_375, sigmoid_114
#   x_se_112 => mean_29
#   x_se_113 => convolution_143
#   x_se_114 => mul_376, sigmoid_115
# Graph fragment:
#   %sigmoid_114 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_188,), kwargs = {})
#   %mul_375 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_188, %sigmoid_114), kwargs = {})
#   %mean_29 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_375, [2, 3], True), kwargs = {})
#   %convolution_143 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_29, %arg239_1, %arg240_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_115 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_143,), kwargs = {})
#   %mul_376 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, %sigmoid_115), kwargs = {})
triton_poi_fused_convolution_mean_silu_67 = async_compile.triton('triton_poi_fused_convolution_mean_silu_67', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_67', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_67(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/2l/c2l3gnhc6fib5etaiv265a53c7eyaicyo27sxqb37v6mk3bhvdgc.py
# Topologically Sorted Source Nodes: [x_288, x_se_112, x_se_113, x_se_114, x_se_115, sigmoid_28, x_289], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_28 => sigmoid_116
#   x_288 => mul_375, sigmoid_114
#   x_289 => mul_377
#   x_se_112 => mean_29
#   x_se_113 => convolution_143
#   x_se_114 => mul_376, sigmoid_115
#   x_se_115 => convolution_144
# Graph fragment:
#   %sigmoid_114 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_188,), kwargs = {})
#   %mul_375 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_188, %sigmoid_114), kwargs = {})
#   %mean_29 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_375, [2, 3], True), kwargs = {})
#   %convolution_143 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_29, %arg239_1, %arg240_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_115 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_143,), kwargs = {})
#   %mul_376 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, %sigmoid_115), kwargs = {})
#   %convolution_144 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_376, %arg241_1, %arg242_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_116 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_144,), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_375, %sigmoid_116), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_68 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_68', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_68', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_68(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1152
    x2 = (xindex // 56448)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + (1152*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ed/ced6lxktndy2czj45ty2jw4rhkcxfbno7662zwvf36dq2qna57cn.py
# Topologically Sorted Source Nodes: [x_291, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_291 => add_190, mul_379, mul_380, sub_87
#   x_292 => add_191
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_145, %unsqueeze_697), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_379, %unsqueeze_701), kwargs = {})
#   %add_190 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_380, %unsqueeze_703), kwargs = {})
#   %add_191 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_190, %add_184), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_69 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_69', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_69', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_69(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/ph/cphq6gykcs5etjfbzehj5fujqoywmcbibcvvveyy7em7hp46kwap.py
# Topologically Sorted Source Nodes: [x_321], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_321 => add_211, mul_418, mul_419, sub_96
# Graph fragment:
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_160, %unsqueeze_769), kwargs = {})
#   %mul_418 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %unsqueeze_771), kwargs = {})
#   %mul_419 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_418, %unsqueeze_773), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_419, %unsqueeze_775), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_70 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_70', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_70', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_70(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/xj/cxjtvhftbrlmmnntu2wj5wvrxlav5zd3r32wpduzvhnclogzrlre.py
# Topologically Sorted Source Nodes: [x_323], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_323 => add_213, mul_421, mul_422, sub_97
# Graph fragment:
#   %sub_97 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_161, %unsqueeze_777), kwargs = {})
#   %mul_421 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_97, %unsqueeze_779), kwargs = {})
#   %mul_422 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_421, %unsqueeze_781), kwargs = {})
#   %add_213 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_422, %unsqueeze_783), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_71 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_71', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_71', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_71(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_sahanp/tx/ctxdw7glgtbabnvbvezra6aq2fxy47udwmwuuztojzgyes4rqply.py
# Topologically Sorted Source Nodes: [x_324, x_325], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_324 => mul_423, sigmoid_129
#   x_325 => mean_33
# Graph fragment:
#   %sigmoid_129 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_213,), kwargs = {})
#   %mul_423 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_213, %sigmoid_129), kwargs = {})
#   %mean_33 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_423, [-1, -2], True), kwargs = {})
triton_per_fused_mean_silu_72 = async_compile.triton('triton_per_fused_mean_silu_72', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_72', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_72(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1 = args
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
    assert_size_stride(arg158_1, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg159_1, (480, ), (1, ))
    assert_size_stride(arg160_1, (480, ), (1, ))
    assert_size_stride(arg161_1, (480, ), (1, ))
    assert_size_stride(arg162_1, (480, ), (1, ))
    assert_size_stride(arg163_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg164_1, (20, ), (1, ))
    assert_size_stride(arg165_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg166_1, (480, ), (1, ))
    assert_size_stride(arg167_1, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg168_1, (112, ), (1, ))
    assert_size_stride(arg169_1, (112, ), (1, ))
    assert_size_stride(arg170_1, (112, ), (1, ))
    assert_size_stride(arg171_1, (112, ), (1, ))
    assert_size_stride(arg172_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg173_1, (672, ), (1, ))
    assert_size_stride(arg174_1, (672, ), (1, ))
    assert_size_stride(arg175_1, (672, ), (1, ))
    assert_size_stride(arg176_1, (672, ), (1, ))
    assert_size_stride(arg177_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg178_1, (672, ), (1, ))
    assert_size_stride(arg179_1, (672, ), (1, ))
    assert_size_stride(arg180_1, (672, ), (1, ))
    assert_size_stride(arg181_1, (672, ), (1, ))
    assert_size_stride(arg182_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg183_1, (28, ), (1, ))
    assert_size_stride(arg184_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg185_1, (672, ), (1, ))
    assert_size_stride(arg186_1, (112, 672, 1, 1), (672, 1, 1, 1))
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
    assert_size_stride(arg224_1, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg225_1, (192, ), (1, ))
    assert_size_stride(arg226_1, (192, ), (1, ))
    assert_size_stride(arg227_1, (192, ), (1, ))
    assert_size_stride(arg228_1, (192, ), (1, ))
    assert_size_stride(arg229_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg230_1, (1152, ), (1, ))
    assert_size_stride(arg231_1, (1152, ), (1, ))
    assert_size_stride(arg232_1, (1152, ), (1, ))
    assert_size_stride(arg233_1, (1152, ), (1, ))
    assert_size_stride(arg234_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg235_1, (1152, ), (1, ))
    assert_size_stride(arg236_1, (1152, ), (1, ))
    assert_size_stride(arg237_1, (1152, ), (1, ))
    assert_size_stride(arg238_1, (1152, ), (1, ))
    assert_size_stride(arg239_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg240_1, (48, ), (1, ))
    assert_size_stride(arg241_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg242_1, (1152, ), (1, ))
    assert_size_stride(arg243_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg244_1, (192, ), (1, ))
    assert_size_stride(arg245_1, (192, ), (1, ))
    assert_size_stride(arg246_1, (192, ), (1, ))
    assert_size_stride(arg247_1, (192, ), (1, ))
    assert_size_stride(arg248_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg249_1, (1152, ), (1, ))
    assert_size_stride(arg250_1, (1152, ), (1, ))
    assert_size_stride(arg251_1, (1152, ), (1, ))
    assert_size_stride(arg252_1, (1152, ), (1, ))
    assert_size_stride(arg253_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg254_1, (1152, ), (1, ))
    assert_size_stride(arg255_1, (1152, ), (1, ))
    assert_size_stride(arg256_1, (1152, ), (1, ))
    assert_size_stride(arg257_1, (1152, ), (1, ))
    assert_size_stride(arg258_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg259_1, (48, ), (1, ))
    assert_size_stride(arg260_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg261_1, (1152, ), (1, ))
    assert_size_stride(arg262_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
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
    assert_size_stride(arg291_1, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg292_1, (1152, ), (1, ))
    assert_size_stride(arg293_1, (1152, ), (1, ))
    assert_size_stride(arg294_1, (1152, ), (1, ))
    assert_size_stride(arg295_1, (1152, ), (1, ))
    assert_size_stride(arg296_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg297_1, (48, ), (1, ))
    assert_size_stride(arg298_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg299_1, (1152, ), (1, ))
    assert_size_stride(arg300_1, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg301_1, (320, ), (1, ))
    assert_size_stride(arg302_1, (320, ), (1, ))
    assert_size_stride(arg303_1, (320, ), (1, ))
    assert_size_stride(arg304_1, (320, ), (1, ))
    assert_size_stride(arg305_1, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg306_1, (1280, ), (1, ))
    assert_size_stride(arg307_1, (1280, ), (1, ))
    assert_size_stride(arg308_1, (1280, ), (1, ))
    assert_size_stride(arg309_1, (1280, ), (1, ))
    assert_size_stride(arg310_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg311_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 225, 225), (151875, 1, 675, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0.run(arg1_1, buf0, 24, 50625, grid=grid(24, 50625), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_164, x_165], Original ATen: [aten.constant_pad_nd, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_164, x_165], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((8, 32, 112, 112), (401408, 1, 3584, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_166, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, buf4, 3211264, grid=grid(3211264), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf3
        # Topologically Sorted Source Nodes: [x_167, x_168], Original ATen: [aten.silu, aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf5, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del arg6_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_169], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf7 = empty_strided_cuda((8, 32, 1, 1, 98), (3136, 1, 25088, 25088, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_170, x_se_64], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_4.run(buf6, buf7, 25088, 128, grid=grid(25088), stream=stream0)
        buf9 = empty_strided_cuda((8, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_170, x_se_64], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_5.run(buf7, buf9, 256, 98, grid=grid(256), stream=stream0)
        del buf7
        # Topologically Sorted Source Nodes: [x_170, x_se_64, x_se_65], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg11_1
        del buf9
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_170, x_se_64, x_se_65, x_se_66], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_6.run(buf11, arg12_1, 64, grid=grid(64), stream=stream0)
        del arg12_1
        # Topologically Sorted Source Nodes: [x_170, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg13_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg13_1
        del buf11
        buf13 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_170, x_se_64, x_se_65, x_se_66, x_se_67, sigmoid_16, x_171], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_7.run(buf13, buf12, arg14_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg14_1
        del buf12
        # Topologically Sorted Source Nodes: [x_170, x_se_64, x_se_65, x_se_66, x_se_67, sigmoid_16, x_171, x_172], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf14 = extern_kernels.convolution(buf13, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del arg15_1
        del buf13
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf15, arg16_1, arg17_1, arg18_1, arg19_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        del arg19_1
        # Topologically Sorted Source Nodes: [x_173, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg20_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 96, 112, 112), (1204224, 1, 10752, 96))
        del arg20_1
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_175], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_9.run(buf17, arg21_1, arg22_1, arg23_1, arg24_1, 9633792, grid=grid(9633792), stream=stream0)
        del arg21_1
        del arg22_1
        del arg23_1
        del arg24_1
        buf18 = empty_strided_cuda((8, 96, 113, 113), (1225824, 1, 10848, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_176, x_177], Original ATen: [aten.silu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_silu_10.run(buf17, buf18, 9806592, grid=grid(9806592), stream=stream0)
        del buf17
        # Topologically Sorted Source Nodes: [x_176, x_177, x_178], Original ATen: [aten.silu, aten.constant_pad_nd, aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg25_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf19, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del arg25_1
        del buf18
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_179], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf20, arg26_1, arg27_1, arg28_1, arg29_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg26_1
        del arg27_1
        del arg28_1
        del arg29_1
        buf21 = empty_strided_cuda((8, 96, 1, 1, 25), (2400, 1, 19200, 19200, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_180, x_se_68], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_12.run(buf20, buf21, 19200, 126, grid=grid(19200), stream=stream0)
        buf23 = empty_strided_cuda((8, 96, 1, 1), (96, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_180, x_se_68], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_13.run(buf21, buf23, 768, 25, grid=grid(768), stream=stream0)
        del buf21
        # Topologically Sorted Source Nodes: [x_180, x_se_68, x_se_69], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 4, 1, 1), (4, 1, 1, 1))
        del arg30_1
        del buf23
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_180, x_se_68, x_se_69, x_se_70], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_14.run(buf25, arg31_1, 32, grid=grid(32), stream=stream0)
        del arg31_1
        # Topologically Sorted Source Nodes: [x_180, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg32_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 96, 1, 1), (96, 1, 1, 1))
        del arg32_1
        del buf25
        buf27 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_180, x_se_68, x_se_69, x_se_70, x_se_71, sigmoid_17, x_181], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_15.run(buf27, buf26, arg33_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg33_1
        del buf26
        # Topologically Sorted Source Nodes: [x_180, x_se_68, x_se_69, x_se_70, x_se_71, sigmoid_17, x_181, x_182], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf28 = extern_kernels.convolution(buf27, arg34_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg34_1
        del buf27
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf29, arg35_1, arg36_1, arg37_1, arg38_1, 602112, grid=grid(602112), stream=stream0)
        del arg35_1
        del arg36_1
        del arg37_1
        del arg38_1
        # Topologically Sorted Source Nodes: [x_184], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 144, 56, 56), (451584, 1, 8064, 144))
        del arg39_1
        buf31 = buf30; del buf30  # reuse
        buf32 = empty_strided_cuda((8, 144, 56, 56), (451584, 1, 8064, 144), torch.float32)
        # Topologically Sorted Source Nodes: [x_185, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_17.run(buf31, arg40_1, arg41_1, arg42_1, arg43_1, buf32, 3612672, grid=grid(3612672), stream=stream0)
        del arg40_1
        del arg41_1
        del arg42_1
        del arg43_1
        del buf31
        # Topologically Sorted Source Nodes: [x_186, x_187], Original ATen: [aten.silu, aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg44_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf33, (8, 144, 56, 56), (451584, 1, 8064, 144))
        del arg44_1
        del buf32
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf34, arg45_1, arg46_1, arg47_1, arg48_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg45_1
        del arg46_1
        del arg47_1
        del arg48_1
        buf35 = empty_strided_cuda((8, 144, 1, 1, 25), (3600, 1, 28800, 28800, 144), torch.float32)
        # Topologically Sorted Source Nodes: [x_189, x_se_72], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_19.run(buf34, buf35, 28800, 126, grid=grid(28800), stream=stream0)
        buf37 = empty_strided_cuda((8, 144, 1, 1), (144, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_189, x_se_72], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_20.run(buf35, buf37, 1152, 25, grid=grid(1152), stream=stream0)
        del buf35
        # Topologically Sorted Source Nodes: [x_189, x_se_72, x_se_73], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg49_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 6, 1, 1), (6, 1, 1, 1))
        del arg49_1
        del buf37
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_189, x_se_72, x_se_73, x_se_74], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_21.run(buf39, arg50_1, 48, grid=grid(48), stream=stream0)
        del arg50_1
        # Topologically Sorted Source Nodes: [x_189, x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 144, 1, 1), (144, 1, 1, 1))
        del arg51_1
        del buf39
        buf41 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_189, x_se_72, x_se_73, x_se_74, x_se_75, sigmoid_18, x_190], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_22.run(buf41, buf40, arg52_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg52_1
        # Topologically Sorted Source Nodes: [x_189, x_se_72, x_se_73, x_se_74, x_se_75, sigmoid_18, x_190, x_191], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf42 = extern_kernels.convolution(buf41, arg53_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg53_1
        del buf41
        buf43 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_192, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf43, buf42, arg54_1, arg55_1, arg56_1, arg57_1, 602112, grid=grid(602112), stream=stream0)
        del arg54_1
        del arg55_1
        del arg56_1
        del arg57_1
        del buf42
        # Topologically Sorted Source Nodes: [x_192, x_193, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg58_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 144, 56, 56), (451584, 1, 8064, 144))
        del arg58_1
        del buf43
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [x_195], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf45, arg59_1, arg60_1, arg61_1, arg62_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg59_1
        del arg60_1
        del arg61_1
        del arg62_1
        buf46 = empty_strided_cuda((8, 144, 59, 59), (501264, 1, 8496, 144), torch.float32)
        # Topologically Sorted Source Nodes: [x_196, x_197], Original ATen: [aten.silu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_silu_24.run(buf45, buf46, 4010112, grid=grid(4010112), stream=stream0)
        del buf45
        # Topologically Sorted Source Nodes: [x_196, x_197, x_198], Original ATen: [aten.silu, aten.constant_pad_nd, aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg63_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf47, (8, 144, 28, 28), (112896, 1, 4032, 144))
        del arg63_1
        del buf46
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf48, arg64_1, arg65_1, arg66_1, arg67_1, 903168, grid=grid(903168), stream=stream0)
        del arg64_1
        del arg65_1
        del arg66_1
        del arg67_1
        buf49 = empty_strided_cuda((8, 144, 1, 1, 7), (1008, 1, 8064, 8064, 144), torch.float32)
        # Topologically Sorted Source Nodes: [x_200, x_se_76], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_26.run(buf48, buf49, 8064, 112, grid=grid(8064), stream=stream0)
        buf51 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_200, x_se_76], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_27.run(buf49, buf51, 1152, 7, grid=grid(1152), stream=stream0)
        del buf49
        # Topologically Sorted Source Nodes: [x_200, x_se_76, x_se_77], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 6, 1, 1), (6, 1, 1, 1))
        del arg68_1
        del buf51
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_200, x_se_76, x_se_77, x_se_78], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_21.run(buf53, arg69_1, 48, grid=grid(48), stream=stream0)
        del arg69_1
        # Topologically Sorted Source Nodes: [x_200, x_se_76, x_se_77, x_se_78, x_se_79], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg70_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 144, 1, 1), (144, 1, 1, 1))
        del arg70_1
        del buf53
        buf55 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_200, x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, x_201], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_28.run(buf55, buf54, arg71_1, 903168, grid=grid(903168), stream=stream0)
        del arg71_1
        del buf54
        # Topologically Sorted Source Nodes: [x_200, x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, x_201, x_202], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf56 = extern_kernels.convolution(buf55, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 40, 28, 28), (31360, 1, 1120, 40))
        del arg72_1
        del buf55
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_29.run(buf57, arg73_1, arg74_1, arg75_1, arg76_1, 250880, grid=grid(250880), stream=stream0)
        del arg73_1
        del arg74_1
        del arg75_1
        del arg76_1
        # Topologically Sorted Source Nodes: [x_204], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg77_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 240, 28, 28), (188160, 1, 6720, 240))
        del arg77_1
        buf59 = buf58; del buf58  # reuse
        buf60 = empty_strided_cuda((8, 240, 28, 28), (188160, 1, 6720, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_30.run(buf59, arg78_1, arg79_1, arg80_1, arg81_1, buf60, 1505280, grid=grid(1505280), stream=stream0)
        del arg78_1
        del arg79_1
        del arg80_1
        del arg81_1
        del buf59
        # Topologically Sorted Source Nodes: [x_206, x_207], Original ATen: [aten.silu, aten.convolution]
        buf61 = extern_kernels.convolution(buf60, arg82_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf61, (8, 240, 28, 28), (188160, 1, 6720, 240))
        del arg82_1
        del buf60
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_208], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf62, arg83_1, arg84_1, arg85_1, arg86_1, 1505280, grid=grid(1505280), stream=stream0)
        del arg83_1
        del arg84_1
        del arg85_1
        del arg86_1
        buf63 = empty_strided_cuda((8, 240, 1, 1, 7), (1680, 1, 13440, 13440, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_209, x_se_80], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_32.run(buf62, buf63, 13440, 112, grid=grid(13440), stream=stream0)
        buf65 = empty_strided_cuda((8, 240, 1, 1), (240, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_209, x_se_80], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_33.run(buf63, buf65, 1920, 7, grid=grid(1920), stream=stream0)
        del buf63
        # Topologically Sorted Source Nodes: [x_209, x_se_80, x_se_81], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 10, 1, 1), (10, 1, 1, 1))
        del arg87_1
        del buf65
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_209, x_se_80, x_se_81, x_se_82], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_34.run(buf67, arg88_1, 80, grid=grid(80), stream=stream0)
        del arg88_1
        # Topologically Sorted Source Nodes: [x_209, x_se_80, x_se_81, x_se_82, x_se_83], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg89_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg89_1
        del buf67
        buf69 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_209, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, x_210], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_35.run(buf69, buf68, arg90_1, 1505280, grid=grid(1505280), stream=stream0)
        del arg90_1
        # Topologically Sorted Source Nodes: [x_209, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, x_210, x_211], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf70 = extern_kernels.convolution(buf69, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 40, 28, 28), (31360, 1, 1120, 40))
        del arg91_1
        del buf69
        buf71 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_212, x_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_36.run(buf71, buf70, arg92_1, arg93_1, arg94_1, arg95_1, 250880, grid=grid(250880), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf70
        # Topologically Sorted Source Nodes: [x_212, x_213, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 240, 28, 28), (188160, 1, 6720, 240))
        del arg96_1
        del buf71
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_215], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf73, arg97_1, arg98_1, arg99_1, arg100_1, 1505280, grid=grid(1505280), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        buf74 = empty_strided_cuda((8, 240, 29, 29), (201840, 1, 6960, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_216, x_217], Original ATen: [aten.silu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_silu_37.run(buf73, buf74, 1614720, grid=grid(1614720), stream=stream0)
        del buf73
        # Topologically Sorted Source Nodes: [x_216, x_217, x_218], Original ATen: [aten.silu, aten.constant_pad_nd, aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg101_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf75, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg101_1
        del buf74
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf76, arg102_1, arg103_1, arg104_1, arg105_1, 376320, grid=grid(376320), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        buf77 = empty_strided_cuda((8, 240, 1, 1, 2), (480, 1, 3840, 3840, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_220, x_se_84], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_39.run(buf76, buf77, 3840, 98, grid=grid(3840), stream=stream0)
        buf79 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_220, x_se_84], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_40.run(buf77, buf79, 1920, 2, grid=grid(1920), stream=stream0)
        # Topologically Sorted Source Nodes: [x_220, x_se_84, x_se_85], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 10, 1, 1), (10, 1, 1, 1))
        del arg106_1
        del buf79
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_220, x_se_84, x_se_85, x_se_86], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_34.run(buf81, arg107_1, 80, grid=grid(80), stream=stream0)
        del arg107_1
        # Topologically Sorted Source Nodes: [x_220, x_se_84, x_se_85, x_se_86, x_se_87], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg108_1
        del buf81
        buf83 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_220, x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, x_221], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_41.run(buf83, buf82, arg109_1, 376320, grid=grid(376320), stream=stream0)
        del arg109_1
        del buf82
        # Topologically Sorted Source Nodes: [x_220, x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, x_221, x_222], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf84 = extern_kernels.convolution(buf83, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg110_1
        del buf83
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_42.run(buf85, arg111_1, arg112_1, arg113_1, arg114_1, 125440, grid=grid(125440), stream=stream0)
        del arg111_1
        del arg112_1
        del arg113_1
        del arg114_1
        # Topologically Sorted Source Nodes: [x_224], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 480, 14, 14), (94080, 1, 6720, 480))
        del arg115_1
        buf87 = buf86; del buf86  # reuse
        buf88 = empty_strided_cuda((8, 480, 14, 14), (94080, 1, 6720, 480), torch.float32)
        # Topologically Sorted Source Nodes: [x_225, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_43.run(buf87, arg116_1, arg117_1, arg118_1, arg119_1, buf88, 752640, grid=grid(752640), stream=stream0)
        del arg116_1
        del arg117_1
        del arg118_1
        del arg119_1
        del buf87
        # Topologically Sorted Source Nodes: [x_226, x_227], Original ATen: [aten.silu, aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg120_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf89, (8, 480, 14, 14), (94080, 1, 6720, 480))
        del arg120_1
        del buf88
        buf90 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_44.run(buf90, arg121_1, arg122_1, arg123_1, arg124_1, 752640, grid=grid(752640), stream=stream0)
        del arg121_1
        del arg122_1
        del arg123_1
        del arg124_1
        buf91 = empty_strided_cuda((8, 480, 1, 1, 2), (960, 1, 7680, 7680, 480), torch.float32)
        # Topologically Sorted Source Nodes: [x_229, x_se_88], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_45.run(buf90, buf91, 7680, 98, grid=grid(7680), stream=stream0)
        buf93 = reinterpret_tensor(buf77, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [x_229, x_se_88], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_46.run(buf91, buf93, 3840, 2, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [x_229, x_se_88, x_se_89], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg125_1
        del buf93
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_229, x_se_88, x_se_89, x_se_90], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_47.run(buf95, arg126_1, 160, grid=grid(160), stream=stream0)
        del arg126_1
        # Topologically Sorted Source Nodes: [x_229, x_se_88, x_se_89, x_se_90, x_se_91], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg127_1
        del buf95
        buf97 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_229, x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, x_230], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_48.run(buf97, buf96, arg128_1, 752640, grid=grid(752640), stream=stream0)
        del arg128_1
        # Topologically Sorted Source Nodes: [x_229, x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, x_230, x_231], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf98 = extern_kernels.convolution(buf97, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg129_1
        buf99 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_49.run(buf99, buf98, arg130_1, arg131_1, arg132_1, arg133_1, 125440, grid=grid(125440), stream=stream0)
        del arg130_1
        del arg131_1
        del arg132_1
        del arg133_1
        del buf98
        # Topologically Sorted Source Nodes: [x_234], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 480, 14, 14), (94080, 1, 6720, 480))
        del arg134_1
        buf101 = buf100; del buf100  # reuse
        buf102 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_43.run(buf101, arg135_1, arg136_1, arg137_1, arg138_1, buf102, 752640, grid=grid(752640), stream=stream0)
        del arg135_1
        del arg136_1
        del arg137_1
        del arg138_1
        del buf101
        # Topologically Sorted Source Nodes: [x_236, x_237], Original ATen: [aten.silu, aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg139_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf103, (8, 480, 14, 14), (94080, 1, 6720, 480))
        del arg139_1
        del buf102
        buf104 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_44.run(buf104, arg140_1, arg141_1, arg142_1, arg143_1, 752640, grid=grid(752640), stream=stream0)
        del arg140_1
        del arg141_1
        del arg142_1
        del arg143_1
        buf105 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_239, x_se_92], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_45.run(buf104, buf105, 7680, 98, grid=grid(7680), stream=stream0)
        buf107 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_239, x_se_92], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_46.run(buf105, buf107, 3840, 2, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [x_239, x_se_92, x_se_93], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg144_1
        del buf107
        buf109 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_239, x_se_92, x_se_93, x_se_94], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_47.run(buf109, arg145_1, 160, grid=grid(160), stream=stream0)
        del arg145_1
        # Topologically Sorted Source Nodes: [x_239, x_se_92, x_se_93, x_se_94, x_se_95], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg146_1
        del buf109
        buf111 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_239, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, x_240], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_48.run(buf111, buf110, arg147_1, 752640, grid=grid(752640), stream=stream0)
        del arg147_1
        # Topologically Sorted Source Nodes: [x_239, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, x_240, x_241], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf112 = extern_kernels.convolution(buf111, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg148_1
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_242, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf113, arg149_1, arg150_1, arg151_1, arg152_1, buf99, 125440, grid=grid(125440), stream=stream0)
        del arg149_1
        del arg150_1
        del arg151_1
        del arg152_1
        del buf99
        # Topologically Sorted Source Nodes: [x_242, x_243, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf114 = extern_kernels.convolution(buf113, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 480, 14, 14), (94080, 1, 6720, 480))
        del arg153_1
        del buf113
        buf115 = buf114; del buf114  # reuse
        buf116 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [x_245, x_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_43.run(buf115, arg154_1, arg155_1, arg156_1, arg157_1, buf116, 752640, grid=grid(752640), stream=stream0)
        del arg154_1
        del arg155_1
        del arg156_1
        del arg157_1
        del buf115
        # Topologically Sorted Source Nodes: [x_246, x_247], Original ATen: [aten.silu, aten.convolution]
        buf117 = extern_kernels.convolution(buf116, arg158_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf117, (8, 480, 14, 14), (94080, 1, 6720, 480))
        del arg158_1
        del buf116
        buf118 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_44.run(buf118, arg159_1, arg160_1, arg161_1, arg162_1, 752640, grid=grid(752640), stream=stream0)
        del arg159_1
        del arg160_1
        del arg161_1
        del arg162_1
        buf119 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_249, x_se_96], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_45.run(buf118, buf119, 7680, 98, grid=grid(7680), stream=stream0)
        buf121 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_249, x_se_96], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_46.run(buf119, buf121, 3840, 2, grid=grid(3840), stream=stream0)
        del buf119
        # Topologically Sorted Source Nodes: [x_249, x_se_96, x_se_97], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf122 = extern_kernels.convolution(buf121, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg163_1
        del buf121
        buf123 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_249, x_se_96, x_se_97, x_se_98], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_47.run(buf123, arg164_1, 160, grid=grid(160), stream=stream0)
        del arg164_1
        # Topologically Sorted Source Nodes: [x_249, x_se_96, x_se_97, x_se_98, x_se_99], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg165_1
        del buf123
        buf125 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_249, x_se_96, x_se_97, x_se_98, x_se_99, sigmoid_24, x_250], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_48.run(buf125, buf124, arg166_1, 752640, grid=grid(752640), stream=stream0)
        del arg166_1
        del buf124
        # Topologically Sorted Source Nodes: [x_249, x_se_96, x_se_97, x_se_98, x_se_99, sigmoid_24, x_250, x_251], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf126 = extern_kernels.convolution(buf125, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg167_1
        del buf125
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_252], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_51.run(buf127, arg168_1, arg169_1, arg170_1, arg171_1, 175616, grid=grid(175616), stream=stream0)
        del arg168_1
        del arg169_1
        del arg170_1
        del arg171_1
        # Topologically Sorted Source Nodes: [x_253], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg172_1
        buf129 = buf128; del buf128  # reuse
        buf130 = empty_strided_cuda((8, 672, 14, 14), (131712, 1, 9408, 672), torch.float32)
        # Topologically Sorted Source Nodes: [x_254, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_52.run(buf129, arg173_1, arg174_1, arg175_1, arg176_1, buf130, 1053696, grid=grid(1053696), stream=stream0)
        del arg173_1
        del arg174_1
        del arg175_1
        del arg176_1
        del buf129
        # Topologically Sorted Source Nodes: [x_255, x_256], Original ATen: [aten.silu, aten.convolution]
        buf131 = extern_kernels.convolution(buf130, arg177_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf131, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg177_1
        del buf130
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf132, arg178_1, arg179_1, arg180_1, arg181_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg178_1
        del arg179_1
        del arg180_1
        del arg181_1
        buf133 = empty_strided_cuda((8, 672, 1, 1, 2), (1344, 1, 10752, 10752, 672), torch.float32)
        # Topologically Sorted Source Nodes: [x_258, x_se_100], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_54.run(buf132, buf133, 10752, 98, grid=grid(10752), stream=stream0)
        buf135 = empty_strided_cuda((8, 672, 1, 1), (672, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_258, x_se_100], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_55.run(buf133, buf135, 5376, 2, grid=grid(5376), stream=stream0)
        # Topologically Sorted Source Nodes: [x_258, x_se_100, x_se_101], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg182_1
        del buf135
        buf137 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_258, x_se_100, x_se_101, x_se_102], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_56.run(buf137, arg183_1, 224, grid=grid(224), stream=stream0)
        del arg183_1
        # Topologically Sorted Source Nodes: [x_258, x_se_100, x_se_101, x_se_102, x_se_103], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf138 = extern_kernels.convolution(buf137, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg184_1
        del buf137
        buf139 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [x_258, x_se_100, x_se_101, x_se_102, x_se_103, sigmoid_25, x_259], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_57.run(buf139, buf138, arg185_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg185_1
        # Topologically Sorted Source Nodes: [x_258, x_se_100, x_se_101, x_se_102, x_se_103, sigmoid_25, x_259, x_260], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf140 = extern_kernels.convolution(buf139, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg186_1
        buf141 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_261, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_58.run(buf141, buf140, arg187_1, arg188_1, arg189_1, arg190_1, 175616, grid=grid(175616), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        del buf140
        # Topologically Sorted Source Nodes: [x_263], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg191_1
        buf143 = buf142; del buf142  # reuse
        buf144 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_264, x_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_52.run(buf143, arg192_1, arg193_1, arg194_1, arg195_1, buf144, 1053696, grid=grid(1053696), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        del buf143
        # Topologically Sorted Source Nodes: [x_265, x_266], Original ATen: [aten.silu, aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg196_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf145, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg196_1
        del buf144
        buf146 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf146, arg197_1, arg198_1, arg199_1, arg200_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        buf147 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [x_268, x_se_104], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_54.run(buf146, buf147, 10752, 98, grid=grid(10752), stream=stream0)
        buf149 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [x_268, x_se_104], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_55.run(buf147, buf149, 5376, 2, grid=grid(5376), stream=stream0)
        del buf147
        # Topologically Sorted Source Nodes: [x_268, x_se_104, x_se_105], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf150 = extern_kernels.convolution(buf149, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg201_1
        del buf149
        buf151 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [x_268, x_se_104, x_se_105, x_se_106], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_56.run(buf151, arg202_1, 224, grid=grid(224), stream=stream0)
        del arg202_1
        # Topologically Sorted Source Nodes: [x_268, x_se_104, x_se_105, x_se_106, x_se_107], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg203_1
        del buf151
        buf153 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_268, x_se_104, x_se_105, x_se_106, x_se_107, sigmoid_26, x_269], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_57.run(buf153, buf152, arg204_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg204_1
        # Topologically Sorted Source Nodes: [x_268, x_se_104, x_se_105, x_se_106, x_se_107, sigmoid_26, x_269, x_270], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf154 = extern_kernels.convolution(buf153, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg205_1
        del buf153
        buf155 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [x_271, x_272], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_58.run(buf155, buf154, arg206_1, arg207_1, arg208_1, arg209_1, 175616, grid=grid(175616), stream=stream0)
        del arg206_1
        del arg207_1
        del arg208_1
        del arg209_1
        del buf154
        # Topologically Sorted Source Nodes: [x_271, x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf156 = extern_kernels.convolution(buf155, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg210_1
        del buf155
        buf157 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [x_274], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf157, arg211_1, arg212_1, arg213_1, arg214_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg211_1
        del arg212_1
        del arg213_1
        del arg214_1
        buf158 = empty_strided_cuda((8, 672, 17, 17), (194208, 1, 11424, 672), torch.float32)
        # Topologically Sorted Source Nodes: [x_275, x_276], Original ATen: [aten.silu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_silu_59.run(buf157, buf158, 1553664, grid=grid(1553664), stream=stream0)
        del buf157
        # Topologically Sorted Source Nodes: [x_275, x_276, x_277], Original ATen: [aten.silu, aten.constant_pad_nd, aten.convolution]
        buf159 = extern_kernels.convolution(buf158, arg215_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf159, (8, 672, 7, 7), (32928, 1, 4704, 672))
        del arg215_1
        del buf158
        buf160 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_60.run(buf160, arg216_1, arg217_1, arg218_1, arg219_1, 263424, grid=grid(263424), stream=stream0)
        del arg216_1
        del arg217_1
        del arg218_1
        del arg219_1
        buf162 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [x_279, x_se_108], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_61.run(buf160, buf162, 5376, 49, grid=grid(5376), stream=stream0)
        # Topologically Sorted Source Nodes: [x_279, x_se_108, x_se_109], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf163 = extern_kernels.convolution(buf162, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg220_1
        del buf162
        buf164 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_279, x_se_108, x_se_109, x_se_110], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_56.run(buf164, arg221_1, 224, grid=grid(224), stream=stream0)
        del arg221_1
        # Topologically Sorted Source Nodes: [x_279, x_se_108, x_se_109, x_se_110, x_se_111], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf165 = extern_kernels.convolution(buf164, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg222_1
        del buf164
        buf166 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_279, x_se_108, x_se_109, x_se_110, x_se_111, sigmoid_27, x_280], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_62.run(buf166, buf165, arg223_1, 263424, grid=grid(263424), stream=stream0)
        del arg223_1
        del buf165
        # Topologically Sorted Source Nodes: [x_279, x_se_108, x_se_109, x_se_110, x_se_111, sigmoid_27, x_280, x_281], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf167 = extern_kernels.convolution(buf166, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 192, 7, 7), (9408, 1, 1344, 192))
        del arg224_1
        del buf166
        buf168 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_282], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_63.run(buf168, arg225_1, arg226_1, arg227_1, arg228_1, 75264, grid=grid(75264), stream=stream0)
        del arg225_1
        del arg226_1
        del arg227_1
        del arg228_1
        # Topologically Sorted Source Nodes: [x_283], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, arg229_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg229_1
        buf170 = buf169; del buf169  # reuse
        buf171 = empty_strided_cuda((8, 1152, 7, 7), (56448, 1, 8064, 1152), torch.float32)
        # Topologically Sorted Source Nodes: [x_284, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_64.run(buf170, arg230_1, arg231_1, arg232_1, arg233_1, buf171, 451584, grid=grid(451584), stream=stream0)
        del arg230_1
        del arg231_1
        del arg232_1
        del arg233_1
        del buf170
        # Topologically Sorted Source Nodes: [x_285, x_286], Original ATen: [aten.silu, aten.convolution]
        buf172 = extern_kernels.convolution(buf171, arg234_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf172, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg234_1
        del buf171
        buf173 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [x_287], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_65.run(buf173, arg235_1, arg236_1, arg237_1, arg238_1, 451584, grid=grid(451584), stream=stream0)
        del arg235_1
        del arg236_1
        del arg237_1
        del arg238_1
        buf175 = empty_strided_cuda((8, 1152, 1, 1), (1152, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_288, x_se_112], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_66.run(buf173, buf175, 9216, 49, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_288, x_se_112, x_se_113], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg239_1
        del buf175
        buf177 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [x_288, x_se_112, x_se_113, x_se_114], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_67.run(buf177, arg240_1, 384, grid=grid(384), stream=stream0)
        del arg240_1
        # Topologically Sorted Source Nodes: [x_288, x_se_112, x_se_113, x_se_114, x_se_115], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf178 = extern_kernels.convolution(buf177, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg241_1
        del buf177
        buf179 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [x_288, x_se_112, x_se_113, x_se_114, x_se_115, sigmoid_28, x_289], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_68.run(buf179, buf178, arg242_1, 451584, grid=grid(451584), stream=stream0)
        del arg242_1
        # Topologically Sorted Source Nodes: [x_288, x_se_112, x_se_113, x_se_114, x_se_115, sigmoid_28, x_289, x_290], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf180 = extern_kernels.convolution(buf179, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 192, 7, 7), (9408, 1, 1344, 192))
        del arg243_1
        buf181 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_291, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_69.run(buf181, buf180, arg244_1, arg245_1, arg246_1, arg247_1, 75264, grid=grid(75264), stream=stream0)
        del arg244_1
        del arg245_1
        del arg246_1
        del arg247_1
        del buf180
        # Topologically Sorted Source Nodes: [x_293], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg248_1
        buf183 = buf182; del buf182  # reuse
        buf184 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_294, x_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_64.run(buf183, arg249_1, arg250_1, arg251_1, arg252_1, buf184, 451584, grid=grid(451584), stream=stream0)
        del arg249_1
        del arg250_1
        del arg251_1
        del arg252_1
        del buf183
        # Topologically Sorted Source Nodes: [x_295, x_296], Original ATen: [aten.silu, aten.convolution]
        buf185 = extern_kernels.convolution(buf184, arg253_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf185, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg253_1
        del buf184
        buf186 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [x_297], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_65.run(buf186, arg254_1, arg255_1, arg256_1, arg257_1, 451584, grid=grid(451584), stream=stream0)
        del arg254_1
        del arg255_1
        del arg256_1
        del arg257_1
        buf188 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_298, x_se_116], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_66.run(buf186, buf188, 9216, 49, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_298, x_se_116, x_se_117], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf189 = extern_kernels.convolution(buf188, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg258_1
        del buf188
        buf190 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_298, x_se_116, x_se_117, x_se_118], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_67.run(buf190, arg259_1, 384, grid=grid(384), stream=stream0)
        del arg259_1
        # Topologically Sorted Source Nodes: [x_298, x_se_116, x_se_117, x_se_118, x_se_119], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf191 = extern_kernels.convolution(buf190, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg260_1
        del buf190
        buf192 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_298, x_se_116, x_se_117, x_se_118, x_se_119, sigmoid_29, x_299], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_68.run(buf192, buf191, arg261_1, 451584, grid=grid(451584), stream=stream0)
        del arg261_1
        # Topologically Sorted Source Nodes: [x_298, x_se_116, x_se_117, x_se_118, x_se_119, sigmoid_29, x_299, x_300], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf193 = extern_kernels.convolution(buf192, arg262_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 192, 7, 7), (9408, 1, 1344, 192))
        del arg262_1
        buf194 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [x_301, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_69.run(buf194, buf193, arg263_1, arg264_1, arg265_1, arg266_1, 75264, grid=grid(75264), stream=stream0)
        del arg263_1
        del arg264_1
        del arg265_1
        del arg266_1
        del buf193
        # Topologically Sorted Source Nodes: [x_303], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, arg267_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg267_1
        buf196 = buf195; del buf195  # reuse
        buf197 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [x_304, x_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_64.run(buf196, arg268_1, arg269_1, arg270_1, arg271_1, buf197, 451584, grid=grid(451584), stream=stream0)
        del arg268_1
        del arg269_1
        del arg270_1
        del arg271_1
        del buf196
        # Topologically Sorted Source Nodes: [x_305, x_306], Original ATen: [aten.silu, aten.convolution]
        buf198 = extern_kernels.convolution(buf197, arg272_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf198, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg272_1
        del buf197
        buf199 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_65.run(buf199, arg273_1, arg274_1, arg275_1, arg276_1, 451584, grid=grid(451584), stream=stream0)
        del arg273_1
        del arg274_1
        del arg275_1
        del arg276_1
        buf201 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [x_308, x_se_120], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_66.run(buf199, buf201, 9216, 49, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_308, x_se_120, x_se_121], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf202 = extern_kernels.convolution(buf201, arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg277_1
        del buf201
        buf203 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [x_308, x_se_120, x_se_121, x_se_122], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_67.run(buf203, arg278_1, 384, grid=grid(384), stream=stream0)
        del arg278_1
        # Topologically Sorted Source Nodes: [x_308, x_se_120, x_se_121, x_se_122, x_se_123], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf204 = extern_kernels.convolution(buf203, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg279_1
        del buf203
        buf205 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [x_308, x_se_120, x_se_121, x_se_122, x_se_123, sigmoid_30, x_309], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_68.run(buf205, buf204, arg280_1, 451584, grid=grid(451584), stream=stream0)
        del arg280_1
        # Topologically Sorted Source Nodes: [x_308, x_se_120, x_se_121, x_se_122, x_se_123, sigmoid_30, x_309, x_310], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf206 = extern_kernels.convolution(buf205, arg281_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 192, 7, 7), (9408, 1, 1344, 192))
        del arg281_1
        buf207 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [x_311, x_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_69.run(buf207, buf206, arg282_1, arg283_1, arg284_1, arg285_1, 75264, grid=grid(75264), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        del buf206
        # Topologically Sorted Source Nodes: [x_311, x_312, x_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf208 = extern_kernels.convolution(buf207, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg286_1
        del buf207
        buf209 = buf208; del buf208  # reuse
        buf210 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [x_314, x_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_64.run(buf209, arg287_1, arg288_1, arg289_1, arg290_1, buf210, 451584, grid=grid(451584), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        del buf209
        # Topologically Sorted Source Nodes: [x_315, x_316], Original ATen: [aten.silu, aten.convolution]
        buf211 = extern_kernels.convolution(buf210, arg291_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf211, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
        del arg291_1
        del buf210
        buf212 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [x_317], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_65.run(buf212, arg292_1, arg293_1, arg294_1, arg295_1, 451584, grid=grid(451584), stream=stream0)
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        buf214 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_318, x_se_124], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_66.run(buf212, buf214, 9216, 49, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_318, x_se_124, x_se_125], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf215 = extern_kernels.convolution(buf214, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg296_1
        del buf214
        buf216 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [x_318, x_se_124, x_se_125, x_se_126], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_67.run(buf216, arg297_1, 384, grid=grid(384), stream=stream0)
        del arg297_1
        # Topologically Sorted Source Nodes: [x_318, x_se_124, x_se_125, x_se_126, x_se_127], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf217 = extern_kernels.convolution(buf216, arg298_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg298_1
        del buf216
        buf218 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [x_318, x_se_124, x_se_125, x_se_126, x_se_127, sigmoid_31, x_319], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_68.run(buf218, buf217, arg299_1, 451584, grid=grid(451584), stream=stream0)
        del arg299_1
        del buf217
        # Topologically Sorted Source Nodes: [x_318, x_se_124, x_se_125, x_se_126, x_se_127, sigmoid_31, x_319, x_320], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf219 = extern_kernels.convolution(buf218, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 320, 7, 7), (15680, 1, 2240, 320))
        del arg300_1
        del buf218
        buf220 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_321], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_70.run(buf220, arg301_1, arg302_1, arg303_1, arg304_1, 125440, grid=grid(125440), stream=stream0)
        del arg301_1
        del arg302_1
        del arg303_1
        del arg304_1
        # Topologically Sorted Source Nodes: [x_321, x_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf221 = extern_kernels.convolution(buf220, arg305_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
        del arg305_1
        del buf220
        buf222 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [x_323], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_71.run(buf222, arg306_1, arg307_1, arg308_1, arg309_1, 501760, grid=grid(501760), stream=stream0)
        del arg306_1
        del arg307_1
        del arg308_1
        del arg309_1
        buf224 = empty_strided_cuda((8, 1280, 1, 1), (1280, 1, 10240, 10240), torch.float32)
        # Topologically Sorted Source Nodes: [x_324, x_325], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_72.run(buf222, buf224, 10240, 49, grid=grid(10240), stream=stream0)
        del buf222
        buf225 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_327], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg311_1, reinterpret_tensor(buf224, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg310_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf225)
        del arg310_1
        del arg311_1
        del buf224
    return (buf225, )


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
    arg158_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    arg224_1 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    arg291_1 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tf_efficientnet_b0', benchmark_compiled_module)
