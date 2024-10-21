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


# kernel path: /tmp/torchinductor_sahanp/vs/cvsvm4wq6lo2o7x7dlugddwusjww6ldtv5chb7rbtf5gsdtzowcx.py
# Topologically Sorted Source Nodes: [batch_norm_57, x_11, input_8], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_57 => var_mean_57
#   input_8 => convolution_81
#   x_11 => constant_pad_nd_5
# Graph fragment:
#   %var_mean_57 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_172, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %constant_pad_nd_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg0_1, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_5, %view_174, %arg3_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_0 = async_compile.triton('triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[16, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_0(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 27
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (27*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 27, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 27.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.19245008972987526
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (27*x0)), tmp27, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3m/c3mc3pqcniksfjceyjwpcckytg24cqwtspvn6kqainkfzdkscdck.py
# Topologically Sorted Source Nodes: [batch_norm_58, x_11, input_8, gelu_52, input_9, input_10], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   batch_norm_58 => var_mean_58
#   gelu_52 => add_122, erf_52, mul_442, mul_443, mul_444
#   input_10 => convolution_82
#   input_8 => convolution_81
#   input_9 => mul_445
#   x_11 => constant_pad_nd_5
# Graph fragment:
#   %var_mean_58 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_175, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %constant_pad_nd_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg0_1, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_5, %view_174, %arg3_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.5), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.7071067811865476), kwargs = {})
#   %erf_52 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_443,), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_52, 1), kwargs = {})
#   %mul_444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_442, %add_122), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_444, 1.7015043497085571), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_445, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_1 = async_compile.triton('triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_1(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 144.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.08333333333333333
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (144*x0)), tmp27, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hf/chf3a4d3htld7zdhu32x5velix3c6xyrzmnhhda5xdbq2rs5muon.py
# Topologically Sorted Source Nodes: [batch_norm_59, x_11, input_8, gelu_52, input_9, input_10, gelu_53, input_11, input_12], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   batch_norm_59 => var_mean_59
#   gelu_52 => add_122, erf_52, mul_442, mul_443, mul_444
#   gelu_53 => add_124, erf_53, mul_449, mul_450, mul_451
#   input_10 => convolution_82
#   input_11 => mul_452
#   input_12 => convolution_83
#   input_8 => convolution_81
#   input_9 => mul_445
#   x_11 => constant_pad_nd_5
# Graph fragment:
#   %var_mean_59 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_178, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %constant_pad_nd_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg0_1, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_5, %view_174, %arg3_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.5), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.7071067811865476), kwargs = {})
#   %erf_52 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_443,), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_52, 1), kwargs = {})
#   %mul_444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_442, %add_122), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_444, 1.7015043497085571), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_445, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_449 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, 0.5), kwargs = {})
#   %mul_450 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, 0.7071067811865476), kwargs = {})
#   %erf_53 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_450,), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_53, 1), kwargs = {})
#   %mul_451 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_449, %add_124), kwargs = {})
#   %mul_452 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_451, 1.7015043497085571), kwargs = {})
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_452, %view_180, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_2 = async_compile.triton('triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_2(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (288*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 288, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 288.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.05892556509887896
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (288*x0)), tmp27, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uk/cuky4ih7ytgsz6z4kvnqqghrrkyijcab6clcegqpk4hktf7yb7ze.py
# Topologically Sorted Source Nodes: [batch_norm_60, x_11, input_8, gelu_52, input_9, input_10, gelu_53, input_11, input_12, gelu_54, input_13, x_12, input_14], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   batch_norm_60 => var_mean_60
#   gelu_52 => add_122, erf_52, mul_442, mul_443, mul_444
#   gelu_53 => add_124, erf_53, mul_449, mul_450, mul_451
#   gelu_54 => add_126, erf_54, mul_456, mul_457, mul_458
#   input_10 => convolution_82
#   input_11 => mul_452
#   input_12 => convolution_83
#   input_13 => mul_459
#   input_14 => convolution_84
#   input_8 => convolution_81
#   input_9 => mul_445
#   x_11 => constant_pad_nd_5
#   x_12 => constant_pad_nd_6
# Graph fragment:
#   %var_mean_60 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_181, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %constant_pad_nd_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg0_1, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_5, %view_174, %arg3_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.5), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.7071067811865476), kwargs = {})
#   %erf_52 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_443,), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_52, 1), kwargs = {})
#   %mul_444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_442, %add_122), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_444, 1.7015043497085571), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_445, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_449 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, 0.5), kwargs = {})
#   %mul_450 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, 0.7071067811865476), kwargs = {})
#   %erf_53 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_450,), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_53, 1), kwargs = {})
#   %mul_451 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_449, %add_124), kwargs = {})
#   %mul_452 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_451, 1.7015043497085571), kwargs = {})
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_452, %view_180, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_456 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_83, 0.5), kwargs = {})
#   %mul_457 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_83, 0.7071067811865476), kwargs = {})
#   %erf_54 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_457,), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_54, 1), kwargs = {})
#   %mul_458 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_456, %add_126), kwargs = {})
#   %mul_459 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_458, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_6 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_459, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_84 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_6, %view_183, %arg12_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_3 = async_compile.triton('triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_3(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (576*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 576, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 576.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.041666666666666664
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (576*x0)), tmp27, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bb/cbbim6dsndtnozh5bu6e3utluh7lhhcdegabstkwroc35whrjlyo.py
# Topologically Sorted Source Nodes: [batch_norm_61, shortcut_4], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_61 => var_mean_61
#   shortcut_4 => convolution_85
# Graph fragment:
#   %var_mean_61 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_184, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %convolution_85 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_467, %view_186, %arg15_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_4 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_4(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 128.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.08838834764831845
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ln/clnu6h62yya667jqfppid2525uugpaqhhznhmh4jn73ihemma6lw.py
# Topologically Sorted Source Nodes: [batch_norm_62, out_85], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_62 => var_mean_62
#   out_85 => convolution_86
# Graph fragment:
#   %var_mean_62 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_187, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_467, %view_189, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_5 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_5(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 128.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.08838834764831845
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7x/c7xngw6cqiukfncigu3yoro7qtyddp3hjzpjaiwcdtrwkrfdxfen.py
# Topologically Sorted Source Nodes: [batch_norm_63, out_85, gelu_56, mul__68, out_86], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   batch_norm_63 => var_mean_63
#   gelu_56 => add_131, erf_56, mul_474, mul_475, mul_476
#   mul__68 => mul_477
#   out_85 => convolution_86
#   out_86 => convolution_87
# Graph fragment:
#   %var_mean_63 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_190, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_467, %view_189, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_474 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, 0.5), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, 0.7071067811865476), kwargs = {})
#   %erf_56 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_475,), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_56, 1), kwargs = {})
#   %mul_476 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_474, %add_131), kwargs = {})
#   %mul_477 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_476, 1.7015043497085571), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_477, %view_192, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_6 = async_compile.triton('triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_6(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 1152.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = 0.02946278254943948
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1152*x0)), tmp16, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/76/c762iur2yvqxe5s6dipr2gjyfojgovqq2njg34bscmlrcd4hhddu.py
# Topologically Sorted Source Nodes: [batch_norm_66, avg_pool2d_3, shortcut_5], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
# Source node to ATen node mapping:
#   avg_pool2d_3 => avg_pool2d_3
#   batch_norm_66 => var_mean_66
#   shortcut_5 => convolution_92
# Graph fragment:
#   %var_mean_66 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_199, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %avg_pool2d_3 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_503, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_92 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_3, %view_201, %arg35_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_7 = async_compile.triton('triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_7(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None)
    tmp21 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 256.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = 0.0625
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/y7/cy7sws7beb37xtopitdv7b4opbtyujurvtywql2sh4e2e5nr6xmi.py
# Topologically Sorted Source Nodes: [batch_norm_67, out_92], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_67 => var_mean_67
#   out_92 => convolution_93
# Graph fragment:
#   %var_mean_67 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_202, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_503, %view_204, %arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_8 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_8(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None)
    tmp21 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 256.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = 0.0625
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5t/c5tj5rvanzca5z2ft2zbkre22y7b4ieyhsdgheibrvsk7jrhiw47.py
# Topologically Sorted Source Nodes: [batch_norm_68, out_92, gelu_60, mul__73, x_13, out_93], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   batch_norm_68 => var_mean_68
#   gelu_60 => add_141, erf_60, mul_510, mul_511, mul_512
#   mul__73 => mul_513
#   out_92 => convolution_93
#   out_93 => convolution_94
#   x_13 => constant_pad_nd_7
# Graph fragment:
#   %var_mean_68 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_205, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_503, %view_204, %arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.5), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.7071067811865476), kwargs = {})
#   %erf_60 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_511,), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_60, 1), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_510, %add_141), kwargs = {})
#   %mul_513 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_512, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_7 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_513, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_94 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_7, %view_207, %arg41_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 2), kwargs = {})
triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9 = async_compile.triton('triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 1152.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = 0.02946278254943948
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1152*x0)), tmp16, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6m/c6mtp4qfi3c3rndt2scqze7o3567aurvw2rhekw724m65htlfqiu.py
# Topologically Sorted Source Nodes: [batch_norm_71, gelu_63, mul__77, out_98, out_99], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_71 => var_mean_71
#   gelu_63 => add_148, erf_63, mul_535, mul_536, mul_537
#   mul__77 => mul_538
#   out_98 => mul_539
#   out_99 => convolution_99
# Graph fragment:
#   %var_mean_71 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_214, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %mul_535 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_147, 0.5), kwargs = {})
#   %mul_536 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_147, 0.7071067811865476), kwargs = {})
#   %erf_63 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_536,), kwargs = {})
#   %add_148 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_63, 1), kwargs = {})
#   %mul_537 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_535, %add_148), kwargs = {})
#   %mul_538 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_537, 1.7015043497085571), kwargs = {})
#   %mul_539 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_538, 0.9805806756909201), kwargs = {})
#   %convolution_99 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_539, %view_216, %arg55_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_gelu_mul_10 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_gelu_mul_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_gelu_mul_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_gelu_mul_10(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp21 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 512, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = 0.04419417382415922
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wj/cwjenlnpjrrs3jcpqtzhfdgf7jztajgskmnk6r7dqu5dmig3s5sd.py
# Topologically Sorted Source Nodes: [batch_norm_75, avg_pool2d_4, shortcut_6], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
# Source node to ATen node mapping:
#   avg_pool2d_4 => avg_pool2d_4
#   batch_norm_75 => var_mean_75
#   shortcut_6 => convolution_105
# Graph fragment:
#   %var_mean_75 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_226, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %avg_pool2d_4 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_572, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_105 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_4, %view_228, %arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_11 = async_compile.triton('triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_11(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1536
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp21 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 512, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = 0.04419417382415922
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ui/cui4cxni2ew4bd55uqvrwtywrhepe4bjoyyomn23raleztuqkher.py
# Topologically Sorted Source Nodes: [batch_norm_76, out_106], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_76 => var_mean_76
#   out_106 => convolution_106
# Graph fragment:
#   %var_mean_76 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_229, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_572, %view_231, %arg75_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_12 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_12(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 768
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp21 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 512, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = 0.04419417382415922
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zd/czdbumtndoejf5ozviego2s7k4qhlaonjypc2hnwrtisxc2klmjz.py
# Topologically Sorted Source Nodes: [batch_norm_77, out_106, gelu_68, mul__83, x_14, out_107], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   batch_norm_77 => var_mean_77
#   gelu_68 => add_160, erf_68, mul_579, mul_580, mul_581
#   mul__83 => mul_582
#   out_106 => convolution_106
#   out_107 => convolution_107
#   x_14 => constant_pad_nd_8
# Graph fragment:
#   %var_mean_77 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_232, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_572, %view_231, %arg75_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_579 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.5), kwargs = {})
#   %mul_580 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.7071067811865476), kwargs = {})
#   %erf_68 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_580,), kwargs = {})
#   %add_160 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_68, 1), kwargs = {})
#   %mul_581 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_579, %add_160), kwargs = {})
#   %mul_582 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_581, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_8 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_582, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_8, %view_234, %arg78_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6), kwargs = {})
triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13 = async_compile.triton('triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 1152.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = 0.02946278254943948
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1152*x0)), tmp16, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ja/cjaup5xhukbsy76nqha22pvq7sd3frmvmuptwsvpxdndo7km5iqz.py
# Topologically Sorted Source Nodes: [batch_norm_79, out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108, gelu_70, mul__85, out_109], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   batch_norm_79 => var_mean_79
#   gelu_68 => add_160, erf_68, mul_579, mul_580, mul_581
#   gelu_69 => add_162, erf_69, mul_586, mul_587, mul_588
#   gelu_70 => add_164, erf_70, mul_593, mul_594, mul_595
#   mul__83 => mul_582
#   mul__84 => mul_589
#   mul__85 => mul_596
#   out_106 => convolution_106
#   out_107 => convolution_107
#   out_108 => convolution_108
#   out_109 => convolution_109
#   x_14 => constant_pad_nd_8
# Graph fragment:
#   %var_mean_79 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_238, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_572, %view_231, %arg75_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_579 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.5), kwargs = {})
#   %mul_580 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.7071067811865476), kwargs = {})
#   %erf_68 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_580,), kwargs = {})
#   %add_160 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_68, 1), kwargs = {})
#   %mul_581 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_579, %add_160), kwargs = {})
#   %mul_582 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_581, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_8 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_582, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_8, %view_234, %arg78_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_586 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, 0.5), kwargs = {})
#   %mul_587 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, 0.7071067811865476), kwargs = {})
#   %erf_69 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_587,), kwargs = {})
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_69, 1), kwargs = {})
#   %mul_588 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_586, %add_162), kwargs = {})
#   %mul_589 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_588, 1.7015043497085571), kwargs = {})
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_589, %view_237, %arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_593 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, 0.5), kwargs = {})
#   %mul_594 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, 0.7071067811865476), kwargs = {})
#   %erf_70 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_594,), kwargs = {})
#   %add_164 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_70, 1), kwargs = {})
#   %mul_595 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_593, %add_164), kwargs = {})
#   %mul_596 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_595, 1.7015043497085571), kwargs = {})
#   %convolution_109 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_596, %view_240, %arg84_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14 = async_compile.triton('triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1536
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 768.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.03608439182435161
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h4/ch4d2rpxwpji3nyop3p4gxnxnjj6s7nc46ekh7ly4je5nvysvkkj.py
# Topologically Sorted Source Nodes: [batch_norm_80, gelu_71, mul__87, out_112, out_113], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_80 => var_mean_80
#   gelu_71 => add_167, erf_71, mul_604, mul_605, mul_606
#   mul__87 => mul_607
#   out_112 => mul_608
#   out_113 => convolution_112
# Graph fragment:
#   %var_mean_80 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_241, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %mul_604 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_166, 0.5), kwargs = {})
#   %mul_605 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_166, 0.7071067811865476), kwargs = {})
#   %erf_71 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_605,), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_71, 1), kwargs = {})
#   %mul_606 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_604, %add_167), kwargs = {})
#   %mul_607 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_606, 1.7015043497085571), kwargs = {})
#   %mul_608 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_607, 0.9805806756909201), kwargs = {})
#   %convolution_112 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_608, %view_243, %arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15 = async_compile.triton('triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 1536.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = 0.02551551815399144
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp16, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rz/crzut74iglghjbbpcyw4535jrr3bcrigvgeebpgnba6ewm7ischx.py
# Topologically Sorted Source Nodes: [batch_norm_100, avg_pool2d_5, shortcut_7], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
# Source node to ATen node mapping:
#   avg_pool2d_5 => avg_pool2d_5
#   batch_norm_100 => var_mean_100
#   shortcut_7 => convolution_142
# Graph fragment:
#   %var_mean_100 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_301, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %avg_pool2d_5 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_773, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_142 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_5, %view_303, %arg177_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_red_fused__native_batch_norm_legit_avg_pool2d_convolution_16 = async_compile.triton('triton_red_fused__native_batch_norm_legit_avg_pool2d_convolution_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_avg_pool2d_convolution_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_avg_pool2d_convolution_16(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 1536.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = 0.02551551815399144
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp16, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lc/clcoyvyeam5xjjdjin2qt24pnwunyzjfsmhwu6rbfreskz2cto66.py
# Topologically Sorted Source Nodes: [batch_norm_113, gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, mul_206, out_166, mul__126, mul_208, out_167, x_16], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   batch_norm_113 => var_mean_113
#   gelu_100 => add_233, erf_100, mul_846, mul_847, mul_848
#   gelu_101 => add_235, erf_101, mul_853, mul_854, mul_855
#   gelu_102 => add_237, erf_102, mul_860, mul_861, mul_862
#   gelu_99 => add_231, erf_99, mul_838, mul_839, mul_840
#   mul_206 => mul_867
#   mul_208 => mul_870
#   mul__122 => mul_841
#   mul__123 => mul_849
#   mul__124 => mul_856
#   mul__125 => mul_863
#   mul__126 => mul_869
#   out_161 => mul_842
#   out_162 => convolution_155
#   out_163 => convolution_156
#   out_164 => convolution_157
#   out_165 => convolution_158
#   out_166 => mul_868
#   out_167 => add_239
#   sigmoid_23 => sigmoid_23
#   x_16 => convolution_161
#   x_se_92 => mean_24
#   x_se_93 => convolution_159
#   x_se_94 => relu_23
#   x_se_95 => convolution_160
# Graph fragment:
#   %var_mean_113 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_340, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %mul_838 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, 0.5), kwargs = {})
#   %mul_839 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, 0.7071067811865476), kwargs = {})
#   %erf_99 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_839,), kwargs = {})
#   %add_231 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_99, 1), kwargs = {})
#   %mul_840 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_838, %add_231), kwargs = {})
#   %mul_841 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_840, 1.7015043497085571), kwargs = {})
#   %mul_842 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_841, 0.9622504486493761), kwargs = {})
#   %convolution_155 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_842, %view_330, %arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_846 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_155, 0.5), kwargs = {})
#   %mul_847 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_155, 0.7071067811865476), kwargs = {})
#   %erf_100 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_847,), kwargs = {})
#   %add_233 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_100, 1), kwargs = {})
#   %mul_848 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_846, %add_233), kwargs = {})
#   %mul_849 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_848, 1.7015043497085571), kwargs = {})
#   %convolution_156 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_849, %view_333, %arg217_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_853 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_156, 0.5), kwargs = {})
#   %mul_854 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_156, 0.7071067811865476), kwargs = {})
#   %erf_101 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_854,), kwargs = {})
#   %add_235 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_101, 1), kwargs = {})
#   %mul_855 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_853, %add_235), kwargs = {})
#   %mul_856 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_855, 1.7015043497085571), kwargs = {})
#   %convolution_157 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_856, %view_336, %arg220_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_860 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_157, 0.5), kwargs = {})
#   %mul_861 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_157, 0.7071067811865476), kwargs = {})
#   %erf_102 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_861,), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_102, 1), kwargs = {})
#   %mul_862 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_860, %add_237), kwargs = {})
#   %mul_863 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_862, 1.7015043497085571), kwargs = {})
#   %convolution_158 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_863, %view_339, %arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_158, [2, 3], True), kwargs = {})
#   %convolution_159 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_24, %arg224_1, %arg225_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_159,), kwargs = {})
#   %convolution_160 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg226_1, %arg227_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_23 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_160,), kwargs = {})
#   %mul_867 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_158, %sigmoid_23), kwargs = {})
#   %mul_868 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_867, 2.0), kwargs = {})
#   %mul_869 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_868, %arg228_1), kwargs = {})
#   %mul_870 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_869, 0.2), kwargs = {})
#   %add_239 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_870, %add_230), kwargs = {})
#   %convolution_161 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_239, %view_342, %arg231_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_red_fused__native_batch_norm_legit_add_convolution_gelu_mean_mul_relu_sigmoid_17 = async_compile.triton('triton_red_fused__native_batch_norm_legit_add_convolution_gelu_mean_mul_relu_sigmoid_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_add_convolution_gelu_mean_mul_relu_sigmoid_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_add_convolution_gelu_mean_mul_relu_sigmoid_17(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 1536.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = 0.02551551815399144
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp16, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qh/cqh7ksp3k3knhviwbuqkpqjsqdnmultq2i6s3j35l34l4y3safg4.py
# Topologically Sorted Source Nodes: [x_11, input_8], Original ATen: [aten.constant_pad_nd, aten.convolution]
# Source node to ATen node mapping:
#   input_8 => convolution_81
#   x_11 => constant_pad_nd_5
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg0_1, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_5, %view_174, %arg3_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_18 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1585176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 257) % 257
    x0 = xindex % 257
    x2 = (xindex // 66049)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (256*x1) + (65536*x2)), tmp5 & xmask, other=0.0)
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6e/c6eyizfxdfh6cbjjc536msj22smtqhxjugvflxpymqul4qdlflkd.py
# Topologically Sorted Source Nodes: [x_11, input_8, gelu_52, input_9, input_10], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   gelu_52 => add_122, erf_52, mul_442, mul_443, mul_444
#   input_10 => convolution_82
#   input_8 => convolution_81
#   input_9 => mul_445
#   x_11 => constant_pad_nd_5
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg0_1, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_5, %view_174, %arg3_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.5), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.7071067811865476), kwargs = {})
#   %erf_52 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_443,), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_52, 1), kwargs = {})
#   %mul_444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_442, %add_122), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_444, 1.7015043497085571), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_445, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_19 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_gelu_mul_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_gelu_mul_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 16384) % 16
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ed/ced3cwrw7pzc2n5odkymdus5nb6ww7u3tozjivyj3zz7ck2r4323.py
# Topologically Sorted Source Nodes: [x_11, input_8, gelu_52, input_9, input_10, gelu_53, input_11, input_12], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   gelu_52 => add_122, erf_52, mul_442, mul_443, mul_444
#   gelu_53 => add_124, erf_53, mul_449, mul_450, mul_451
#   input_10 => convolution_82
#   input_11 => mul_452
#   input_12 => convolution_83
#   input_8 => convolution_81
#   input_9 => mul_445
#   x_11 => constant_pad_nd_5
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg0_1, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_5, %view_174, %arg3_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.5), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.7071067811865476), kwargs = {})
#   %erf_52 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_443,), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_52, 1), kwargs = {})
#   %mul_444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_442, %add_122), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_444, 1.7015043497085571), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_445, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_449 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, 0.5), kwargs = {})
#   %mul_450 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, 0.7071067811865476), kwargs = {})
#   %erf_53 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_450,), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_53, 1), kwargs = {})
#   %mul_451 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_449, %add_124), kwargs = {})
#   %mul_452 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_451, 1.7015043497085571), kwargs = {})
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_452, %view_180, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_20 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_gelu_mul_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_gelu_mul_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 16384) % 32
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/w5/cw5f7g6v4luw4tmbj3bk563b23yurpd263fjvskeh3o5kr2m4xxl.py
# Topologically Sorted Source Nodes: [x_11, input_8, gelu_52, input_9, input_10, gelu_53, input_11, input_12, gelu_54, input_13, x_12, input_14], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   gelu_52 => add_122, erf_52, mul_442, mul_443, mul_444
#   gelu_53 => add_124, erf_53, mul_449, mul_450, mul_451
#   gelu_54 => add_126, erf_54, mul_456, mul_457, mul_458
#   input_10 => convolution_82
#   input_11 => mul_452
#   input_12 => convolution_83
#   input_13 => mul_459
#   input_14 => convolution_84
#   input_8 => convolution_81
#   input_9 => mul_445
#   x_11 => constant_pad_nd_5
#   x_12 => constant_pad_nd_6
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg0_1, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_5, %view_174, %arg3_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.5), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.7071067811865476), kwargs = {})
#   %erf_52 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_443,), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_52, 1), kwargs = {})
#   %mul_444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_442, %add_122), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_444, 1.7015043497085571), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_445, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_449 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, 0.5), kwargs = {})
#   %mul_450 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, 0.7071067811865476), kwargs = {})
#   %erf_53 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_450,), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_53, 1), kwargs = {})
#   %mul_451 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_449, %add_124), kwargs = {})
#   %mul_452 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_451, 1.7015043497085571), kwargs = {})
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_452, %view_180, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_456 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_83, 0.5), kwargs = {})
#   %mul_457 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_83, 0.7071067811865476), kwargs = {})
#   %erf_54 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_457,), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_54, 1), kwargs = {})
#   %mul_458 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_456, %add_126), kwargs = {})
#   %mul_459 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_458, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_6 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_459, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_84 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_6, %view_183, %arg12_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_21 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_gelu_mul_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_gelu_mul_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8520192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 129) % 129
    x0 = xindex % 129
    x4 = (xindex // 16641)
    x2 = (xindex // 16641) % 64
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 128, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (128*x1) + (16384*x4)), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = libdevice.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tmp17 = 1.7015043497085571
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tl.store(out_ptr0 + (x5), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/45/c4526xkhvgth7ektjstkd26ov4c5edxmtwegjsfecji34celzd6b.py
# Topologically Sorted Source Nodes: [x_11, input_8, gelu_52, input_9, input_10, gelu_53, input_11, input_12, gelu_54, input_13, x_12, input_14, gelu_55, mul__67, out_84], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   gelu_52 => add_122, erf_52, mul_442, mul_443, mul_444
#   gelu_53 => add_124, erf_53, mul_449, mul_450, mul_451
#   gelu_54 => add_126, erf_54, mul_456, mul_457, mul_458
#   gelu_55 => add_128, erf_55, mul_463, mul_464, mul_465
#   input_10 => convolution_82
#   input_11 => mul_452
#   input_12 => convolution_83
#   input_13 => mul_459
#   input_14 => convolution_84
#   input_8 => convolution_81
#   input_9 => mul_445
#   mul__67 => mul_466
#   out_84 => mul_467
#   x_11 => constant_pad_nd_5
#   x_12 => constant_pad_nd_6
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg0_1, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_5, %view_174, %arg3_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.5), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.7071067811865476), kwargs = {})
#   %erf_52 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_443,), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_52, 1), kwargs = {})
#   %mul_444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_442, %add_122), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_444, 1.7015043497085571), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_445, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_449 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, 0.5), kwargs = {})
#   %mul_450 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, 0.7071067811865476), kwargs = {})
#   %erf_53 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_450,), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_53, 1), kwargs = {})
#   %mul_451 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_449, %add_124), kwargs = {})
#   %mul_452 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_451, 1.7015043497085571), kwargs = {})
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_452, %view_180, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_456 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_83, 0.5), kwargs = {})
#   %mul_457 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_83, 0.7071067811865476), kwargs = {})
#   %erf_54 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_457,), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_54, 1), kwargs = {})
#   %mul_458 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_456, %add_126), kwargs = {})
#   %mul_459 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_458, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_6 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_459, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_84 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_6, %view_183, %arg12_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_463 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_84, 0.5), kwargs = {})
#   %mul_464 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_84, 0.7071067811865476), kwargs = {})
#   %erf_55 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_464,), kwargs = {})
#   %add_128 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_55, 1), kwargs = {})
#   %mul_465 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_463, %add_128), kwargs = {})
#   %mul_466 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_465, 1.7015043497085571), kwargs = {})
#   %mul_467 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_466, 1.0), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_22 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_gelu_mul_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_gelu_mul_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 4096) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 * tmp8
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/52/c52zxt7z6jwzetnsfqujuwxf3ejv45trx4xbbaizzjlxljop6gm2.py
# Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86], Original ATen: [aten.convolution, aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   gelu_56 => add_131, erf_56, mul_474, mul_475, mul_476
#   mul__68 => mul_477
#   out_85 => convolution_86
#   out_86 => convolution_87
# Graph fragment:
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_467, %view_189, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_474 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, 0.5), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, 0.7071067811865476), kwargs = {})
#   %erf_56 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_475,), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_56, 1), kwargs = {})
#   %mul_476 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_474, %add_131), kwargs = {})
#   %mul_477 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_476, 1.7015043497085571), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_477, %view_192, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_gelu_mul_23 = async_compile.triton('triton_poi_fused_convolution_gelu_mul_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_mul_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_mul_23(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 4096) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eo/ceoyo4czksdi6ouf3qf55wwjhbc4sj3iclnr4a52u33z7x4yuz6c.py
# Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87, gelu_58, mul__70, out_88, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   gelu_56 => add_131, erf_56, mul_474, mul_475, mul_476
#   gelu_57 => add_133, erf_57, mul_481, mul_482, mul_483
#   gelu_58 => add_135, erf_58, mul_488, mul_489, mul_490
#   mul__68 => mul_477
#   mul__69 => mul_484
#   mul__70 => mul_491
#   out_85 => convolution_86
#   out_86 => convolution_87
#   out_87 => convolution_88
#   out_88 => convolution_89
#   x_se_48 => mean_13
#   x_se_49 => convolution_90
# Graph fragment:
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_467, %view_189, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_474 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, 0.5), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, 0.7071067811865476), kwargs = {})
#   %erf_56 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_475,), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_56, 1), kwargs = {})
#   %mul_476 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_474, %add_131), kwargs = {})
#   %mul_477 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_476, 1.7015043497085571), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_477, %view_192, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_481 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.5), kwargs = {})
#   %mul_482 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.7071067811865476), kwargs = {})
#   %erf_57 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_482,), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_57, 1), kwargs = {})
#   %mul_483 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_481, %add_133), kwargs = {})
#   %mul_484 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_483, 1.7015043497085571), kwargs = {})
#   %convolution_88 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_484, %view_195, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_488 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, 0.5), kwargs = {})
#   %mul_489 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, 0.7071067811865476), kwargs = {})
#   %erf_58 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_489,), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_58, 1), kwargs = {})
#   %mul_490 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_488, %add_135), kwargs = {})
#   %mul_491 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_490, 1.7015043497085571), kwargs = {})
#   %convolution_89 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_491, %view_198, %arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_89, [2, 3], True), kwargs = {})
#   %convolution_90 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_13, %arg28_1, %arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_red_fused_convolution_gelu_mean_mul_24 = async_compile.triton('triton_red_fused_convolution_gelu_mean_mul_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_gelu_mean_mul_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_gelu_mean_mul_24(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 256
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (4096*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 4096.0
    tmp7 = tmp4 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/og/cogdw4v6u5jk2w3aor3wskpl2zqtl274le7dexcguzitp2ak2hkq.py
# Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87, gelu_58, mul__70, out_88, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.mean, aten.relu]
# Source node to ATen node mapping:
#   gelu_56 => add_131, erf_56, mul_474, mul_475, mul_476
#   gelu_57 => add_133, erf_57, mul_481, mul_482, mul_483
#   gelu_58 => add_135, erf_58, mul_488, mul_489, mul_490
#   mul__68 => mul_477
#   mul__69 => mul_484
#   mul__70 => mul_491
#   out_85 => convolution_86
#   out_86 => convolution_87
#   out_87 => convolution_88
#   out_88 => convolution_89
#   x_se_48 => mean_13
#   x_se_49 => convolution_90
#   x_se_50 => relu_12
#   x_se_51 => convolution_91
# Graph fragment:
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_467, %view_189, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_474 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, 0.5), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, 0.7071067811865476), kwargs = {})
#   %erf_56 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_475,), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_56, 1), kwargs = {})
#   %mul_476 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_474, %add_131), kwargs = {})
#   %mul_477 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_476, 1.7015043497085571), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_477, %view_192, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_481 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.5), kwargs = {})
#   %mul_482 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.7071067811865476), kwargs = {})
#   %erf_57 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_482,), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_57, 1), kwargs = {})
#   %mul_483 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_481, %add_133), kwargs = {})
#   %mul_484 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_483, 1.7015043497085571), kwargs = {})
#   %convolution_88 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_484, %view_195, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_488 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, 0.5), kwargs = {})
#   %mul_489 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, 0.7071067811865476), kwargs = {})
#   %erf_58 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_489,), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_58, 1), kwargs = {})
#   %mul_490 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_488, %add_135), kwargs = {})
#   %mul_491 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_490, 1.7015043497085571), kwargs = {})
#   %convolution_89 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_491, %view_198, %arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_89, [2, 3], True), kwargs = {})
#   %convolution_90 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_13, %arg28_1, %arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_90,), kwargs = {})
#   %convolution_91 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg30_1, %arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_gelu_mean_mul_relu_25 = async_compile.triton('triton_poi_fused_convolution_gelu_mean_mul_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_mean_mul_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_mean_mul_relu_25(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3y/c3yyxt3wte5rpuaev4xzvxkrq572vg3s2f67yaj4sgktejw5lnvr.py
# Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87, gelu_58, mul__70, out_88, x_se_48, x_se_49, x_se_50, x_se_51, sigmoid_12, mul_115, out_89, mul__71, mul_117, shortcut_4, out_90, gelu_59, mul__72, out_91], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   gelu_56 => add_131, erf_56, mul_474, mul_475, mul_476
#   gelu_57 => add_133, erf_57, mul_481, mul_482, mul_483
#   gelu_58 => add_135, erf_58, mul_488, mul_489, mul_490
#   gelu_59 => add_138, erf_59, mul_499, mul_500, mul_501
#   mul_115 => mul_495
#   mul_117 => mul_498
#   mul__68 => mul_477
#   mul__69 => mul_484
#   mul__70 => mul_491
#   mul__71 => mul_497
#   mul__72 => mul_502
#   out_85 => convolution_86
#   out_86 => convolution_87
#   out_87 => convolution_88
#   out_88 => convolution_89
#   out_89 => mul_496
#   out_90 => add_137
#   out_91 => mul_503
#   shortcut_4 => convolution_85
#   sigmoid_12 => sigmoid_12
#   x_se_48 => mean_13
#   x_se_49 => convolution_90
#   x_se_50 => relu_12
#   x_se_51 => convolution_91
# Graph fragment:
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_467, %view_189, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_474 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, 0.5), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, 0.7071067811865476), kwargs = {})
#   %erf_56 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_475,), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_56, 1), kwargs = {})
#   %mul_476 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_474, %add_131), kwargs = {})
#   %mul_477 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_476, 1.7015043497085571), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_477, %view_192, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_481 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.5), kwargs = {})
#   %mul_482 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.7071067811865476), kwargs = {})
#   %erf_57 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_482,), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_57, 1), kwargs = {})
#   %mul_483 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_481, %add_133), kwargs = {})
#   %mul_484 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_483, 1.7015043497085571), kwargs = {})
#   %convolution_88 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_484, %view_195, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_488 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, 0.5), kwargs = {})
#   %mul_489 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, 0.7071067811865476), kwargs = {})
#   %erf_58 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_489,), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_58, 1), kwargs = {})
#   %mul_490 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_488, %add_135), kwargs = {})
#   %mul_491 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_490, 1.7015043497085571), kwargs = {})
#   %convolution_89 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_491, %view_198, %arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_89, [2, 3], True), kwargs = {})
#   %convolution_90 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_13, %arg28_1, %arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_90,), kwargs = {})
#   %convolution_91 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg30_1, %arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_91,), kwargs = {})
#   %mul_495 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_89, %sigmoid_12), kwargs = {})
#   %mul_496 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_495, 2.0), kwargs = {})
#   %mul_497 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_496, %arg32_1), kwargs = {})
#   %mul_498 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_497, 0.2), kwargs = {})
#   %convolution_85 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_467, %view_186, %arg15_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_137 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_498, %convolution_85), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, 0.5), kwargs = {})
#   %mul_500 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, 0.7071067811865476), kwargs = {})
#   %erf_59 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_500,), kwargs = {})
#   %add_138 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_59, 1), kwargs = {})
#   %mul_501 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_499, %add_138), kwargs = {})
#   %mul_502 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_501, 1.7015043497085571), kwargs = {})
#   %mul_503 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_502, 0.9805806756909201), kwargs = {})
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_26 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 4096) % 256
    x4 = (xindex // 4096)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = 0.7071067811865476
    tmp22 = tmp18 * tmp21
    tmp23 = libdevice.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 1.7015043497085571
    tmp28 = tmp26 * tmp27
    tmp29 = 0.9805806756909201
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr0 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kq/ckqo3rubvo2gvmsc7omcnor65kwenqj2d7kwh3vl6kgwkslxeaca.py
# Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   gelu_60 => add_141, erf_60, mul_510, mul_511, mul_512
#   mul__73 => mul_513
#   out_92 => convolution_93
#   out_93 => convolution_94
#   x_13 => constant_pad_nd_7
# Graph fragment:
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_503, %view_204, %arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.5), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.7071067811865476), kwargs = {})
#   %erf_60 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_511,), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_60, 1), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_510, %add_141), kwargs = {})
#   %mul_513 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_512, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_7 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_513, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_94 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_7, %view_207, %arg41_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 2), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_27 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_gelu_mul_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_gelu_mul_27(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8652800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 65) % 65
    x0 = xindex % 65
    x4 = (xindex // 4225)
    x2 = (xindex // 4225) % 256
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (64*x1) + (4096*x4)), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = libdevice.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tmp17 = 1.7015043497085571
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tl.store(out_ptr0 + (x5), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qc/cqcluyzpndzhypcak4q47hh3ms45t636xm3pajrl24bkvjxlwipe.py
# Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   gelu_60 => add_141, erf_60, mul_510, mul_511, mul_512
#   gelu_61 => add_143, erf_61, mul_517, mul_518, mul_519
#   mul__73 => mul_513
#   mul__74 => mul_520
#   out_92 => convolution_93
#   out_93 => convolution_94
#   out_94 => convolution_95
#   x_13 => constant_pad_nd_7
# Graph fragment:
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_503, %view_204, %arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.5), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.7071067811865476), kwargs = {})
#   %erf_60 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_511,), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_60, 1), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_510, %add_141), kwargs = {})
#   %mul_513 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_512, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_7 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_513, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_94 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_7, %view_207, %arg41_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 2), kwargs = {})
#   %mul_517 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, 0.5), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, 0.7071067811865476), kwargs = {})
#   %erf_61 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_518,), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_61, 1), kwargs = {})
#   %mul_519 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_517, %add_143), kwargs = {})
#   %mul_520 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_519, 1.7015043497085571), kwargs = {})
#   %convolution_95 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_520, %view_210, %arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 1024) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eh/cehjaisknkkqgdj6kbjxkyvzkzs3nk7f6nypqzjno3cczsbhtjb7.py
# Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94, gelu_62, mul__75, out_95, x_se_52, x_se_53], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean]
# Source node to ATen node mapping:
#   gelu_60 => add_141, erf_60, mul_510, mul_511, mul_512
#   gelu_61 => add_143, erf_61, mul_517, mul_518, mul_519
#   gelu_62 => add_145, erf_62, mul_524, mul_525, mul_526
#   mul__73 => mul_513
#   mul__74 => mul_520
#   mul__75 => mul_527
#   out_92 => convolution_93
#   out_93 => convolution_94
#   out_94 => convolution_95
#   out_95 => convolution_96
#   x_13 => constant_pad_nd_7
#   x_se_52 => mean_14
#   x_se_53 => convolution_97
# Graph fragment:
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_503, %view_204, %arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.5), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.7071067811865476), kwargs = {})
#   %erf_60 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_511,), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_60, 1), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_510, %add_141), kwargs = {})
#   %mul_513 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_512, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_7 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_513, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_94 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_7, %view_207, %arg41_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 2), kwargs = {})
#   %mul_517 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, 0.5), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, 0.7071067811865476), kwargs = {})
#   %erf_61 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_518,), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_61, 1), kwargs = {})
#   %mul_519 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_517, %add_143), kwargs = {})
#   %mul_520 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_519, 1.7015043497085571), kwargs = {})
#   %convolution_95 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_520, %view_210, %arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %mul_524 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_95, 0.5), kwargs = {})
#   %mul_525 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_95, 0.7071067811865476), kwargs = {})
#   %erf_62 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_525,), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_62, 1), kwargs = {})
#   %mul_526 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_524, %add_145), kwargs = {})
#   %mul_527 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_526, 1.7015043497085571), kwargs = {})
#   %convolution_96 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_527, %view_213, %arg47_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_96, [2, 3], True), kwargs = {})
#   %convolution_97 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_14, %arg48_1, %arg49_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_29 = async_compile.triton('triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_29(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = 1024.0
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qc/cqc2yxkn44fejroo2eqafz7if7ygrilct7k2sxan7mkbumcz2y6h.py
# Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94, gelu_62, mul__75, out_95, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu]
# Source node to ATen node mapping:
#   gelu_60 => add_141, erf_60, mul_510, mul_511, mul_512
#   gelu_61 => add_143, erf_61, mul_517, mul_518, mul_519
#   gelu_62 => add_145, erf_62, mul_524, mul_525, mul_526
#   mul__73 => mul_513
#   mul__74 => mul_520
#   mul__75 => mul_527
#   out_92 => convolution_93
#   out_93 => convolution_94
#   out_94 => convolution_95
#   out_95 => convolution_96
#   x_13 => constant_pad_nd_7
#   x_se_52 => mean_14
#   x_se_53 => convolution_97
#   x_se_54 => relu_13
#   x_se_55 => convolution_98
# Graph fragment:
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_503, %view_204, %arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.5), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.7071067811865476), kwargs = {})
#   %erf_60 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_511,), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_60, 1), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_510, %add_141), kwargs = {})
#   %mul_513 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_512, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_7 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_513, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_94 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_7, %view_207, %arg41_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 2), kwargs = {})
#   %mul_517 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, 0.5), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, 0.7071067811865476), kwargs = {})
#   %erf_61 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_518,), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_61, 1), kwargs = {})
#   %mul_519 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_517, %add_143), kwargs = {})
#   %mul_520 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_519, 1.7015043497085571), kwargs = {})
#   %convolution_95 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_520, %view_210, %arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %mul_524 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_95, 0.5), kwargs = {})
#   %mul_525 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_95, 0.7071067811865476), kwargs = {})
#   %erf_62 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_525,), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_62, 1), kwargs = {})
#   %mul_526 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_524, %add_145), kwargs = {})
#   %mul_527 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_526, 1.7015043497085571), kwargs = {})
#   %convolution_96 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_527, %view_213, %arg47_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_96, [2, 3], True), kwargs = {})
#   %convolution_97 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_14, %arg48_1, %arg49_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_97,), kwargs = {})
#   %convolution_98 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg50_1, %arg51_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_30 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_30(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/e2/ce2ij52rbix2tscwh6v3ecspag2lt3rkmkfabxfrtluwt2ujskc6.py
# Topologically Sorted Source Nodes: [avg_pool2d_3, shortcut_5], Original ATen: [aten.avg_pool2d, aten.convolution]
# Source node to ATen node mapping:
#   avg_pool2d_3 => avg_pool2d_3
#   shortcut_5 => convolution_92
# Graph fragment:
#   %avg_pool2d_3 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_503, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_92 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_3, %view_201, %arg35_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_avg_pool2d_convolution_31 = async_compile.triton('triton_poi_fused_avg_pool2d_convolution_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_convolution_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + (2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + (2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ot/cot7lyfpmncmlsnefvmrkzdulxfs26z4oaytgwzl6xglqa4co7ox.py
# Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94, gelu_62, mul__75, out_95, x_se_52, x_se_53, x_se_54, x_se_55, sigmoid_13, mul_124, out_96, mul__76, mul_126, avg_pool2d_3, shortcut_5, out_97, gelu_63, mul__77, out_98, out_99], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu, aten.sigmoid, aten.avg_pool2d, aten.add]
# Source node to ATen node mapping:
#   avg_pool2d_3 => avg_pool2d_3
#   gelu_60 => add_141, erf_60, mul_510, mul_511, mul_512
#   gelu_61 => add_143, erf_61, mul_517, mul_518, mul_519
#   gelu_62 => add_145, erf_62, mul_524, mul_525, mul_526
#   gelu_63 => add_148, erf_63, mul_535, mul_536, mul_537
#   mul_124 => mul_531
#   mul_126 => mul_534
#   mul__73 => mul_513
#   mul__74 => mul_520
#   mul__75 => mul_527
#   mul__76 => mul_533
#   mul__77 => mul_538
#   out_92 => convolution_93
#   out_93 => convolution_94
#   out_94 => convolution_95
#   out_95 => convolution_96
#   out_96 => mul_532
#   out_97 => add_147
#   out_98 => mul_539
#   out_99 => convolution_99
#   shortcut_5 => convolution_92
#   sigmoid_13 => sigmoid_13
#   x_13 => constant_pad_nd_7
#   x_se_52 => mean_14
#   x_se_53 => convolution_97
#   x_se_54 => relu_13
#   x_se_55 => convolution_98
# Graph fragment:
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_503, %view_204, %arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.5), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.7071067811865476), kwargs = {})
#   %erf_60 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_511,), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_60, 1), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_510, %add_141), kwargs = {})
#   %mul_513 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_512, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_7 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_513, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_94 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_7, %view_207, %arg41_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 2), kwargs = {})
#   %mul_517 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, 0.5), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, 0.7071067811865476), kwargs = {})
#   %erf_61 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_518,), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_61, 1), kwargs = {})
#   %mul_519 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_517, %add_143), kwargs = {})
#   %mul_520 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_519, 1.7015043497085571), kwargs = {})
#   %convolution_95 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_520, %view_210, %arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %mul_524 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_95, 0.5), kwargs = {})
#   %mul_525 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_95, 0.7071067811865476), kwargs = {})
#   %erf_62 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_525,), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_62, 1), kwargs = {})
#   %mul_526 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_524, %add_145), kwargs = {})
#   %mul_527 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_526, 1.7015043497085571), kwargs = {})
#   %convolution_96 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_527, %view_213, %arg47_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_96, [2, 3], True), kwargs = {})
#   %convolution_97 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_14, %arg48_1, %arg49_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_97,), kwargs = {})
#   %convolution_98 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg50_1, %arg51_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_98,), kwargs = {})
#   %mul_531 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_96, %sigmoid_13), kwargs = {})
#   %mul_532 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_531, 2.0), kwargs = {})
#   %mul_533 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_532, %arg52_1), kwargs = {})
#   %mul_534 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_533, 0.2), kwargs = {})
#   %avg_pool2d_3 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_503, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_92 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_3, %view_201, %arg35_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_147 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_534, %convolution_92), kwargs = {})
#   %mul_535 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_147, 0.5), kwargs = {})
#   %mul_536 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_147, 0.7071067811865476), kwargs = {})
#   %erf_63 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_536,), kwargs = {})
#   %add_148 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_63, 1), kwargs = {})
#   %mul_537 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_535, %add_148), kwargs = {})
#   %mul_538 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_537, 1.7015043497085571), kwargs = {})
#   %mul_539 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_538, 0.9805806756909201), kwargs = {})
#   %convolution_99 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_539, %view_216, %arg55_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_32 = async_compile.triton('triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 1024) % 512
    x4 = (xindex // 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = 0.7071067811865476
    tmp22 = tmp18 * tmp21
    tmp23 = libdevice.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 1.7015043497085571
    tmp28 = tmp26 * tmp27
    tmp29 = 0.9805806756909201
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yw/cywz56gdxwc7flr42xkftaupd2vdgsiujux4rmu45h23ug6v3cj5.py
# Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100, gelu_65, mul__79, out_101, gelu_66, mul__80, out_102, x_se_56, x_se_57, x_se_58, x_se_59, sigmoid_14, mul_132, out_103, mul__81, mul_134, out_104, gelu_67, mul__82, out_105], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   gelu_63 => add_148, erf_63, mul_535, mul_536, mul_537
#   gelu_64 => add_150, erf_64, mul_543, mul_544, mul_545
#   gelu_65 => add_152, erf_65, mul_550, mul_551, mul_552
#   gelu_66 => add_154, erf_66, mul_557, mul_558, mul_559
#   gelu_67 => add_157, erf_67, mul_568, mul_569, mul_570
#   mul_132 => mul_564
#   mul_134 => mul_567
#   mul__77 => mul_538
#   mul__78 => mul_546
#   mul__79 => mul_553
#   mul__80 => mul_560
#   mul__81 => mul_566
#   mul__82 => mul_571
#   out_100 => convolution_100
#   out_101 => convolution_101
#   out_102 => convolution_102
#   out_103 => mul_565
#   out_104 => add_156
#   out_105 => mul_572
#   out_98 => mul_539
#   out_99 => convolution_99
#   sigmoid_14 => sigmoid_14
#   x_se_56 => mean_15
#   x_se_57 => convolution_103
#   x_se_58 => relu_14
#   x_se_59 => convolution_104
# Graph fragment:
#   %mul_535 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_147, 0.5), kwargs = {})
#   %mul_536 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_147, 0.7071067811865476), kwargs = {})
#   %erf_63 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_536,), kwargs = {})
#   %add_148 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_63, 1), kwargs = {})
#   %mul_537 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_535, %add_148), kwargs = {})
#   %mul_538 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_537, 1.7015043497085571), kwargs = {})
#   %mul_539 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_538, 0.9805806756909201), kwargs = {})
#   %convolution_99 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_539, %view_216, %arg55_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_543 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_99, 0.5), kwargs = {})
#   %mul_544 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_99, 0.7071067811865476), kwargs = {})
#   %erf_64 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_544,), kwargs = {})
#   %add_150 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_64, 1), kwargs = {})
#   %mul_545 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_543, %add_150), kwargs = {})
#   %mul_546 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_545, 1.7015043497085571), kwargs = {})
#   %convolution_100 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_546, %view_219, %arg58_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %mul_550 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_100, 0.5), kwargs = {})
#   %mul_551 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_100, 0.7071067811865476), kwargs = {})
#   %erf_65 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_551,), kwargs = {})
#   %add_152 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_65, 1), kwargs = {})
#   %mul_552 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_550, %add_152), kwargs = {})
#   %mul_553 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_552, 1.7015043497085571), kwargs = {})
#   %convolution_101 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_553, %view_222, %arg61_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %mul_557 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_101, 0.5), kwargs = {})
#   %mul_558 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_101, 0.7071067811865476), kwargs = {})
#   %erf_66 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_558,), kwargs = {})
#   %add_154 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_66, 1), kwargs = {})
#   %mul_559 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_557, %add_154), kwargs = {})
#   %mul_560 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_559, 1.7015043497085571), kwargs = {})
#   %convolution_102 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_560, %view_225, %arg64_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_102, [2, 3], True), kwargs = {})
#   %convolution_103 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_15, %arg65_1, %arg66_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_103,), kwargs = {})
#   %convolution_104 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg67_1, %arg68_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_14 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_104,), kwargs = {})
#   %mul_564 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_102, %sigmoid_14), kwargs = {})
#   %mul_565 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_564, 2.0), kwargs = {})
#   %mul_566 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_565, %arg69_1), kwargs = {})
#   %mul_567 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_566, 0.2), kwargs = {})
#   %add_156 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_567, %add_147), kwargs = {})
#   %mul_568 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_156, 0.5), kwargs = {})
#   %mul_569 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_156, 0.7071067811865476), kwargs = {})
#   %erf_67 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_569,), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_67, 1), kwargs = {})
#   %mul_570 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_568, %add_157), kwargs = {})
#   %mul_571 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_570, 1.7015043497085571), kwargs = {})
#   %mul_572 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_571, 0.9622504486493761), kwargs = {})
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_33 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 1024) % 512
    x4 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.9622504486493761
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ya/cya7dfje5rr5gaiidjucofvkiw3bmry7xzv2pha5rsop5gbindnx.py
# Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   gelu_68 => add_160, erf_68, mul_579, mul_580, mul_581
#   mul__83 => mul_582
#   out_106 => convolution_106
#   out_107 => convolution_107
#   x_14 => constant_pad_nd_8
# Graph fragment:
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_572, %view_231, %arg75_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_579 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.5), kwargs = {})
#   %mul_580 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.7071067811865476), kwargs = {})
#   %erf_68 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_580,), kwargs = {})
#   %add_160 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_68, 1), kwargs = {})
#   %mul_581 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_579, %add_160), kwargs = {})
#   %mul_582 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_581, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_8 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_582, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_8, %view_234, %arg78_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_34 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_gelu_mul_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_gelu_mul_34(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6690816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 33) % 33
    x0 = xindex % 33
    x4 = (xindex // 1089)
    x2 = (xindex // 1089) % 768
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 32, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (32*x1) + (1024*x4)), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = libdevice.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tmp17 = 1.7015043497085571
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tl.store(out_ptr0 + (x5), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ux/cuxx65cadou2k57w6brn4i52ichixovoh5t7el6xp4jtqkw4vys2.py
# Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   gelu_68 => add_160, erf_68, mul_579, mul_580, mul_581
#   gelu_69 => add_162, erf_69, mul_586, mul_587, mul_588
#   mul__83 => mul_582
#   mul__84 => mul_589
#   out_106 => convolution_106
#   out_107 => convolution_107
#   out_108 => convolution_108
#   x_14 => constant_pad_nd_8
# Graph fragment:
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_572, %view_231, %arg75_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_579 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.5), kwargs = {})
#   %mul_580 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.7071067811865476), kwargs = {})
#   %erf_68 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_580,), kwargs = {})
#   %add_160 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_68, 1), kwargs = {})
#   %mul_581 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_579, %add_160), kwargs = {})
#   %mul_582 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_581, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_8 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_582, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_8, %view_234, %arg78_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_586 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, 0.5), kwargs = {})
#   %mul_587 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, 0.7071067811865476), kwargs = {})
#   %erf_69 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_587,), kwargs = {})
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_69, 1), kwargs = {})
#   %mul_588 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_586, %add_162), kwargs = {})
#   %mul_589 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_588, 1.7015043497085571), kwargs = {})
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_589, %view_237, %arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 256) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/za/cza4vjsg6n2edj3hofv777biofmmlfgq2dwv53bw6ey5h26xdowi.py
# Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108, gelu_70, mul__85, out_109, x_se_60, x_se_61], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean]
# Source node to ATen node mapping:
#   gelu_68 => add_160, erf_68, mul_579, mul_580, mul_581
#   gelu_69 => add_162, erf_69, mul_586, mul_587, mul_588
#   gelu_70 => add_164, erf_70, mul_593, mul_594, mul_595
#   mul__83 => mul_582
#   mul__84 => mul_589
#   mul__85 => mul_596
#   out_106 => convolution_106
#   out_107 => convolution_107
#   out_108 => convolution_108
#   out_109 => convolution_109
#   x_14 => constant_pad_nd_8
#   x_se_60 => mean_16
#   x_se_61 => convolution_110
# Graph fragment:
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_572, %view_231, %arg75_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_579 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.5), kwargs = {})
#   %mul_580 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.7071067811865476), kwargs = {})
#   %erf_68 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_580,), kwargs = {})
#   %add_160 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_68, 1), kwargs = {})
#   %mul_581 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_579, %add_160), kwargs = {})
#   %mul_582 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_581, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_8 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_582, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_8, %view_234, %arg78_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_586 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, 0.5), kwargs = {})
#   %mul_587 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, 0.7071067811865476), kwargs = {})
#   %erf_69 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_587,), kwargs = {})
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_69, 1), kwargs = {})
#   %mul_588 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_586, %add_162), kwargs = {})
#   %mul_589 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_588, 1.7015043497085571), kwargs = {})
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_589, %view_237, %arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_593 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, 0.5), kwargs = {})
#   %mul_594 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, 0.7071067811865476), kwargs = {})
#   %erf_70 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_594,), kwargs = {})
#   %add_164 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_70, 1), kwargs = {})
#   %mul_595 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_593, %add_164), kwargs = {})
#   %mul_596 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_595, 1.7015043497085571), kwargs = {})
#   %convolution_109 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_596, %view_240, %arg84_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_109, [2, 3], True), kwargs = {})
#   %convolution_110 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_16, %arg85_1, %arg86_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36 = async_compile.triton('triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
    xnumel = 12288
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ic/cicjxobyli7kuvhtnrnrmhwnxkjuqazahj75bew6wvgrklzwttxa.py
# Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108, gelu_70, mul__85, out_109, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu]
# Source node to ATen node mapping:
#   gelu_68 => add_160, erf_68, mul_579, mul_580, mul_581
#   gelu_69 => add_162, erf_69, mul_586, mul_587, mul_588
#   gelu_70 => add_164, erf_70, mul_593, mul_594, mul_595
#   mul__83 => mul_582
#   mul__84 => mul_589
#   mul__85 => mul_596
#   out_106 => convolution_106
#   out_107 => convolution_107
#   out_108 => convolution_108
#   out_109 => convolution_109
#   x_14 => constant_pad_nd_8
#   x_se_60 => mean_16
#   x_se_61 => convolution_110
#   x_se_62 => relu_15
#   x_se_63 => convolution_111
# Graph fragment:
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_572, %view_231, %arg75_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_579 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.5), kwargs = {})
#   %mul_580 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.7071067811865476), kwargs = {})
#   %erf_68 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_580,), kwargs = {})
#   %add_160 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_68, 1), kwargs = {})
#   %mul_581 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_579, %add_160), kwargs = {})
#   %mul_582 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_581, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_8 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_582, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_8, %view_234, %arg78_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_586 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, 0.5), kwargs = {})
#   %mul_587 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, 0.7071067811865476), kwargs = {})
#   %erf_69 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_587,), kwargs = {})
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_69, 1), kwargs = {})
#   %mul_588 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_586, %add_162), kwargs = {})
#   %mul_589 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_588, 1.7015043497085571), kwargs = {})
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_589, %view_237, %arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_593 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, 0.5), kwargs = {})
#   %mul_594 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, 0.7071067811865476), kwargs = {})
#   %erf_70 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_594,), kwargs = {})
#   %add_164 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_70, 1), kwargs = {})
#   %mul_595 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_593, %add_164), kwargs = {})
#   %mul_596 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_595, 1.7015043497085571), kwargs = {})
#   %convolution_109 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_596, %view_240, %arg84_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_109, [2, 3], True), kwargs = {})
#   %convolution_110 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_16, %arg85_1, %arg86_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_110,), kwargs = {})
#   %convolution_111 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg87_1, %arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ka/cka4sjbmxlpw45ajj3c43xbjosm3orlo7km36nrwkfbdu5rc6d2l.py
# Topologically Sorted Source Nodes: [avg_pool2d_4, shortcut_6], Original ATen: [aten.avg_pool2d, aten.convolution]
# Source node to ATen node mapping:
#   avg_pool2d_4 => avg_pool2d_4
#   shortcut_6 => convolution_105
# Graph fragment:
#   %avg_pool2d_4 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_572, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_105 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_4, %view_228, %arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_avg_pool2d_convolution_38 = async_compile.triton('triton_poi_fused_avg_pool2d_convolution_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_convolution_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + (2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + (2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rw/crw2swt3i52aqbo54wf6h25i2klyddpsiplhqjemjc5ifs26atpx.py
# Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108, gelu_70, mul__85, out_109, x_se_60, x_se_61, x_se_62, x_se_63, sigmoid_15, mul_141, out_110, mul__86, mul_143, avg_pool2d_4, shortcut_6, out_111, gelu_71, mul__87, out_112, out_113], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu, aten.sigmoid, aten.avg_pool2d, aten.add]
# Source node to ATen node mapping:
#   avg_pool2d_4 => avg_pool2d_4
#   gelu_68 => add_160, erf_68, mul_579, mul_580, mul_581
#   gelu_69 => add_162, erf_69, mul_586, mul_587, mul_588
#   gelu_70 => add_164, erf_70, mul_593, mul_594, mul_595
#   gelu_71 => add_167, erf_71, mul_604, mul_605, mul_606
#   mul_141 => mul_600
#   mul_143 => mul_603
#   mul__83 => mul_582
#   mul__84 => mul_589
#   mul__85 => mul_596
#   mul__86 => mul_602
#   mul__87 => mul_607
#   out_106 => convolution_106
#   out_107 => convolution_107
#   out_108 => convolution_108
#   out_109 => convolution_109
#   out_110 => mul_601
#   out_111 => add_166
#   out_112 => mul_608
#   out_113 => convolution_112
#   shortcut_6 => convolution_105
#   sigmoid_15 => sigmoid_15
#   x_14 => constant_pad_nd_8
#   x_se_60 => mean_16
#   x_se_61 => convolution_110
#   x_se_62 => relu_15
#   x_se_63 => convolution_111
# Graph fragment:
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_572, %view_231, %arg75_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_579 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.5), kwargs = {})
#   %mul_580 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, 0.7071067811865476), kwargs = {})
#   %erf_68 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_580,), kwargs = {})
#   %add_160 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_68, 1), kwargs = {})
#   %mul_581 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_579, %add_160), kwargs = {})
#   %mul_582 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_581, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_8 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_582, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_8, %view_234, %arg78_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_586 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, 0.5), kwargs = {})
#   %mul_587 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, 0.7071067811865476), kwargs = {})
#   %erf_69 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_587,), kwargs = {})
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_69, 1), kwargs = {})
#   %mul_588 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_586, %add_162), kwargs = {})
#   %mul_589 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_588, 1.7015043497085571), kwargs = {})
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_589, %view_237, %arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_593 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, 0.5), kwargs = {})
#   %mul_594 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, 0.7071067811865476), kwargs = {})
#   %erf_70 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_594,), kwargs = {})
#   %add_164 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_70, 1), kwargs = {})
#   %mul_595 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_593, %add_164), kwargs = {})
#   %mul_596 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_595, 1.7015043497085571), kwargs = {})
#   %convolution_109 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_596, %view_240, %arg84_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_109, [2, 3], True), kwargs = {})
#   %convolution_110 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_16, %arg85_1, %arg86_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_110,), kwargs = {})
#   %convolution_111 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg87_1, %arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_111,), kwargs = {})
#   %mul_600 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_109, %sigmoid_15), kwargs = {})
#   %mul_601 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_600, 2.0), kwargs = {})
#   %mul_602 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_601, %arg89_1), kwargs = {})
#   %mul_603 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_602, 0.2), kwargs = {})
#   %avg_pool2d_4 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_572, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_105 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_4, %view_228, %arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_166 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_603, %convolution_105), kwargs = {})
#   %mul_604 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_166, 0.5), kwargs = {})
#   %mul_605 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_166, 0.7071067811865476), kwargs = {})
#   %erf_71 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_605,), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_71, 1), kwargs = {})
#   %mul_606 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_604, %add_167), kwargs = {})
#   %mul_607 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_606, 1.7015043497085571), kwargs = {})
#   %mul_608 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_607, 0.9805806756909201), kwargs = {})
#   %convolution_112 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_608, %view_243, %arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_39 = async_compile.triton('triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 256) % 1536
    x4 = (xindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = 0.7071067811865476
    tmp22 = tmp18 * tmp21
    tmp23 = libdevice.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 1.7015043497085571
    tmp28 = tmp26 * tmp27
    tmp29 = 0.9805806756909201
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ny/cnykjd2g5gcs4ob5es73qqhzcbl6nmkhbea6bomed7rtehxt3i3b.py
# Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114, gelu_73, mul__89, out_115, gelu_74, mul__90, out_116, x_se_64, x_se_65, x_se_66, x_se_67, sigmoid_16, mul_149, out_117, mul__91, mul_151, out_118, gelu_75, mul__92, out_119, out_120], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   gelu_71 => add_167, erf_71, mul_604, mul_605, mul_606
#   gelu_72 => add_169, erf_72, mul_612, mul_613, mul_614
#   gelu_73 => add_171, erf_73, mul_619, mul_620, mul_621
#   gelu_74 => add_173, erf_74, mul_626, mul_627, mul_628
#   gelu_75 => add_176, erf_75, mul_637, mul_638, mul_639
#   mul_149 => mul_633
#   mul_151 => mul_636
#   mul__87 => mul_607
#   mul__88 => mul_615
#   mul__89 => mul_622
#   mul__90 => mul_629
#   mul__91 => mul_635
#   mul__92 => mul_640
#   out_112 => mul_608
#   out_113 => convolution_112
#   out_114 => convolution_113
#   out_115 => convolution_114
#   out_116 => convolution_115
#   out_117 => mul_634
#   out_118 => add_175
#   out_119 => mul_641
#   out_120 => convolution_118
#   sigmoid_16 => sigmoid_16
#   x_se_64 => mean_17
#   x_se_65 => convolution_116
#   x_se_66 => relu_16
#   x_se_67 => convolution_117
# Graph fragment:
#   %mul_604 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_166, 0.5), kwargs = {})
#   %mul_605 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_166, 0.7071067811865476), kwargs = {})
#   %erf_71 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_605,), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_71, 1), kwargs = {})
#   %mul_606 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_604, %add_167), kwargs = {})
#   %mul_607 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_606, 1.7015043497085571), kwargs = {})
#   %mul_608 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_607, 0.9805806756909201), kwargs = {})
#   %convolution_112 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_608, %view_243, %arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_612 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_112, 0.5), kwargs = {})
#   %mul_613 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_112, 0.7071067811865476), kwargs = {})
#   %erf_72 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_613,), kwargs = {})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_72, 1), kwargs = {})
#   %mul_614 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_612, %add_169), kwargs = {})
#   %mul_615 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_614, 1.7015043497085571), kwargs = {})
#   %convolution_113 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_615, %view_246, %arg95_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_619 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_113, 0.5), kwargs = {})
#   %mul_620 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_113, 0.7071067811865476), kwargs = {})
#   %erf_73 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_620,), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_73, 1), kwargs = {})
#   %mul_621 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_619, %add_171), kwargs = {})
#   %mul_622 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_621, 1.7015043497085571), kwargs = {})
#   %convolution_114 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_622, %view_249, %arg98_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_626 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_114, 0.5), kwargs = {})
#   %mul_627 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_114, 0.7071067811865476), kwargs = {})
#   %erf_74 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_627,), kwargs = {})
#   %add_173 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_74, 1), kwargs = {})
#   %mul_628 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_626, %add_173), kwargs = {})
#   %mul_629 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_628, 1.7015043497085571), kwargs = {})
#   %convolution_115 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_629, %view_252, %arg101_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_115, [2, 3], True), kwargs = {})
#   %convolution_116 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_17, %arg102_1, %arg103_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_116,), kwargs = {})
#   %convolution_117 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_16, %arg104_1, %arg105_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_117,), kwargs = {})
#   %mul_633 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_115, %sigmoid_16), kwargs = {})
#   %mul_634 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_633, 2.0), kwargs = {})
#   %mul_635 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_634, %arg106_1), kwargs = {})
#   %mul_636 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_635, 0.2), kwargs = {})
#   %add_175 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_636, %add_166), kwargs = {})
#   %mul_637 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_175, 0.5), kwargs = {})
#   %mul_638 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_175, 0.7071067811865476), kwargs = {})
#   %erf_75 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_638,), kwargs = {})
#   %add_176 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_75, 1), kwargs = {})
#   %mul_639 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_637, %add_176), kwargs = {})
#   %mul_640 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_639, 1.7015043497085571), kwargs = {})
#   %mul_641 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_640, 0.9622504486493761), kwargs = {})
#   %convolution_118 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_641, %view_255, %arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_40 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_40', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 256) % 1536
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.9622504486493761
    tmp28 = tmp26 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qu/cquys73abu2oo4n5v4pw5xfftqzxax2pbm5lilhtkwcnfdqpnxvt.py
# Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121, gelu_77, mul__94, out_122, gelu_78, mul__95, out_123, x_se_68, x_se_69, x_se_70, x_se_71, sigmoid_17, mul_157, out_124, mul__96, mul_159, out_125, gelu_79, mul__97, out_126, out_127], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   gelu_75 => add_176, erf_75, mul_637, mul_638, mul_639
#   gelu_76 => add_178, erf_76, mul_645, mul_646, mul_647
#   gelu_77 => add_180, erf_77, mul_652, mul_653, mul_654
#   gelu_78 => add_182, erf_78, mul_659, mul_660, mul_661
#   gelu_79 => add_185, erf_79, mul_670, mul_671, mul_672
#   mul_157 => mul_666
#   mul_159 => mul_669
#   mul__92 => mul_640
#   mul__93 => mul_648
#   mul__94 => mul_655
#   mul__95 => mul_662
#   mul__96 => mul_668
#   mul__97 => mul_673
#   out_119 => mul_641
#   out_120 => convolution_118
#   out_121 => convolution_119
#   out_122 => convolution_120
#   out_123 => convolution_121
#   out_124 => mul_667
#   out_125 => add_184
#   out_126 => mul_674
#   out_127 => convolution_124
#   sigmoid_17 => sigmoid_17
#   x_se_68 => mean_18
#   x_se_69 => convolution_122
#   x_se_70 => relu_17
#   x_se_71 => convolution_123
# Graph fragment:
#   %mul_637 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_175, 0.5), kwargs = {})
#   %mul_638 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_175, 0.7071067811865476), kwargs = {})
#   %erf_75 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_638,), kwargs = {})
#   %add_176 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_75, 1), kwargs = {})
#   %mul_639 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_637, %add_176), kwargs = {})
#   %mul_640 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_639, 1.7015043497085571), kwargs = {})
#   %mul_641 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_640, 0.9622504486493761), kwargs = {})
#   %convolution_118 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_641, %view_255, %arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_645 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_118, 0.5), kwargs = {})
#   %mul_646 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_118, 0.7071067811865476), kwargs = {})
#   %erf_76 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_646,), kwargs = {})
#   %add_178 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_76, 1), kwargs = {})
#   %mul_647 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_645, %add_178), kwargs = {})
#   %mul_648 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_647, 1.7015043497085571), kwargs = {})
#   %convolution_119 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_648, %view_258, %arg112_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_652 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_119, 0.5), kwargs = {})
#   %mul_653 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_119, 0.7071067811865476), kwargs = {})
#   %erf_77 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_653,), kwargs = {})
#   %add_180 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_77, 1), kwargs = {})
#   %mul_654 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_652, %add_180), kwargs = {})
#   %mul_655 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_654, 1.7015043497085571), kwargs = {})
#   %convolution_120 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_655, %view_261, %arg115_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_659 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_120, 0.5), kwargs = {})
#   %mul_660 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_120, 0.7071067811865476), kwargs = {})
#   %erf_78 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_660,), kwargs = {})
#   %add_182 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_78, 1), kwargs = {})
#   %mul_661 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_659, %add_182), kwargs = {})
#   %mul_662 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_661, 1.7015043497085571), kwargs = {})
#   %convolution_121 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_662, %view_264, %arg118_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_121, [2, 3], True), kwargs = {})
#   %convolution_122 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_18, %arg119_1, %arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_122,), kwargs = {})
#   %convolution_123 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg121_1, %arg122_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_123,), kwargs = {})
#   %mul_666 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_121, %sigmoid_17), kwargs = {})
#   %mul_667 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_666, 2.0), kwargs = {})
#   %mul_668 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_667, %arg123_1), kwargs = {})
#   %mul_669 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_668, 0.2), kwargs = {})
#   %add_184 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_669, %add_175), kwargs = {})
#   %mul_670 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_184, 0.5), kwargs = {})
#   %mul_671 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_184, 0.7071067811865476), kwargs = {})
#   %erf_79 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_671,), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_79, 1), kwargs = {})
#   %mul_672 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_670, %add_185), kwargs = {})
#   %mul_673 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_672, 1.7015043497085571), kwargs = {})
#   %mul_674 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_673, 0.9449111825230679), kwargs = {})
#   %convolution_124 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_674, %view_267, %arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_41 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_41', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 256) % 1536
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.9449111825230679
    tmp28 = tmp26 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5s/c5selpypbtvhg2igmej2biq4bctsul6aeow2wgaxfjnpl3wc2jdn.py
# Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128, gelu_81, mul__99, out_129, gelu_82, mul__100, out_130, x_se_72, x_se_73, x_se_74, x_se_75, sigmoid_18, mul_165, out_131, mul__101, mul_167, out_132, gelu_83, mul__102, out_133, out_134], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   gelu_79 => add_185, erf_79, mul_670, mul_671, mul_672
#   gelu_80 => add_187, erf_80, mul_678, mul_679, mul_680
#   gelu_81 => add_189, erf_81, mul_685, mul_686, mul_687
#   gelu_82 => add_191, erf_82, mul_692, mul_693, mul_694
#   gelu_83 => add_194, erf_83, mul_703, mul_704, mul_705
#   mul_165 => mul_699
#   mul_167 => mul_702
#   mul__100 => mul_695
#   mul__101 => mul_701
#   mul__102 => mul_706
#   mul__97 => mul_673
#   mul__98 => mul_681
#   mul__99 => mul_688
#   out_126 => mul_674
#   out_127 => convolution_124
#   out_128 => convolution_125
#   out_129 => convolution_126
#   out_130 => convolution_127
#   out_131 => mul_700
#   out_132 => add_193
#   out_133 => mul_707
#   out_134 => convolution_130
#   sigmoid_18 => sigmoid_18
#   x_se_72 => mean_19
#   x_se_73 => convolution_128
#   x_se_74 => relu_18
#   x_se_75 => convolution_129
# Graph fragment:
#   %mul_670 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_184, 0.5), kwargs = {})
#   %mul_671 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_184, 0.7071067811865476), kwargs = {})
#   %erf_79 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_671,), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_79, 1), kwargs = {})
#   %mul_672 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_670, %add_185), kwargs = {})
#   %mul_673 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_672, 1.7015043497085571), kwargs = {})
#   %mul_674 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_673, 0.9449111825230679), kwargs = {})
#   %convolution_124 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_674, %view_267, %arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_678 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_124, 0.5), kwargs = {})
#   %mul_679 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_124, 0.7071067811865476), kwargs = {})
#   %erf_80 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_679,), kwargs = {})
#   %add_187 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_80, 1), kwargs = {})
#   %mul_680 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_678, %add_187), kwargs = {})
#   %mul_681 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_680, 1.7015043497085571), kwargs = {})
#   %convolution_125 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_681, %view_270, %arg129_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_685 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_125, 0.5), kwargs = {})
#   %mul_686 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_125, 0.7071067811865476), kwargs = {})
#   %erf_81 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_686,), kwargs = {})
#   %add_189 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_81, 1), kwargs = {})
#   %mul_687 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_685, %add_189), kwargs = {})
#   %mul_688 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_687, 1.7015043497085571), kwargs = {})
#   %convolution_126 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_688, %view_273, %arg132_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_692 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_126, 0.5), kwargs = {})
#   %mul_693 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_126, 0.7071067811865476), kwargs = {})
#   %erf_82 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_693,), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_82, 1), kwargs = {})
#   %mul_694 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_692, %add_191), kwargs = {})
#   %mul_695 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_694, 1.7015043497085571), kwargs = {})
#   %convolution_127 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_695, %view_276, %arg135_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_127, [2, 3], True), kwargs = {})
#   %convolution_128 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_19, %arg136_1, %arg137_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_128,), kwargs = {})
#   %convolution_129 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %arg138_1, %arg139_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_18 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_129,), kwargs = {})
#   %mul_699 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_127, %sigmoid_18), kwargs = {})
#   %mul_700 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_699, 2.0), kwargs = {})
#   %mul_701 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_700, %arg140_1), kwargs = {})
#   %mul_702 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_701, 0.2), kwargs = {})
#   %add_193 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_702, %add_184), kwargs = {})
#   %mul_703 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_193, 0.5), kwargs = {})
#   %mul_704 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_193, 0.7071067811865476), kwargs = {})
#   %erf_83 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_704,), kwargs = {})
#   %add_194 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_83, 1), kwargs = {})
#   %mul_705 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_703, %add_194), kwargs = {})
#   %mul_706 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_705, 1.7015043497085571), kwargs = {})
#   %mul_707 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_706, 0.9284766908852592), kwargs = {})
#   %convolution_130 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_707, %view_279, %arg143_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_42 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_42', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 256) % 1536
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.9284766908852592
    tmp28 = tmp26 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mb/cmbi5qjlajpsyupftq5xxnof2lg6azeplr6j7h5kzoxk6gpkcz3x.py
# Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135, gelu_85, mul__104, out_136, gelu_86, mul__105, out_137, x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, mul_173, out_138, mul__106, mul_175, out_139, gelu_87, mul__107, out_140, out_141], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   gelu_83 => add_194, erf_83, mul_703, mul_704, mul_705
#   gelu_84 => add_196, erf_84, mul_711, mul_712, mul_713
#   gelu_85 => add_198, erf_85, mul_718, mul_719, mul_720
#   gelu_86 => add_200, erf_86, mul_725, mul_726, mul_727
#   gelu_87 => add_203, erf_87, mul_736, mul_737, mul_738
#   mul_173 => mul_732
#   mul_175 => mul_735
#   mul__102 => mul_706
#   mul__103 => mul_714
#   mul__104 => mul_721
#   mul__105 => mul_728
#   mul__106 => mul_734
#   mul__107 => mul_739
#   out_133 => mul_707
#   out_134 => convolution_130
#   out_135 => convolution_131
#   out_136 => convolution_132
#   out_137 => convolution_133
#   out_138 => mul_733
#   out_139 => add_202
#   out_140 => mul_740
#   out_141 => convolution_136
#   sigmoid_19 => sigmoid_19
#   x_se_76 => mean_20
#   x_se_77 => convolution_134
#   x_se_78 => relu_19
#   x_se_79 => convolution_135
# Graph fragment:
#   %mul_703 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_193, 0.5), kwargs = {})
#   %mul_704 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_193, 0.7071067811865476), kwargs = {})
#   %erf_83 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_704,), kwargs = {})
#   %add_194 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_83, 1), kwargs = {})
#   %mul_705 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_703, %add_194), kwargs = {})
#   %mul_706 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_705, 1.7015043497085571), kwargs = {})
#   %mul_707 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_706, 0.9284766908852592), kwargs = {})
#   %convolution_130 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_707, %view_279, %arg143_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_711 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_130, 0.5), kwargs = {})
#   %mul_712 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_130, 0.7071067811865476), kwargs = {})
#   %erf_84 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_712,), kwargs = {})
#   %add_196 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_84, 1), kwargs = {})
#   %mul_713 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_711, %add_196), kwargs = {})
#   %mul_714 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_713, 1.7015043497085571), kwargs = {})
#   %convolution_131 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_714, %view_282, %arg146_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_718 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_131, 0.5), kwargs = {})
#   %mul_719 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_131, 0.7071067811865476), kwargs = {})
#   %erf_85 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_719,), kwargs = {})
#   %add_198 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_85, 1), kwargs = {})
#   %mul_720 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_718, %add_198), kwargs = {})
#   %mul_721 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_720, 1.7015043497085571), kwargs = {})
#   %convolution_132 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_721, %view_285, %arg149_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_725 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_132, 0.5), kwargs = {})
#   %mul_726 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_132, 0.7071067811865476), kwargs = {})
#   %erf_86 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_726,), kwargs = {})
#   %add_200 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_86, 1), kwargs = {})
#   %mul_727 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_725, %add_200), kwargs = {})
#   %mul_728 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_727, 1.7015043497085571), kwargs = {})
#   %convolution_133 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_728, %view_288, %arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_133, [2, 3], True), kwargs = {})
#   %convolution_134 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_20, %arg153_1, %arg154_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_134,), kwargs = {})
#   %convolution_135 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg155_1, %arg156_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_19 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_135,), kwargs = {})
#   %mul_732 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_133, %sigmoid_19), kwargs = {})
#   %mul_733 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_732, 2.0), kwargs = {})
#   %mul_734 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_733, %arg157_1), kwargs = {})
#   %mul_735 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_734, 0.2), kwargs = {})
#   %add_202 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_735, %add_193), kwargs = {})
#   %mul_736 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_202, 0.5), kwargs = {})
#   %mul_737 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_202, 0.7071067811865476), kwargs = {})
#   %erf_87 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_737,), kwargs = {})
#   %add_203 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_87, 1), kwargs = {})
#   %mul_738 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_736, %add_203), kwargs = {})
#   %mul_739 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_738, 1.7015043497085571), kwargs = {})
#   %mul_740 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_739, 0.9128709291752768), kwargs = {})
#   %convolution_136 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_740, %view_291, %arg160_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_43 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_43', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 256) % 1536
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.9128709291752768
    tmp28 = tmp26 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/la/claca3w5u2ijq6snowb7pudqmuwco4r5t5ivjfb5m6tnixc5dfta.py
# Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142, gelu_89, mul__109, out_143, gelu_90, mul__110, out_144, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, mul_181, out_145, mul__111, mul_183, out_146, gelu_91, mul__112, out_147], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   gelu_87 => add_203, erf_87, mul_736, mul_737, mul_738
#   gelu_88 => add_205, erf_88, mul_744, mul_745, mul_746
#   gelu_89 => add_207, erf_89, mul_751, mul_752, mul_753
#   gelu_90 => add_209, erf_90, mul_758, mul_759, mul_760
#   gelu_91 => add_212, erf_91, mul_769, mul_770, mul_771
#   mul_181 => mul_765
#   mul_183 => mul_768
#   mul__107 => mul_739
#   mul__108 => mul_747
#   mul__109 => mul_754
#   mul__110 => mul_761
#   mul__111 => mul_767
#   mul__112 => mul_772
#   out_140 => mul_740
#   out_141 => convolution_136
#   out_142 => convolution_137
#   out_143 => convolution_138
#   out_144 => convolution_139
#   out_145 => mul_766
#   out_146 => add_211
#   out_147 => mul_773
#   sigmoid_20 => sigmoid_20
#   x_se_80 => mean_21
#   x_se_81 => convolution_140
#   x_se_82 => relu_20
#   x_se_83 => convolution_141
# Graph fragment:
#   %mul_736 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_202, 0.5), kwargs = {})
#   %mul_737 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_202, 0.7071067811865476), kwargs = {})
#   %erf_87 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_737,), kwargs = {})
#   %add_203 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_87, 1), kwargs = {})
#   %mul_738 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_736, %add_203), kwargs = {})
#   %mul_739 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_738, 1.7015043497085571), kwargs = {})
#   %mul_740 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_739, 0.9128709291752768), kwargs = {})
#   %convolution_136 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_740, %view_291, %arg160_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_744 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_136, 0.5), kwargs = {})
#   %mul_745 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_136, 0.7071067811865476), kwargs = {})
#   %erf_88 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_745,), kwargs = {})
#   %add_205 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_88, 1), kwargs = {})
#   %mul_746 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_744, %add_205), kwargs = {})
#   %mul_747 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_746, 1.7015043497085571), kwargs = {})
#   %convolution_137 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_747, %view_294, %arg163_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_751 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_137, 0.5), kwargs = {})
#   %mul_752 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_137, 0.7071067811865476), kwargs = {})
#   %erf_89 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_752,), kwargs = {})
#   %add_207 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_89, 1), kwargs = {})
#   %mul_753 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_751, %add_207), kwargs = {})
#   %mul_754 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_753, 1.7015043497085571), kwargs = {})
#   %convolution_138 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_754, %view_297, %arg166_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_758 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_138, 0.5), kwargs = {})
#   %mul_759 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_138, 0.7071067811865476), kwargs = {})
#   %erf_90 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_759,), kwargs = {})
#   %add_209 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_90, 1), kwargs = {})
#   %mul_760 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_758, %add_209), kwargs = {})
#   %mul_761 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_760, 1.7015043497085571), kwargs = {})
#   %convolution_139 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_761, %view_300, %arg169_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_139, [2, 3], True), kwargs = {})
#   %convolution_140 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_21, %arg170_1, %arg171_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_140,), kwargs = {})
#   %convolution_141 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %arg172_1, %arg173_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_141,), kwargs = {})
#   %mul_765 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_139, %sigmoid_20), kwargs = {})
#   %mul_766 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_765, 2.0), kwargs = {})
#   %mul_767 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_766, %arg174_1), kwargs = {})
#   %mul_768 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_767, 0.2), kwargs = {})
#   %add_211 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_768, %add_202), kwargs = {})
#   %mul_769 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_211, 0.5), kwargs = {})
#   %mul_770 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_211, 0.7071067811865476), kwargs = {})
#   %erf_91 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_770,), kwargs = {})
#   %add_212 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_91, 1), kwargs = {})
#   %mul_771 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_769, %add_212), kwargs = {})
#   %mul_772 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_771, 1.7015043497085571), kwargs = {})
#   %mul_773 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_772, 0.8980265101338745), kwargs = {})
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_44 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 256) % 1536
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.8980265101338745
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oe/coeij4tlkpno7tytnu2dkc5lgcqb5k2vbgcnrlzzqmyzergzut46.py
# Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   gelu_92 => add_215, erf_92, mul_780, mul_781, mul_782
#   mul__113 => mul_783
#   out_148 => convolution_143
#   out_149 => convolution_144
#   x_15 => constant_pad_nd_9
# Graph fragment:
#   %convolution_143 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_773, %view_306, %arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_780 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, 0.5), kwargs = {})
#   %mul_781 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, 0.7071067811865476), kwargs = {})
#   %erf_92 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_781,), kwargs = {})
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_92, 1), kwargs = {})
#   %mul_782 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_780, %add_215), kwargs = {})
#   %mul_783 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_782, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_9 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_783, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_144 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_9, %view_309, %arg183_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_45 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_gelu_mul_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_gelu_mul_45(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1775616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 17) % 17
    x0 = xindex % 17
    x4 = (xindex // 289)
    x2 = (xindex // 289) % 768
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 16, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (16*x1) + (256*x4)), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = libdevice.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tmp17 = 1.7015043497085571
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tl.store(out_ptr0 + (x5), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6k/c6k5l77svzghlptiud3hcz5xdthi7v6uakgwr5f3qgvabau6uilt.py
# Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   gelu_92 => add_215, erf_92, mul_780, mul_781, mul_782
#   gelu_93 => add_217, erf_93, mul_787, mul_788, mul_789
#   mul__113 => mul_783
#   mul__114 => mul_790
#   out_148 => convolution_143
#   out_149 => convolution_144
#   out_150 => convolution_145
#   x_15 => constant_pad_nd_9
# Graph fragment:
#   %convolution_143 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_773, %view_306, %arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_780 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, 0.5), kwargs = {})
#   %mul_781 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, 0.7071067811865476), kwargs = {})
#   %erf_92 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_781,), kwargs = {})
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_92, 1), kwargs = {})
#   %mul_782 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_780, %add_215), kwargs = {})
#   %mul_783 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_782, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_9 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_783, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_144 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_9, %view_309, %arg183_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_787 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_144, 0.5), kwargs = {})
#   %mul_788 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_144, 0.7071067811865476), kwargs = {})
#   %erf_93 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_788,), kwargs = {})
#   %add_217 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_93, 1), kwargs = {})
#   %mul_789 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_787, %add_217), kwargs = {})
#   %mul_790 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_789, 1.7015043497085571), kwargs = {})
#   %convolution_145 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_790, %view_312, %arg186_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 64) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cy/ccyzsizmyfvdkg7jom7fqdee5tcbcumhm6uk6lxuivqcu4uh45wn.py
# Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150, gelu_94, mul__115, out_151, x_se_84, x_se_85], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean]
# Source node to ATen node mapping:
#   gelu_92 => add_215, erf_92, mul_780, mul_781, mul_782
#   gelu_93 => add_217, erf_93, mul_787, mul_788, mul_789
#   gelu_94 => add_219, erf_94, mul_794, mul_795, mul_796
#   mul__113 => mul_783
#   mul__114 => mul_790
#   mul__115 => mul_797
#   out_148 => convolution_143
#   out_149 => convolution_144
#   out_150 => convolution_145
#   out_151 => convolution_146
#   x_15 => constant_pad_nd_9
#   x_se_84 => mean_22
#   x_se_85 => convolution_147
# Graph fragment:
#   %convolution_143 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_773, %view_306, %arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_780 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, 0.5), kwargs = {})
#   %mul_781 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, 0.7071067811865476), kwargs = {})
#   %erf_92 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_781,), kwargs = {})
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_92, 1), kwargs = {})
#   %mul_782 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_780, %add_215), kwargs = {})
#   %mul_783 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_782, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_9 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_783, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_144 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_9, %view_309, %arg183_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_787 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_144, 0.5), kwargs = {})
#   %mul_788 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_144, 0.7071067811865476), kwargs = {})
#   %erf_93 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_788,), kwargs = {})
#   %add_217 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_93, 1), kwargs = {})
#   %mul_789 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_787, %add_217), kwargs = {})
#   %mul_790 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_789, 1.7015043497085571), kwargs = {})
#   %convolution_145 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_790, %view_312, %arg186_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_794 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_145, 0.5), kwargs = {})
#   %mul_795 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_145, 0.7071067811865476), kwargs = {})
#   %erf_94 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_795,), kwargs = {})
#   %add_219 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_94, 1), kwargs = {})
#   %mul_796 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_794, %add_219), kwargs = {})
#   %mul_797 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_796, 1.7015043497085571), kwargs = {})
#   %convolution_146 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_797, %view_315, %arg189_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_146, [2, 3], True), kwargs = {})
#   %convolution_147 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_22, %arg190_1, %arg191_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_47 = async_compile.triton('triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_47(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 64.0
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6w/c6wpi2ho6fpoicsfu6sjgzlnznpujdgchsyt6nwb5au7k5mkw7ad.py
# Topologically Sorted Source Nodes: [avg_pool2d_5, shortcut_7], Original ATen: [aten.avg_pool2d, aten.convolution]
# Source node to ATen node mapping:
#   avg_pool2d_5 => avg_pool2d_5
#   shortcut_7 => convolution_142
# Graph fragment:
#   %avg_pool2d_5 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_773, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_142 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_5, %view_303, %arg177_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_avg_pool2d_convolution_48 = async_compile.triton('triton_poi_fused_avg_pool2d_convolution_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_convolution_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 8
    x1 = (xindex // 8)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + (2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + (2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6z/c6z3mtn2qqexe644q5w2dk76tishjuz3qmpoazmm3clqiis4ikct.py
# Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150, gelu_94, mul__115, out_151, x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, mul_190, out_152, mul__116, mul_192, avg_pool2d_5, shortcut_7, out_153, gelu_95, mul__117, out_154, out_155], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu, aten.sigmoid, aten.avg_pool2d, aten.add]
# Source node to ATen node mapping:
#   avg_pool2d_5 => avg_pool2d_5
#   gelu_92 => add_215, erf_92, mul_780, mul_781, mul_782
#   gelu_93 => add_217, erf_93, mul_787, mul_788, mul_789
#   gelu_94 => add_219, erf_94, mul_794, mul_795, mul_796
#   gelu_95 => add_222, erf_95, mul_805, mul_806, mul_807
#   mul_190 => mul_801
#   mul_192 => mul_804
#   mul__113 => mul_783
#   mul__114 => mul_790
#   mul__115 => mul_797
#   mul__116 => mul_803
#   mul__117 => mul_808
#   out_148 => convolution_143
#   out_149 => convolution_144
#   out_150 => convolution_145
#   out_151 => convolution_146
#   out_152 => mul_802
#   out_153 => add_221
#   out_154 => mul_809
#   out_155 => convolution_149
#   shortcut_7 => convolution_142
#   sigmoid_21 => sigmoid_21
#   x_15 => constant_pad_nd_9
#   x_se_84 => mean_22
#   x_se_85 => convolution_147
#   x_se_86 => relu_21
#   x_se_87 => convolution_148
# Graph fragment:
#   %convolution_143 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_773, %view_306, %arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_780 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, 0.5), kwargs = {})
#   %mul_781 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, 0.7071067811865476), kwargs = {})
#   %erf_92 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_781,), kwargs = {})
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_92, 1), kwargs = {})
#   %mul_782 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_780, %add_215), kwargs = {})
#   %mul_783 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_782, 1.7015043497085571), kwargs = {})
#   %constant_pad_nd_9 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_783, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_144 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_9, %view_309, %arg183_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_787 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_144, 0.5), kwargs = {})
#   %mul_788 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_144, 0.7071067811865476), kwargs = {})
#   %erf_93 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_788,), kwargs = {})
#   %add_217 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_93, 1), kwargs = {})
#   %mul_789 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_787, %add_217), kwargs = {})
#   %mul_790 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_789, 1.7015043497085571), kwargs = {})
#   %convolution_145 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_790, %view_312, %arg186_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_794 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_145, 0.5), kwargs = {})
#   %mul_795 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_145, 0.7071067811865476), kwargs = {})
#   %erf_94 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_795,), kwargs = {})
#   %add_219 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_94, 1), kwargs = {})
#   %mul_796 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_794, %add_219), kwargs = {})
#   %mul_797 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_796, 1.7015043497085571), kwargs = {})
#   %convolution_146 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_797, %view_315, %arg189_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_146, [2, 3], True), kwargs = {})
#   %convolution_147 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_22, %arg190_1, %arg191_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_147,), kwargs = {})
#   %convolution_148 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg192_1, %arg193_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_21 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_148,), kwargs = {})
#   %mul_801 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_146, %sigmoid_21), kwargs = {})
#   %mul_802 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_801, 2.0), kwargs = {})
#   %mul_803 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_802, %arg194_1), kwargs = {})
#   %mul_804 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_803, 0.2), kwargs = {})
#   %avg_pool2d_5 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_773, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_142 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_5, %view_303, %arg177_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_221 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_804, %convolution_142), kwargs = {})
#   %mul_805 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_221, 0.5), kwargs = {})
#   %mul_806 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_221, 0.7071067811865476), kwargs = {})
#   %erf_95 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_806,), kwargs = {})
#   %add_222 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_95, 1), kwargs = {})
#   %mul_807 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_805, %add_222), kwargs = {})
#   %mul_808 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_807, 1.7015043497085571), kwargs = {})
#   %mul_809 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_808, 0.9805806756909201), kwargs = {})
#   %convolution_149 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_809, %view_318, %arg197_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_49 = async_compile.triton('triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_49', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 64) % 1536
    x4 = (xindex // 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = 0.7071067811865476
    tmp22 = tmp18 * tmp21
    tmp23 = libdevice.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 1.7015043497085571
    tmp28 = tmp26 * tmp27
    tmp29 = 0.9805806756909201
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qr/cqr2jzbuo4vd7wcqqfktuljpkhqovazevbuu3mjidjg3mbgswg4i.py
# Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156, gelu_97, mul__119, out_157, gelu_98, mul__120, out_158, x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, mul_198, out_159, mul__121, mul_200, out_160, gelu_99, mul__122, out_161, out_162], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   gelu_95 => add_222, erf_95, mul_805, mul_806, mul_807
#   gelu_96 => add_224, erf_96, mul_813, mul_814, mul_815
#   gelu_97 => add_226, erf_97, mul_820, mul_821, mul_822
#   gelu_98 => add_228, erf_98, mul_827, mul_828, mul_829
#   gelu_99 => add_231, erf_99, mul_838, mul_839, mul_840
#   mul_198 => mul_834
#   mul_200 => mul_837
#   mul__117 => mul_808
#   mul__118 => mul_816
#   mul__119 => mul_823
#   mul__120 => mul_830
#   mul__121 => mul_836
#   mul__122 => mul_841
#   out_154 => mul_809
#   out_155 => convolution_149
#   out_156 => convolution_150
#   out_157 => convolution_151
#   out_158 => convolution_152
#   out_159 => mul_835
#   out_160 => add_230
#   out_161 => mul_842
#   out_162 => convolution_155
#   sigmoid_22 => sigmoid_22
#   x_se_88 => mean_23
#   x_se_89 => convolution_153
#   x_se_90 => relu_22
#   x_se_91 => convolution_154
# Graph fragment:
#   %mul_805 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_221, 0.5), kwargs = {})
#   %mul_806 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_221, 0.7071067811865476), kwargs = {})
#   %erf_95 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_806,), kwargs = {})
#   %add_222 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_95, 1), kwargs = {})
#   %mul_807 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_805, %add_222), kwargs = {})
#   %mul_808 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_807, 1.7015043497085571), kwargs = {})
#   %mul_809 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_808, 0.9805806756909201), kwargs = {})
#   %convolution_149 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_809, %view_318, %arg197_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_813 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_149, 0.5), kwargs = {})
#   %mul_814 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_149, 0.7071067811865476), kwargs = {})
#   %erf_96 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_814,), kwargs = {})
#   %add_224 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_96, 1), kwargs = {})
#   %mul_815 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_813, %add_224), kwargs = {})
#   %mul_816 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_815, 1.7015043497085571), kwargs = {})
#   %convolution_150 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_816, %view_321, %arg200_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_820 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_150, 0.5), kwargs = {})
#   %mul_821 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_150, 0.7071067811865476), kwargs = {})
#   %erf_97 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_821,), kwargs = {})
#   %add_226 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_97, 1), kwargs = {})
#   %mul_822 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_820, %add_226), kwargs = {})
#   %mul_823 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_822, 1.7015043497085571), kwargs = {})
#   %convolution_151 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_823, %view_324, %arg203_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_827 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_151, 0.5), kwargs = {})
#   %mul_828 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_151, 0.7071067811865476), kwargs = {})
#   %erf_98 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_828,), kwargs = {})
#   %add_228 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_98, 1), kwargs = {})
#   %mul_829 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_827, %add_228), kwargs = {})
#   %mul_830 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_829, 1.7015043497085571), kwargs = {})
#   %convolution_152 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_830, %view_327, %arg206_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_23 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_152, [2, 3], True), kwargs = {})
#   %convolution_153 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_23, %arg207_1, %arg208_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_153,), kwargs = {})
#   %convolution_154 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %arg209_1, %arg210_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_22 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_154,), kwargs = {})
#   %mul_834 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_152, %sigmoid_22), kwargs = {})
#   %mul_835 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_834, 2.0), kwargs = {})
#   %mul_836 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_835, %arg211_1), kwargs = {})
#   %mul_837 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_836, 0.2), kwargs = {})
#   %add_230 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_837, %add_221), kwargs = {})
#   %mul_838 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, 0.5), kwargs = {})
#   %mul_839 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, 0.7071067811865476), kwargs = {})
#   %erf_99 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_839,), kwargs = {})
#   %add_231 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_99, 1), kwargs = {})
#   %mul_840 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_838, %add_231), kwargs = {})
#   %mul_841 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_840, 1.7015043497085571), kwargs = {})
#   %mul_842 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_841, 0.9622504486493761), kwargs = {})
#   %convolution_155 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_842, %view_330, %arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_50 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 64) % 1536
    x4 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.9622504486493761
    tmp28 = tmp26 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ul/culy4tvvj3dangdqtiivdczfkhebfe43fxeljivfqd4lttujtffz.py
# Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, mul_206, out_166, mul__126, mul_208, out_167, x_16], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   gelu_100 => add_233, erf_100, mul_846, mul_847, mul_848
#   gelu_101 => add_235, erf_101, mul_853, mul_854, mul_855
#   gelu_102 => add_237, erf_102, mul_860, mul_861, mul_862
#   gelu_99 => add_231, erf_99, mul_838, mul_839, mul_840
#   mul_206 => mul_867
#   mul_208 => mul_870
#   mul__122 => mul_841
#   mul__123 => mul_849
#   mul__124 => mul_856
#   mul__125 => mul_863
#   mul__126 => mul_869
#   out_161 => mul_842
#   out_162 => convolution_155
#   out_163 => convolution_156
#   out_164 => convolution_157
#   out_165 => convolution_158
#   out_166 => mul_868
#   out_167 => add_239
#   sigmoid_23 => sigmoid_23
#   x_16 => convolution_161
#   x_se_92 => mean_24
#   x_se_93 => convolution_159
#   x_se_94 => relu_23
#   x_se_95 => convolution_160
# Graph fragment:
#   %mul_838 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, 0.5), kwargs = {})
#   %mul_839 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, 0.7071067811865476), kwargs = {})
#   %erf_99 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_839,), kwargs = {})
#   %add_231 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_99, 1), kwargs = {})
#   %mul_840 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_838, %add_231), kwargs = {})
#   %mul_841 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_840, 1.7015043497085571), kwargs = {})
#   %mul_842 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_841, 0.9622504486493761), kwargs = {})
#   %convolution_155 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_842, %view_330, %arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_846 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_155, 0.5), kwargs = {})
#   %mul_847 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_155, 0.7071067811865476), kwargs = {})
#   %erf_100 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_847,), kwargs = {})
#   %add_233 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_100, 1), kwargs = {})
#   %mul_848 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_846, %add_233), kwargs = {})
#   %mul_849 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_848, 1.7015043497085571), kwargs = {})
#   %convolution_156 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_849, %view_333, %arg217_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_853 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_156, 0.5), kwargs = {})
#   %mul_854 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_156, 0.7071067811865476), kwargs = {})
#   %erf_101 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_854,), kwargs = {})
#   %add_235 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_101, 1), kwargs = {})
#   %mul_855 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_853, %add_235), kwargs = {})
#   %mul_856 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_855, 1.7015043497085571), kwargs = {})
#   %convolution_157 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_856, %view_336, %arg220_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_860 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_157, 0.5), kwargs = {})
#   %mul_861 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_157, 0.7071067811865476), kwargs = {})
#   %erf_102 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_861,), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_102, 1), kwargs = {})
#   %mul_862 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_860, %add_237), kwargs = {})
#   %mul_863 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_862, 1.7015043497085571), kwargs = {})
#   %convolution_158 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_863, %view_339, %arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_158, [2, 3], True), kwargs = {})
#   %convolution_159 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_24, %arg224_1, %arg225_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_159,), kwargs = {})
#   %convolution_160 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg226_1, %arg227_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_23 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_160,), kwargs = {})
#   %mul_867 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_158, %sigmoid_23), kwargs = {})
#   %mul_868 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_867, 2.0), kwargs = {})
#   %mul_869 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_868, %arg228_1), kwargs = {})
#   %mul_870 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_869, 0.2), kwargs = {})
#   %add_239 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_870, %add_230), kwargs = {})
#   %convolution_161 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_239, %view_342, %arg231_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_51 = async_compile.triton('triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 64) % 1536
    x4 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/as/casp4qmnvz637cn3wyk2gfoh27d3xhe37wwvscd2loijm73fq7s2.py
# Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, mul_206, out_166, mul__126, mul_208, out_167, x_16, gelu_103, x_17, x_18], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   gelu_100 => add_233, erf_100, mul_846, mul_847, mul_848
#   gelu_101 => add_235, erf_101, mul_853, mul_854, mul_855
#   gelu_102 => add_237, erf_102, mul_860, mul_861, mul_862
#   gelu_103 => add_241, erf_103, mul_874, mul_875, mul_876
#   gelu_99 => add_231, erf_99, mul_838, mul_839, mul_840
#   mul_206 => mul_867
#   mul_208 => mul_870
#   mul__122 => mul_841
#   mul__123 => mul_849
#   mul__124 => mul_856
#   mul__125 => mul_863
#   mul__126 => mul_869
#   out_161 => mul_842
#   out_162 => convolution_155
#   out_163 => convolution_156
#   out_164 => convolution_157
#   out_165 => convolution_158
#   out_166 => mul_868
#   out_167 => add_239
#   sigmoid_23 => sigmoid_23
#   x_16 => convolution_161
#   x_17 => mul_877
#   x_18 => mean_25
#   x_se_92 => mean_24
#   x_se_93 => convolution_159
#   x_se_94 => relu_23
#   x_se_95 => convolution_160
# Graph fragment:
#   %mul_838 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, 0.5), kwargs = {})
#   %mul_839 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, 0.7071067811865476), kwargs = {})
#   %erf_99 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_839,), kwargs = {})
#   %add_231 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_99, 1), kwargs = {})
#   %mul_840 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_838, %add_231), kwargs = {})
#   %mul_841 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_840, 1.7015043497085571), kwargs = {})
#   %mul_842 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_841, 0.9622504486493761), kwargs = {})
#   %convolution_155 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_842, %view_330, %arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_846 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_155, 0.5), kwargs = {})
#   %mul_847 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_155, 0.7071067811865476), kwargs = {})
#   %erf_100 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_847,), kwargs = {})
#   %add_233 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_100, 1), kwargs = {})
#   %mul_848 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_846, %add_233), kwargs = {})
#   %mul_849 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_848, 1.7015043497085571), kwargs = {})
#   %convolution_156 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_849, %view_333, %arg217_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_853 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_156, 0.5), kwargs = {})
#   %mul_854 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_156, 0.7071067811865476), kwargs = {})
#   %erf_101 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_854,), kwargs = {})
#   %add_235 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_101, 1), kwargs = {})
#   %mul_855 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_853, %add_235), kwargs = {})
#   %mul_856 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_855, 1.7015043497085571), kwargs = {})
#   %convolution_157 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_856, %view_336, %arg220_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %mul_860 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_157, 0.5), kwargs = {})
#   %mul_861 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_157, 0.7071067811865476), kwargs = {})
#   %erf_102 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_861,), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_102, 1), kwargs = {})
#   %mul_862 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_860, %add_237), kwargs = {})
#   %mul_863 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_862, 1.7015043497085571), kwargs = {})
#   %convolution_158 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_863, %view_339, %arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_158, [2, 3], True), kwargs = {})
#   %convolution_159 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_24, %arg224_1, %arg225_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_159,), kwargs = {})
#   %convolution_160 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg226_1, %arg227_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_23 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_160,), kwargs = {})
#   %mul_867 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_158, %sigmoid_23), kwargs = {})
#   %mul_868 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_867, 2.0), kwargs = {})
#   %mul_869 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_868, %arg228_1), kwargs = {})
#   %mul_870 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_869, 0.2), kwargs = {})
#   %add_239 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_870, %add_230), kwargs = {})
#   %convolution_161 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_239, %view_342, %arg231_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_874 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_161, 0.5), kwargs = {})
#   %mul_875 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_161, 0.7071067811865476), kwargs = {})
#   %erf_103 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_875,), kwargs = {})
#   %add_241 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_103, 1), kwargs = {})
#   %mul_876 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_874, %add_241), kwargs = {})
#   %mul_877 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_876, 1.7015043497085571), kwargs = {})
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_877, [-1, -2], True), kwargs = {})
triton_per_fused_add_convolution_gelu_mean_mul_relu_sigmoid_52 = async_compile.triton('triton_per_fused_add_convolution_gelu_mean_mul_relu_sigmoid_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_gelu_mean_mul_relu_sigmoid_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_gelu_mean_mul_relu_sigmoid_52(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp16 = 64.0
    tmp17 = tmp15 / tmp16
    tl.store(out_ptr1 + (x3), tmp17, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(arg1_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg2_1, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg5_1, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg6_1, (32, ), (1, ))
    assert_size_stride(arg7_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg8_1, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg11_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg14_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg17_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg20_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg23_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg26_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (), ())
    assert_size_stride(arg33_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg34_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg37_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg40_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg43_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg46_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (), ())
    assert_size_stride(arg53_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg54_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg57_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg60_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg63_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg67_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (), ())
    assert_size_stride(arg70_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg71_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg72_1, (1536, ), (1, ))
    assert_size_stride(arg73_1, (768, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg74_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg77_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg80_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg83_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg84_1, (1536, ), (1, ))
    assert_size_stride(arg85_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg88_1, (1536, ), (1, ))
    assert_size_stride(arg89_1, (), ())
    assert_size_stride(arg90_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg91_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg94_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg97_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg100_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg101_1, (1536, ), (1, ))
    assert_size_stride(arg102_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg105_1, (1536, ), (1, ))
    assert_size_stride(arg106_1, (), ())
    assert_size_stride(arg107_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg108_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg111_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg114_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg117_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg118_1, (1536, ), (1, ))
    assert_size_stride(arg119_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg122_1, (1536, ), (1, ))
    assert_size_stride(arg123_1, (), ())
    assert_size_stride(arg124_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg125_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg128_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg131_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg134_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg135_1, (1536, ), (1, ))
    assert_size_stride(arg136_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg139_1, (1536, ), (1, ))
    assert_size_stride(arg140_1, (), ())
    assert_size_stride(arg141_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg142_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg145_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg148_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg151_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg152_1, (1536, ), (1, ))
    assert_size_stride(arg153_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg156_1, (1536, ), (1, ))
    assert_size_stride(arg157_1, (), ())
    assert_size_stride(arg158_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg159_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg162_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg165_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg168_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg169_1, (1536, ), (1, ))
    assert_size_stride(arg170_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg173_1, (1536, ), (1, ))
    assert_size_stride(arg174_1, (), ())
    assert_size_stride(arg175_1, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg176_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg177_1, (1536, ), (1, ))
    assert_size_stride(arg178_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg179_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg182_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg185_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg188_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg189_1, (1536, ), (1, ))
    assert_size_stride(arg190_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg193_1, (1536, ), (1, ))
    assert_size_stride(arg194_1, (), ())
    assert_size_stride(arg195_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg196_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg199_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg202_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg203_1, (768, ), (1, ))
    assert_size_stride(arg204_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg205_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg206_1, (1536, ), (1, ))
    assert_size_stride(arg207_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg208_1, (768, ), (1, ))
    assert_size_stride(arg209_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg210_1, (1536, ), (1, ))
    assert_size_stride(arg211_1, (), ())
    assert_size_stride(arg212_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg213_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg214_1, (768, ), (1, ))
    assert_size_stride(arg215_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg216_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg219_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg220_1, (768, ), (1, ))
    assert_size_stride(arg221_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg222_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg223_1, (1536, ), (1, ))
    assert_size_stride(arg224_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg225_1, (768, ), (1, ))
    assert_size_stride(arg226_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg227_1, (1536, ), (1, ))
    assert_size_stride(arg228_1, (), ())
    assert_size_stride(arg229_1, (3072, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg230_1, (3072, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg231_1, (3072, ), (1, ))
    assert_size_stride(arg232_1, (1000, 3072), (3072, 1))
    assert_size_stride(arg233_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf172 = empty_strided_cuda((16, 3, 3, 3), (27, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_57, x_11, input_8], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_0.run(arg1_1, arg2_1, buf172, 16, 27, grid=grid(16), stream=stream0)
        del arg1_1
        del arg2_1
        buf175 = empty_strided_cuda((32, 16, 3, 3), (144, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_58, x_11, input_8, gelu_52, input_9, input_10], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_1.run(arg4_1, arg5_1, buf175, 32, 144, grid=grid(32), stream=stream0)
        del arg4_1
        del arg5_1
        buf178 = empty_strided_cuda((64, 32, 3, 3), (288, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_59, x_11, input_8, gelu_52, input_9, input_10, gelu_53, input_11, input_12], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_2.run(arg7_1, arg8_1, buf178, 64, 288, grid=grid(64), stream=stream0)
        del arg7_1
        del arg8_1
        buf181 = empty_strided_cuda((128, 64, 3, 3), (576, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_60, x_11, input_8, gelu_52, input_9, input_10, gelu_53, input_11, input_12, gelu_54, input_13, x_12, input_14], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_3.run(arg10_1, arg11_1, buf181, 128, 576, grid=grid(128), stream=stream0)
        del arg10_1
        del arg11_1
        buf200 = empty_strided_cuda((256, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_61, shortcut_4], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_4.run(arg13_1, arg14_1, buf200, 256, 128, grid=grid(256), stream=stream0)
        del arg13_1
        del arg14_1
        buf184 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_62, out_85], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_5.run(arg16_1, arg17_1, buf184, 128, 128, grid=grid(128), stream=stream0)
        del arg16_1
        del arg17_1
        buf187 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_63, out_85, gelu_56, mul__68, out_86], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_6.run(arg19_1, arg20_1, buf187, 128, 1152, grid=grid(128), stream=stream0)
        del arg19_1
        del arg20_1
        buf190 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_64, out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_6.run(arg22_1, arg23_1, buf190, 128, 1152, grid=grid(128), stream=stream0)
        del arg22_1
        del arg23_1
        buf193 = empty_strided_cuda((256, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_65, out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87, gelu_58, mul__70, out_88], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_convolution_4.run(arg25_1, arg26_1, buf193, 256, 128, grid=grid(256), stream=stream0)
        del arg25_1
        del arg26_1
        buf221 = empty_strided_cuda((512, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_66, avg_pool2d_3, shortcut_5], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
        triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_7.run(arg33_1, arg34_1, buf221, 512, 256, grid=grid(512), stream=stream0)
        del arg33_1
        del arg34_1
        buf204 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_67, out_92], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_8.run(arg36_1, arg37_1, buf204, 256, 256, grid=grid(256), stream=stream0)
        del arg36_1
        del arg37_1
        buf207 = empty_strided_cuda((256, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_68, out_92, gelu_60, mul__73, x_13, out_93], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9.run(arg39_1, arg40_1, buf207, 256, 1152, grid=grid(256), stream=stream0)
        del arg39_1
        del arg40_1
        buf210 = empty_strided_cuda((256, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_69, out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9.run(arg42_1, arg43_1, buf210, 256, 1152, grid=grid(256), stream=stream0)
        del arg42_1
        del arg43_1
        buf213 = empty_strided_cuda((512, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_70, out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94, gelu_62, mul__75, out_95], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_7.run(arg45_1, arg46_1, buf213, 512, 256, grid=grid(512), stream=stream0)
        del arg45_1
        del arg46_1
        buf225 = empty_strided_cuda((256, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_71, gelu_63, mul__77, out_98, out_99], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_gelu_mul_10.run(arg53_1, arg54_1, buf225, 256, 512, grid=grid(256), stream=stream0)
        del arg53_1
        del arg54_1
        buf228 = empty_strided_cuda((256, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_72, gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9.run(arg56_1, arg57_1, buf228, 256, 1152, grid=grid(256), stream=stream0)
        del arg56_1
        del arg57_1
        buf231 = empty_strided_cuda((256, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_73, gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100, gelu_65, mul__79, out_101], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9.run(arg59_1, arg60_1, buf231, 256, 1152, grid=grid(256), stream=stream0)
        del arg59_1
        del arg60_1
        buf234 = empty_strided_cuda((512, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_74, gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100, gelu_65, mul__79, out_101, gelu_66, mul__80, out_102], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_7.run(arg62_1, arg63_1, buf234, 512, 256, grid=grid(512), stream=stream0)
        del arg62_1
        del arg63_1
        buf260 = empty_strided_cuda((1536, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_75, avg_pool2d_4, shortcut_6], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
        triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_11.run(arg70_1, arg71_1, buf260, 1536, 512, grid=grid(1536), stream=stream0)
        del arg70_1
        del arg71_1
        buf243 = empty_strided_cuda((768, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_76, out_106], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_12.run(arg73_1, arg74_1, buf243, 768, 512, grid=grid(768), stream=stream0)
        del arg73_1
        del arg74_1
        buf246 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_77, out_106, gelu_68, mul__83, x_14, out_107], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg76_1, arg77_1, buf246, 768, 1152, grid=grid(768), stream=stream0)
        del arg76_1
        del arg77_1
        buf249 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_78, out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg79_1, arg80_1, buf249, 768, 1152, grid=grid(768), stream=stream0)
        del arg79_1
        del arg80_1
        buf252 = empty_strided_cuda((1536, 768, 1, 1), (768, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_79, out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108, gelu_70, mul__85, out_109], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg82_1, arg83_1, buf252, 1536, 768, grid=grid(1536), stream=stream0)
        del arg82_1
        del arg83_1
        buf264 = empty_strided_cuda((768, 1536, 1, 1), (1536, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_80, gelu_71, mul__87, out_112, out_113], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg90_1, arg91_1, buf264, 768, 1536, grid=grid(768), stream=stream0)
        del arg90_1
        del arg91_1
        buf267 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_81, gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg93_1, arg94_1, buf267, 768, 1152, grid=grid(768), stream=stream0)
        del arg93_1
        del arg94_1
        buf270 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_82, gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114, gelu_73, mul__89, out_115], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg96_1, arg97_1, buf270, 768, 1152, grid=grid(768), stream=stream0)
        del arg96_1
        del arg97_1
        buf273 = empty_strided_cuda((1536, 768, 1, 1), (768, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_83, gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114, gelu_73, mul__89, out_115, gelu_74, mul__90, out_116], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg99_1, arg100_1, buf273, 1536, 768, grid=grid(1536), stream=stream0)
        del arg100_1
        del arg99_1
        buf282 = empty_strided_cuda((768, 1536, 1, 1), (1536, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_84, gelu_75, mul__92, out_119, out_120], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg107_1, arg108_1, buf282, 768, 1536, grid=grid(768), stream=stream0)
        del arg107_1
        del arg108_1
        buf285 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_85, gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg110_1, arg111_1, buf285, 768, 1152, grid=grid(768), stream=stream0)
        del arg110_1
        del arg111_1
        buf288 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_86, gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121, gelu_77, mul__94, out_122], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg113_1, arg114_1, buf288, 768, 1152, grid=grid(768), stream=stream0)
        del arg113_1
        del arg114_1
        buf291 = empty_strided_cuda((1536, 768, 1, 1), (768, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_87, gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121, gelu_77, mul__94, out_122, gelu_78, mul__95, out_123], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg116_1, arg117_1, buf291, 1536, 768, grid=grid(1536), stream=stream0)
        del arg116_1
        del arg117_1
        buf300 = empty_strided_cuda((768, 1536, 1, 1), (1536, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_88, gelu_79, mul__97, out_126, out_127], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg124_1, arg125_1, buf300, 768, 1536, grid=grid(768), stream=stream0)
        del arg124_1
        del arg125_1
        buf303 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_89, gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg127_1, arg128_1, buf303, 768, 1152, grid=grid(768), stream=stream0)
        del arg127_1
        del arg128_1
        buf306 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_90, gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128, gelu_81, mul__99, out_129], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg130_1, arg131_1, buf306, 768, 1152, grid=grid(768), stream=stream0)
        del arg130_1
        del arg131_1
        buf309 = empty_strided_cuda((1536, 768, 1, 1), (768, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_91, gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128, gelu_81, mul__99, out_129, gelu_82, mul__100, out_130], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg133_1, arg134_1, buf309, 1536, 768, grid=grid(1536), stream=stream0)
        del arg133_1
        del arg134_1
        buf318 = empty_strided_cuda((768, 1536, 1, 1), (1536, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_92, gelu_83, mul__102, out_133, out_134], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg141_1, arg142_1, buf318, 768, 1536, grid=grid(768), stream=stream0)
        del arg141_1
        del arg142_1
        buf321 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_93, gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg144_1, arg145_1, buf321, 768, 1152, grid=grid(768), stream=stream0)
        del arg144_1
        del arg145_1
        buf324 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_94, gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135, gelu_85, mul__104, out_136], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg147_1, arg148_1, buf324, 768, 1152, grid=grid(768), stream=stream0)
        del arg147_1
        del arg148_1
        buf327 = empty_strided_cuda((1536, 768, 1, 1), (768, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_95, gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135, gelu_85, mul__104, out_136, gelu_86, mul__105, out_137], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg150_1, arg151_1, buf327, 1536, 768, grid=grid(1536), stream=stream0)
        del arg150_1
        del arg151_1
        buf336 = empty_strided_cuda((768, 1536, 1, 1), (1536, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_96, gelu_87, mul__107, out_140, out_141], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg158_1, arg159_1, buf336, 768, 1536, grid=grid(768), stream=stream0)
        del arg158_1
        del arg159_1
        buf339 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_97, gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg161_1, arg162_1, buf339, 768, 1152, grid=grid(768), stream=stream0)
        del arg161_1
        del arg162_1
        buf342 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_98, gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142, gelu_89, mul__109, out_143], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg164_1, arg165_1, buf342, 768, 1152, grid=grid(768), stream=stream0)
        del arg164_1
        del arg165_1
        buf345 = empty_strided_cuda((1536, 768, 1, 1), (768, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_99, gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142, gelu_89, mul__109, out_143, gelu_90, mul__110, out_144], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg167_1, arg168_1, buf345, 1536, 768, grid=grid(1536), stream=stream0)
        del arg167_1
        del arg168_1
        buf371 = empty_strided_cuda((1536, 1536, 1, 1), (1536, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_100, avg_pool2d_5, shortcut_7], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
        triton_red_fused__native_batch_norm_legit_avg_pool2d_convolution_16.run(arg175_1, arg176_1, buf371, 1536, 1536, grid=grid(1536), stream=stream0)
        del arg175_1
        del arg176_1
        buf354 = empty_strided_cuda((768, 1536, 1, 1), (1536, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_101, out_148], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg178_1, arg179_1, buf354, 768, 1536, grid=grid(768), stream=stream0)
        del arg178_1
        del arg179_1
        buf357 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_102, out_148, gelu_92, mul__113, x_15, out_149], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg181_1, arg182_1, buf357, 768, 1152, grid=grid(768), stream=stream0)
        del arg181_1
        del arg182_1
        buf360 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_103, out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg184_1, arg185_1, buf360, 768, 1152, grid=grid(768), stream=stream0)
        del arg184_1
        del arg185_1
        buf363 = empty_strided_cuda((1536, 768, 1, 1), (768, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_104, out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150, gelu_94, mul__115, out_151], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg187_1, arg188_1, buf363, 1536, 768, grid=grid(1536), stream=stream0)
        del arg187_1
        del arg188_1
        buf375 = empty_strided_cuda((768, 1536, 1, 1), (1536, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_105, gelu_95, mul__117, out_154, out_155], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg195_1, arg196_1, buf375, 768, 1536, grid=grid(768), stream=stream0)
        del arg195_1
        del arg196_1
        buf378 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_106, gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg198_1, arg199_1, buf378, 768, 1152, grid=grid(768), stream=stream0)
        del arg198_1
        del arg199_1
        buf381 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_107, gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156, gelu_97, mul__119, out_157], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg201_1, arg202_1, buf381, 768, 1152, grid=grid(768), stream=stream0)
        del arg201_1
        del arg202_1
        buf384 = empty_strided_cuda((1536, 768, 1, 1), (768, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_108, gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156, gelu_97, mul__119, out_157, gelu_98, mul__120, out_158], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg204_1, arg205_1, buf384, 1536, 768, grid=grid(1536), stream=stream0)
        del arg204_1
        del arg205_1
        buf393 = empty_strided_cuda((768, 1536, 1, 1), (1536, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_109, gelu_99, mul__122, out_161, out_162], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg212_1, arg213_1, buf393, 768, 1536, grid=grid(768), stream=stream0)
        del arg212_1
        del arg213_1
        buf396 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_110, gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg215_1, arg216_1, buf396, 768, 1152, grid=grid(768), stream=stream0)
        del arg215_1
        del arg216_1
        buf399 = empty_strided_cuda((768, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_111, gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg218_1, arg219_1, buf399, 768, 1152, grid=grid(768), stream=stream0)
        del arg218_1
        del arg219_1
        buf402 = empty_strided_cuda((1536, 768, 1, 1), (768, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_112, gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg221_1, arg222_1, buf402, 1536, 768, grid=grid(1536), stream=stream0)
        del arg221_1
        del arg222_1
        buf410 = empty_strided_cuda((3072, 1536, 1, 1), (1536, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_113, gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, mul_206, out_166, mul__126, mul_208, out_167, x_16], Original ATen: [aten._native_batch_norm_legit, aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_red_fused__native_batch_norm_legit_add_convolution_gelu_mean_mul_relu_sigmoid_17.run(arg229_1, arg230_1, buf410, 3072, 1536, grid=grid(3072), stream=stream0)
        del arg229_1
        del arg230_1
        buf171 = empty_strided_cuda((8, 3, 257, 257), (198147, 66049, 257, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11, input_8], Original ATen: [aten.constant_pad_nd, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_18.run(arg0_1, buf171, 1585176, grid=grid(1585176), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_11, input_8], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf173 = extern_kernels.convolution(buf171, buf172, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 16, 128, 128), (262144, 16384, 128, 1))
        del buf171
        del buf172
        buf174 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [x_11, input_8, gelu_52, input_9, input_10], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_19.run(buf174, arg3_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg3_1
        # Topologically Sorted Source Nodes: [x_11, input_8, gelu_52, input_9, input_10], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf176 = extern_kernels.convolution(buf174, buf175, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 32, 128, 128), (524288, 16384, 128, 1))
        del buf174
        del buf175
        buf177 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [x_11, input_8, gelu_52, input_9, input_10, gelu_53, input_11, input_12], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_20.run(buf177, arg6_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [x_11, input_8, gelu_52, input_9, input_10, gelu_53, input_11, input_12], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf179 = extern_kernels.convolution(buf177, buf178, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del buf177
        del buf178
        buf180 = empty_strided_cuda((8, 64, 129, 129), (1065024, 16641, 129, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11, input_8, gelu_52, input_9, input_10, gelu_53, input_11, input_12, gelu_54, input_13, x_12, input_14], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_21.run(buf179, arg9_1, buf180, 8520192, grid=grid(8520192), stream=stream0)
        del arg9_1
        # Topologically Sorted Source Nodes: [x_11, input_8, gelu_52, input_9, input_10, gelu_53, input_11, input_12, gelu_54, input_13, x_12, input_14], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf182 = extern_kernels.convolution(buf180, buf181, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del buf180
        del buf181
        buf183 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [x_11, input_8, gelu_52, input_9, input_10, gelu_53, input_11, input_12, gelu_54, input_13, x_12, input_14, gelu_55, mul__67, out_84], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_22.run(buf183, arg12_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg12_1
        # Topologically Sorted Source Nodes: [out_85], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf183, buf184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del buf184
        buf186 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_23.run(buf186, arg18_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg18_1
        # Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf188 = extern_kernels.convolution(buf186, buf187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del buf186
        del buf187
        buf189 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_23.run(buf189, arg21_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg21_1
        # Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf191 = extern_kernels.convolution(buf189, buf190, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del buf189
        del buf190
        buf192 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87, gelu_58, mul__70, out_88], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_23.run(buf192, arg24_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg24_1
        # Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87, gelu_58, mul__70, out_88], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf194 = extern_kernels.convolution(buf192, buf193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del buf192
        del buf193
        buf196 = empty_strided_cuda((8, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87, gelu_58, mul__70, out_88, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.mean]
        triton_red_fused_convolution_gelu_mean_mul_24.run(buf194, arg27_1, buf196, 2048, 4096, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87, gelu_58, mul__70, out_88, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.mean]
        buf197 = extern_kernels.convolution(buf196, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg28_1
        del buf196
        buf198 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87, gelu_58, mul__70, out_88, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.mean, aten.relu]
        triton_poi_fused_convolution_gelu_mean_mul_relu_25.run(buf198, arg29_1, 1024, grid=grid(1024), stream=stream0)
        del arg29_1
        # Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87, gelu_58, mul__70, out_88, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.mean, aten.relu]
        buf199 = extern_kernels.convolution(buf198, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg30_1
        del buf198
        # Topologically Sorted Source Nodes: [shortcut_4], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf183, buf200, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del buf200
        buf202 = buf194; del buf194  # reuse
        buf203 = reinterpret_tensor(buf179, (8, 256, 64, 64), (1048576, 4096, 64, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [out_85, gelu_56, mul__68, out_86, gelu_57, mul__69, out_87, gelu_58, mul__70, out_88, x_se_48, x_se_49, x_se_50, x_se_51, sigmoid_12, mul_115, out_89, mul__71, mul_117, shortcut_4, out_90, gelu_59, mul__72, out_91], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_26.run(buf202, arg27_1, buf199, arg31_1, arg32_1, buf201, arg15_1, buf203, 8388608, grid=grid(8388608), stream=stream0)
        del arg15_1
        del arg27_1
        del arg31_1
        del arg32_1
        del buf199
        del buf201
        del buf202
        # Topologically Sorted Source Nodes: [out_92], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf203, buf204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del buf204
        buf206 = empty_strided_cuda((8, 256, 65, 65), (1081600, 4225, 65, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_27.run(buf205, arg38_1, buf206, 8652800, grid=grid(8652800), stream=stream0)
        del arg38_1
        del buf205
        # Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        buf208 = extern_kernels.convolution(buf206, buf207, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf208, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del buf206
        del buf207
        buf209 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28.run(buf209, arg41_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg41_1
        # Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        buf211 = extern_kernels.convolution(buf209, buf210, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf211, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del buf209
        del buf210
        buf212 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94, gelu_62, mul__75, out_95], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28.run(buf212, arg44_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg44_1
        # Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94, gelu_62, mul__75, out_95], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        buf214 = extern_kernels.convolution(buf212, buf213, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del buf213
        buf216 = empty_strided_cuda((8, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94, gelu_62, mul__75, out_95, x_se_52, x_se_53], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_29.run(buf214, arg47_1, buf216, 4096, 1024, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94, gelu_62, mul__75, out_95, x_se_52, x_se_53], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean]
        buf217 = extern_kernels.convolution(buf216, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg48_1
        del buf216
        buf218 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94, gelu_62, mul__75, out_95, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_30.run(buf218, arg49_1, 2048, grid=grid(2048), stream=stream0)
        del arg49_1
        # Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94, gelu_62, mul__75, out_95, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu]
        buf219 = extern_kernels.convolution(buf218, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg50_1
        del buf218
        buf220 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [avg_pool2d_3, shortcut_5], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_31.run(buf203, buf220, 2097152, grid=grid(2097152), stream=stream0)
        del buf203
        # Topologically Sorted Source Nodes: [avg_pool2d_3, shortcut_5], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf222 = extern_kernels.convolution(buf220, buf221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del buf220
        del buf221
        buf223 = buf214; del buf214  # reuse
        buf224 = reinterpret_tensor(buf183, (8, 512, 32, 32), (524288, 1024, 32, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [out_92, gelu_60, mul__73, x_13, out_93, gelu_61, mul__74, out_94, gelu_62, mul__75, out_95, x_se_52, x_se_53, x_se_54, x_se_55, sigmoid_13, mul_124, out_96, mul__76, mul_126, avg_pool2d_3, shortcut_5, out_97, gelu_63, mul__77, out_98, out_99], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu, aten.sigmoid, aten.avg_pool2d, aten.add]
        triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_32.run(buf223, arg47_1, buf219, arg51_1, arg52_1, buf222, arg35_1, buf224, 4194304, grid=grid(4194304), stream=stream0)
        del arg35_1
        del arg47_1
        del arg51_1
        del arg52_1
        del buf222
        # Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf226 = extern_kernels.convolution(buf224, buf225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del buf225
        buf227 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28.run(buf227, arg55_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg55_1
        # Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf229 = extern_kernels.convolution(buf227, buf228, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf229, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del buf227
        del buf228
        buf230 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100, gelu_65, mul__79, out_101], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28.run(buf230, arg58_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg58_1
        # Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100, gelu_65, mul__79, out_101], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf232 = extern_kernels.convolution(buf230, buf231, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf232, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del buf230
        del buf231
        buf233 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100, gelu_65, mul__79, out_101, gelu_66, mul__80, out_102], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28.run(buf233, arg61_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg61_1
        # Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100, gelu_65, mul__79, out_101, gelu_66, mul__80, out_102], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf235 = extern_kernels.convolution(buf233, buf234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del buf233
        del buf234
        buf237 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100, gelu_65, mul__79, out_101, gelu_66, mul__80, out_102, x_se_56, x_se_57], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_29.run(buf235, arg64_1, buf237, 4096, 1024, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100, gelu_65, mul__79, out_101, gelu_66, mul__80, out_102, x_se_56, x_se_57], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        buf238 = extern_kernels.convolution(buf237, arg65_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg65_1
        del buf237
        buf239 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100, gelu_65, mul__79, out_101, gelu_66, mul__80, out_102, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_30.run(buf239, arg66_1, 2048, grid=grid(2048), stream=stream0)
        del arg66_1
        # Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100, gelu_65, mul__79, out_101, gelu_66, mul__80, out_102, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf240 = extern_kernels.convolution(buf239, arg67_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg67_1
        del buf239
        buf241 = buf223; del buf223  # reuse
        buf242 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [gelu_63, mul__77, out_98, out_99, gelu_64, mul__78, out_100, gelu_65, mul__79, out_101, gelu_66, mul__80, out_102, x_se_56, x_se_57, x_se_58, x_se_59, sigmoid_14, mul_132, out_103, mul__81, mul_134, out_104, gelu_67, mul__82, out_105], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_33.run(buf241, buf235, arg64_1, buf240, arg68_1, arg69_1, buf242, 4194304, grid=grid(4194304), stream=stream0)
        del arg64_1
        del arg68_1
        del arg69_1
        del buf235
        del buf240
        del buf241
        # Topologically Sorted Source Nodes: [out_106], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf242, buf243, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (8, 768, 32, 32), (786432, 1024, 32, 1))
        del buf243
        buf245 = empty_strided_cuda((8, 768, 33, 33), (836352, 1089, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_34.run(buf244, arg75_1, buf245, 6690816, grid=grid(6690816), stream=stream0)
        del arg75_1
        del buf244
        # Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        buf247 = extern_kernels.convolution(buf245, buf246, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf247, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf245
        del buf246
        buf248 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf248, arg78_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg78_1
        # Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        buf250 = extern_kernels.convolution(buf248, buf249, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf250, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf248
        del buf249
        buf251 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108, gelu_70, mul__85, out_109], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf251, arg81_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg81_1
        # Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108, gelu_70, mul__85, out_109], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        buf253 = extern_kernels.convolution(buf251, buf252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf251
        del buf252
        buf255 = empty_strided_cuda((8, 1536, 1, 1), (1536, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108, gelu_70, mul__85, out_109, x_se_60, x_se_61], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36.run(buf253, arg84_1, buf255, 12288, 256, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108, gelu_70, mul__85, out_109, x_se_60, x_se_61], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean]
        buf256 = extern_kernels.convolution(buf255, arg85_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg85_1
        del buf255
        buf257 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108, gelu_70, mul__85, out_109, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf257, arg86_1, 6144, grid=grid(6144), stream=stream0)
        del arg86_1
        # Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108, gelu_70, mul__85, out_109, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu]
        buf258 = extern_kernels.convolution(buf257, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg87_1
        del buf257
        buf259 = empty_strided_cuda((8, 512, 16, 16), (131072, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [avg_pool2d_4, shortcut_6], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_38.run(buf242, buf259, 1048576, grid=grid(1048576), stream=stream0)
        del buf242
        # Topologically Sorted Source Nodes: [avg_pool2d_4, shortcut_6], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf261 = extern_kernels.convolution(buf259, buf260, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf259
        buf262 = buf253; del buf253  # reuse
        buf263 = empty_strided_cuda((8, 1536, 16, 16), (393216, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_106, gelu_68, mul__83, x_14, out_107, gelu_69, mul__84, out_108, gelu_70, mul__85, out_109, x_se_60, x_se_61, x_se_62, x_se_63, sigmoid_15, mul_141, out_110, mul__86, mul_143, avg_pool2d_4, shortcut_6, out_111, gelu_71, mul__87, out_112, out_113], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu, aten.sigmoid, aten.avg_pool2d, aten.add]
        triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_39.run(buf262, arg84_1, buf258, arg88_1, arg89_1, buf261, arg72_1, buf263, 3145728, grid=grid(3145728), stream=stream0)
        del arg72_1
        del arg84_1
        del arg88_1
        del arg89_1
        del buf261
        # Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf265 = extern_kernels.convolution(buf263, buf264, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf264
        buf266 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf266, arg92_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg92_1
        # Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf268 = extern_kernels.convolution(buf266, buf267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf268, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf266
        del buf267
        buf269 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114, gelu_73, mul__89, out_115], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf269, arg95_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg95_1
        # Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114, gelu_73, mul__89, out_115], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf271 = extern_kernels.convolution(buf269, buf270, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf271, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf269
        del buf270
        buf272 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114, gelu_73, mul__89, out_115, gelu_74, mul__90, out_116], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf272, arg98_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg98_1
        # Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114, gelu_73, mul__89, out_115, gelu_74, mul__90, out_116], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf274 = extern_kernels.convolution(buf272, buf273, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf272
        del buf273
        buf276 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114, gelu_73, mul__89, out_115, gelu_74, mul__90, out_116, x_se_64, x_se_65], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36.run(buf274, arg101_1, buf276, 12288, 256, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114, gelu_73, mul__89, out_115, gelu_74, mul__90, out_116, x_se_64, x_se_65], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        buf277 = extern_kernels.convolution(buf276, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg102_1
        del buf276
        buf278 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114, gelu_73, mul__89, out_115, gelu_74, mul__90, out_116, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf278, arg103_1, 6144, grid=grid(6144), stream=stream0)
        del arg103_1
        # Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114, gelu_73, mul__89, out_115, gelu_74, mul__90, out_116, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf279 = extern_kernels.convolution(buf278, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg104_1
        del buf278
        buf280 = buf262; del buf262  # reuse
        buf281 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [gelu_71, mul__87, out_112, out_113, gelu_72, mul__88, out_114, gelu_73, mul__89, out_115, gelu_74, mul__90, out_116, x_se_64, x_se_65, x_se_66, x_se_67, sigmoid_16, mul_149, out_117, mul__91, mul_151, out_118, gelu_75, mul__92, out_119, out_120], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_40.run(buf280, buf274, arg101_1, buf279, arg105_1, arg106_1, buf281, 3145728, grid=grid(3145728), stream=stream0)
        del arg101_1
        del arg105_1
        del arg106_1
        del buf274
        # Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf283 = extern_kernels.convolution(buf281, buf282, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf282
        buf284 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf284, arg109_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg109_1
        # Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf286 = extern_kernels.convolution(buf284, buf285, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf286, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf284
        del buf285
        buf287 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121, gelu_77, mul__94, out_122], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf287, arg112_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg112_1
        # Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121, gelu_77, mul__94, out_122], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf289 = extern_kernels.convolution(buf287, buf288, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf289, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf287
        del buf288
        buf290 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121, gelu_77, mul__94, out_122, gelu_78, mul__95, out_123], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf290, arg115_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg115_1
        # Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121, gelu_77, mul__94, out_122, gelu_78, mul__95, out_123], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf292 = extern_kernels.convolution(buf290, buf291, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf290
        del buf291
        buf294 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121, gelu_77, mul__94, out_122, gelu_78, mul__95, out_123, x_se_68, x_se_69], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36.run(buf292, arg118_1, buf294, 12288, 256, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121, gelu_77, mul__94, out_122, gelu_78, mul__95, out_123, x_se_68, x_se_69], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        buf295 = extern_kernels.convolution(buf294, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg119_1
        del buf294
        buf296 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121, gelu_77, mul__94, out_122, gelu_78, mul__95, out_123, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf296, arg120_1, 6144, grid=grid(6144), stream=stream0)
        del arg120_1
        # Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121, gelu_77, mul__94, out_122, gelu_78, mul__95, out_123, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf297 = extern_kernels.convolution(buf296, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg121_1
        del buf296
        buf298 = buf280; del buf280  # reuse
        buf299 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [gelu_75, mul__92, out_119, out_120, gelu_76, mul__93, out_121, gelu_77, mul__94, out_122, gelu_78, mul__95, out_123, x_se_68, x_se_69, x_se_70, x_se_71, sigmoid_17, mul_157, out_124, mul__96, mul_159, out_125, gelu_79, mul__97, out_126, out_127], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_41.run(buf298, buf292, arg118_1, buf297, arg122_1, arg123_1, buf299, 3145728, grid=grid(3145728), stream=stream0)
        del arg118_1
        del arg122_1
        del arg123_1
        del buf292
        # Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf301 = extern_kernels.convolution(buf299, buf300, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf300
        buf302 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf302, arg126_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg126_1
        # Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf304 = extern_kernels.convolution(buf302, buf303, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf304, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf302
        del buf303
        buf305 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128, gelu_81, mul__99, out_129], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf305, arg129_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg129_1
        # Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128, gelu_81, mul__99, out_129], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf307 = extern_kernels.convolution(buf305, buf306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf307, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf305
        del buf306
        buf308 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128, gelu_81, mul__99, out_129, gelu_82, mul__100, out_130], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf308, arg132_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg132_1
        # Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128, gelu_81, mul__99, out_129, gelu_82, mul__100, out_130], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf310 = extern_kernels.convolution(buf308, buf309, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf308
        del buf309
        buf312 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128, gelu_81, mul__99, out_129, gelu_82, mul__100, out_130, x_se_72, x_se_73], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36.run(buf310, arg135_1, buf312, 12288, 256, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128, gelu_81, mul__99, out_129, gelu_82, mul__100, out_130, x_se_72, x_se_73], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        buf313 = extern_kernels.convolution(buf312, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg136_1
        del buf312
        buf314 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128, gelu_81, mul__99, out_129, gelu_82, mul__100, out_130, x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf314, arg137_1, 6144, grid=grid(6144), stream=stream0)
        del arg137_1
        # Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128, gelu_81, mul__99, out_129, gelu_82, mul__100, out_130, x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf315 = extern_kernels.convolution(buf314, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg138_1
        del buf314
        buf316 = buf298; del buf298  # reuse
        buf317 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [gelu_79, mul__97, out_126, out_127, gelu_80, mul__98, out_128, gelu_81, mul__99, out_129, gelu_82, mul__100, out_130, x_se_72, x_se_73, x_se_74, x_se_75, sigmoid_18, mul_165, out_131, mul__101, mul_167, out_132, gelu_83, mul__102, out_133, out_134], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_42.run(buf316, buf310, arg135_1, buf315, arg139_1, arg140_1, buf317, 3145728, grid=grid(3145728), stream=stream0)
        del arg135_1
        del arg139_1
        del arg140_1
        del buf310
        # Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf319 = extern_kernels.convolution(buf317, buf318, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf318
        buf320 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf320, arg143_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg143_1
        # Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf322 = extern_kernels.convolution(buf320, buf321, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf322, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf320
        del buf321
        buf323 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135, gelu_85, mul__104, out_136], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf323, arg146_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg146_1
        # Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135, gelu_85, mul__104, out_136], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf325 = extern_kernels.convolution(buf323, buf324, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf325, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf323
        del buf324
        buf326 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135, gelu_85, mul__104, out_136, gelu_86, mul__105, out_137], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf326, arg149_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg149_1
        # Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135, gelu_85, mul__104, out_136, gelu_86, mul__105, out_137], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf328 = extern_kernels.convolution(buf326, buf327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf326
        del buf327
        buf330 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135, gelu_85, mul__104, out_136, gelu_86, mul__105, out_137, x_se_76, x_se_77], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36.run(buf328, arg152_1, buf330, 12288, 256, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135, gelu_85, mul__104, out_136, gelu_86, mul__105, out_137, x_se_76, x_se_77], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        buf331 = extern_kernels.convolution(buf330, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg153_1
        del buf330
        buf332 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135, gelu_85, mul__104, out_136, gelu_86, mul__105, out_137, x_se_76, x_se_77, x_se_78, x_se_79], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf332, arg154_1, 6144, grid=grid(6144), stream=stream0)
        del arg154_1
        # Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135, gelu_85, mul__104, out_136, gelu_86, mul__105, out_137, x_se_76, x_se_77, x_se_78, x_se_79], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf333 = extern_kernels.convolution(buf332, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg155_1
        del buf332
        buf334 = buf316; del buf316  # reuse
        buf335 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [gelu_83, mul__102, out_133, out_134, gelu_84, mul__103, out_135, gelu_85, mul__104, out_136, gelu_86, mul__105, out_137, x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, mul_173, out_138, mul__106, mul_175, out_139, gelu_87, mul__107, out_140, out_141], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_43.run(buf334, buf328, arg152_1, buf333, arg156_1, arg157_1, buf335, 3145728, grid=grid(3145728), stream=stream0)
        del arg152_1
        del arg156_1
        del arg157_1
        del buf328
        # Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf337 = extern_kernels.convolution(buf335, buf336, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf336
        buf338 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf338, arg160_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg160_1
        # Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf340 = extern_kernels.convolution(buf338, buf339, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf340, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf338
        del buf339
        buf341 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142, gelu_89, mul__109, out_143], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf341, arg163_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg163_1
        # Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142, gelu_89, mul__109, out_143], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf343 = extern_kernels.convolution(buf341, buf342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf343, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf341
        del buf342
        buf344 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142, gelu_89, mul__109, out_143, gelu_90, mul__110, out_144], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf344, arg166_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg166_1
        # Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142, gelu_89, mul__109, out_143, gelu_90, mul__110, out_144], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf346 = extern_kernels.convolution(buf344, buf345, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf344
        del buf345
        buf348 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142, gelu_89, mul__109, out_143, gelu_90, mul__110, out_144, x_se_80, x_se_81], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36.run(buf346, arg169_1, buf348, 12288, 256, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142, gelu_89, mul__109, out_143, gelu_90, mul__110, out_144, x_se_80, x_se_81], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        buf349 = extern_kernels.convolution(buf348, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg170_1
        del buf348
        buf350 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142, gelu_89, mul__109, out_143, gelu_90, mul__110, out_144, x_se_80, x_se_81, x_se_82, x_se_83], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf350, arg171_1, 6144, grid=grid(6144), stream=stream0)
        del arg171_1
        # Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142, gelu_89, mul__109, out_143, gelu_90, mul__110, out_144, x_se_80, x_se_81, x_se_82, x_se_83], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf351 = extern_kernels.convolution(buf350, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg172_1
        del buf350
        buf352 = buf334; del buf334  # reuse
        buf353 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [gelu_87, mul__107, out_140, out_141, gelu_88, mul__108, out_142, gelu_89, mul__109, out_143, gelu_90, mul__110, out_144, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, mul_181, out_145, mul__111, mul_183, out_146, gelu_91, mul__112, out_147], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_44.run(buf352, buf346, arg169_1, buf351, arg173_1, arg174_1, buf353, 3145728, grid=grid(3145728), stream=stream0)
        del arg169_1
        del arg173_1
        del arg174_1
        del buf346
        del buf352
        # Topologically Sorted Source Nodes: [out_148], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf353, buf354, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf354
        buf356 = empty_strided_cuda((8, 768, 17, 17), (221952, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_45.run(buf355, arg180_1, buf356, 1775616, grid=grid(1775616), stream=stream0)
        del arg180_1
        del buf355
        # Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        buf358 = extern_kernels.convolution(buf356, buf357, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf358, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf356
        del buf357
        buf359 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf359, arg183_1, 393216, grid=grid(393216), stream=stream0)
        del arg183_1
        # Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        buf361 = extern_kernels.convolution(buf359, buf360, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf361, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf359
        del buf360
        buf362 = buf361; del buf361  # reuse
        # Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150, gelu_94, mul__115, out_151], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf362, arg186_1, 393216, grid=grid(393216), stream=stream0)
        del arg186_1
        # Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150, gelu_94, mul__115, out_151], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd]
        buf364 = extern_kernels.convolution(buf362, buf363, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (8, 1536, 8, 8), (98304, 64, 8, 1))
        del buf362
        del buf363
        buf366 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150, gelu_94, mul__115, out_151, x_se_84, x_se_85], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_47.run(buf364, arg189_1, buf366, 12288, 64, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150, gelu_94, mul__115, out_151, x_se_84, x_se_85], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean]
        buf367 = extern_kernels.convolution(buf366, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg190_1
        del buf366
        buf368 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150, gelu_94, mul__115, out_151, x_se_84, x_se_85, x_se_86, x_se_87], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf368, arg191_1, 6144, grid=grid(6144), stream=stream0)
        del arg191_1
        # Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150, gelu_94, mul__115, out_151, x_se_84, x_se_85, x_se_86, x_se_87], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu]
        buf369 = extern_kernels.convolution(buf368, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg192_1
        del buf368
        buf370 = reinterpret_tensor(buf260, (8, 1536, 8, 8), (98304, 64, 8, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [avg_pool2d_5, shortcut_7], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_48.run(buf353, buf370, 786432, grid=grid(786432), stream=stream0)
        del buf353
        # Topologically Sorted Source Nodes: [avg_pool2d_5, shortcut_7], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf372 = extern_kernels.convolution(buf370, buf371, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (8, 1536, 8, 8), (98304, 64, 8, 1))
        del buf371
        buf373 = buf364; del buf364  # reuse
        buf374 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [out_148, gelu_92, mul__113, x_15, out_149, gelu_93, mul__114, out_150, gelu_94, mul__115, out_151, x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, mul_190, out_152, mul__116, mul_192, avg_pool2d_5, shortcut_7, out_153, gelu_95, mul__117, out_154, out_155], Original ATen: [aten.convolution, aten.gelu, aten.mul, aten.constant_pad_nd, aten.mean, aten.relu, aten.sigmoid, aten.avg_pool2d, aten.add]
        triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_49.run(buf373, arg189_1, buf369, arg193_1, arg194_1, buf372, arg177_1, buf374, 786432, grid=grid(786432), stream=stream0)
        del arg177_1
        del arg189_1
        del arg193_1
        del arg194_1
        del buf372
        # Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf376 = extern_kernels.convolution(buf374, buf375, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf375
        buf377 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf377, arg197_1, 393216, grid=grid(393216), stream=stream0)
        del arg197_1
        # Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf379 = extern_kernels.convolution(buf377, buf378, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf379, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf377
        del buf378
        buf380 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156, gelu_97, mul__119, out_157], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf380, arg200_1, 393216, grid=grid(393216), stream=stream0)
        del arg200_1
        # Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156, gelu_97, mul__119, out_157], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf382 = extern_kernels.convolution(buf380, buf381, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf382, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf380
        del buf381
        buf383 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156, gelu_97, mul__119, out_157, gelu_98, mul__120, out_158], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf383, arg203_1, 393216, grid=grid(393216), stream=stream0)
        del arg203_1
        # Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156, gelu_97, mul__119, out_157, gelu_98, mul__120, out_158], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf385 = extern_kernels.convolution(buf383, buf384, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (8, 1536, 8, 8), (98304, 64, 8, 1))
        del buf383
        del buf384
        buf387 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156, gelu_97, mul__119, out_157, gelu_98, mul__120, out_158, x_se_88, x_se_89], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_47.run(buf385, arg206_1, buf387, 12288, 64, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156, gelu_97, mul__119, out_157, gelu_98, mul__120, out_158, x_se_88, x_se_89], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        buf388 = extern_kernels.convolution(buf387, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf388, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg207_1
        del buf387
        buf389 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156, gelu_97, mul__119, out_157, gelu_98, mul__120, out_158, x_se_88, x_se_89, x_se_90, x_se_91], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf389, arg208_1, 6144, grid=grid(6144), stream=stream0)
        del arg208_1
        # Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156, gelu_97, mul__119, out_157, gelu_98, mul__120, out_158, x_se_88, x_se_89, x_se_90, x_se_91], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf390 = extern_kernels.convolution(buf389, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg209_1
        del buf389
        buf391 = buf373; del buf373  # reuse
        buf392 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [gelu_95, mul__117, out_154, out_155, gelu_96, mul__118, out_156, gelu_97, mul__119, out_157, gelu_98, mul__120, out_158, x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, mul_198, out_159, mul__121, mul_200, out_160, gelu_99, mul__122, out_161, out_162], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_50.run(buf391, buf385, arg206_1, buf390, arg210_1, arg211_1, buf392, 786432, grid=grid(786432), stream=stream0)
        del arg206_1
        del arg210_1
        del arg211_1
        del buf385
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf394 = extern_kernels.convolution(buf392, buf393, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf392
        del buf393
        buf395 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf395, arg214_1, 393216, grid=grid(393216), stream=stream0)
        del arg214_1
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf397 = extern_kernels.convolution(buf395, buf396, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf397, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf395
        del buf396
        buf398 = buf397; del buf397  # reuse
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf398, arg217_1, 393216, grid=grid(393216), stream=stream0)
        del arg217_1
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf400 = extern_kernels.convolution(buf398, buf399, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf400, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf398
        del buf399
        buf401 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf401, arg220_1, 393216, grid=grid(393216), stream=stream0)
        del arg220_1
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165], Original ATen: [aten.gelu, aten.mul, aten.convolution]
        buf403 = extern_kernels.convolution(buf401, buf402, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (8, 1536, 8, 8), (98304, 64, 8, 1))
        del buf401
        del buf402
        buf405 = buf390; del buf390  # reuse
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165, x_se_92, x_se_93], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_47.run(buf403, arg223_1, buf405, 12288, 64, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165, x_se_92, x_se_93], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean]
        buf406 = extern_kernels.convolution(buf405, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg224_1
        del buf405
        buf407 = buf406; del buf406  # reuse
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165, x_se_92, x_se_93, x_se_94, x_se_95], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf407, arg225_1, 6144, grid=grid(6144), stream=stream0)
        del arg225_1
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165, x_se_92, x_se_93, x_se_94, x_se_95], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf408 = extern_kernels.convolution(buf407, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg226_1
        del buf407
        buf409 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, mul_206, out_166, mul__126, mul_208, out_167, x_16], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_51.run(buf409, buf403, arg223_1, buf408, arg227_1, arg228_1, 786432, grid=grid(786432), stream=stream0)
        del arg223_1
        del arg227_1
        del arg228_1
        del buf403
        del buf408
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, mul_206, out_166, mul__126, mul_208, out_167, x_16], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        buf411 = extern_kernels.convolution(buf409, buf410, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (8, 3072, 8, 8), (196608, 64, 8, 1))
        del buf409
        del buf410
        buf413 = empty_strided_cuda((8, 3072, 1, 1), (3072, 1, 24576, 24576), torch.float32)
        # Topologically Sorted Source Nodes: [gelu_99, mul__122, out_161, out_162, gelu_100, mul__123, out_163, gelu_101, mul__124, out_164, gelu_102, mul__125, out_165, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, mul_206, out_166, mul__126, mul_208, out_167, x_16, gelu_103, x_17, x_18], Original ATen: [aten.gelu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_per_fused_add_convolution_gelu_mean_mul_relu_sigmoid_52.run(buf411, arg231_1, buf413, 24576, 64, grid=grid(24576), stream=stream0)
        del arg231_1
        del buf411
        buf414 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg233_1, reinterpret_tensor(buf413, (8, 3072), (3072, 1), 0), reinterpret_tensor(arg232_1, (3072, 1000), (1, 3072), 0), alpha=1, beta=1, out=buf414)
        del arg232_1
        del arg233_1
        del buf413
    return (buf414, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((3072, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((3072, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1000, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dm_nfnet_f0', benchmark_compiled_module)
