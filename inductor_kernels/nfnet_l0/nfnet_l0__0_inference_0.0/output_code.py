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


# kernel path: /tmp/torchinductor_sahanp/mu/cmu6szj4z2eo6o4rc4ipujhdfjjxaptb3jlptexno4d42axltjny.py
# Topologically Sorted Source Nodes: [batch_norm_57], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_57 => var_mean_57
# Graph fragment:
#   %var_mean_57 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_172, [0, 2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused__native_batch_norm_legit_0 = async_compile.triton('triton_per_fused__native_batch_norm_legit_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oc/cocltq6bc6zv6nf7j44ku7rgx7aacd7jxj6q6v2ux5mugjrikxhb.py
# Topologically Sorted Source Nodes: [batch_norm_58], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_58 => var_mean_58
# Graph fragment:
#   %var_mean_58 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_175, [0, 2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused__native_batch_norm_legit_1 = async_compile.triton('triton_per_fused__native_batch_norm_legit_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_1(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/np/cnpbkzydcz5azholrjx25bxc43rrzommttm7lgk47cw64fsfhhgs.py
# Topologically Sorted Source Nodes: [batch_norm_59], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_59 => var_mean_59
# Graph fragment:
#   %var_mean_59 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_178, [0, 2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused__native_batch_norm_legit_2 = async_compile.triton('triton_per_fused__native_batch_norm_legit_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_2(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sf/csfjv2har2tgsvfyhh5mmri2b7qagwutrkjitt35qeheztsoyy6s.py
# Topologically Sorted Source Nodes: [batch_norm_60], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_60 => var_mean_60
# Graph fragment:
#   %var_mean_60 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_181, [0, 2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused__native_batch_norm_legit_3 = async_compile.triton('triton_per_fused__native_batch_norm_legit_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_3(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jw/cjw36ivmjobxlbndysajn6rt5qimdlveunynp6yebuvde6vvavzk.py
# Topologically Sorted Source Nodes: [batch_norm_61], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_61 => add_73, mul_289, mul_290, rsqrt_61, sub_61, var_mean_61
# Graph fragment:
#   %var_mean_61 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_184, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_184, %getitem_123), kwargs = {})
#   %add_73 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_122, 1e-05), kwargs = {})
#   %rsqrt_61 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_73,), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %rsqrt_61), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_61), kwargs = {})
triton_per_fused__native_batch_norm_legit_4 = async_compile.triton('triton_per_fused__native_batch_norm_legit_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_4(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp25 = 0.1580497968320339
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oi/coisq3vpct65l2gx6qhmfohwrgatc5ez77y62xgk6jsgbu4djloj.py
# Topologically Sorted Source Nodes: [batch_norm_62], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_62 => add_74, mul_292, mul_293, rsqrt_62, sub_62, var_mean_62
# Graph fragment:
#   %var_mean_62 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_187, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_187, %getitem_125), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_124, 1e-05), kwargs = {})
#   %rsqrt_62 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_74,), kwargs = {})
#   %mul_292 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %rsqrt_62), kwargs = {})
#   %mul_293 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_292, %unsqueeze_62), kwargs = {})
triton_per_fused__native_batch_norm_legit_5 = async_compile.triton('triton_per_fused__native_batch_norm_legit_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_5(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp25 = 0.1580497968320339
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6p/c6pho6jndyyp2ipp3nhzxnofbnq5qvc4hywgbmtbtr5enz4bxodk.py
# Topologically Sorted Source Nodes: [batch_norm_63], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_63 => var_mean_63
# Graph fragment:
#   %var_mean_63 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_190, [0, 2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused__native_batch_norm_legit_6 = async_compile.triton('triton_per_fused__native_batch_norm_legit_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_6(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 64
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
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xd/cxdcoqfmmzwkfdq6qgohbkz3e3qoqi7e36zlxuwx7jj2qogioepd.py
# Topologically Sorted Source Nodes: [batch_norm_65], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_65 => add_77, mul_304, mul_305, rsqrt_65, sub_65, var_mean_65
# Graph fragment:
#   %var_mean_65 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_196, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_196, %getitem_131), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_130, 1e-05), kwargs = {})
#   %rsqrt_65 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_77,), kwargs = {})
#   %mul_304 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %rsqrt_65), kwargs = {})
#   %mul_305 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_304, %unsqueeze_65), kwargs = {})
triton_per_fused__native_batch_norm_legit_7 = async_compile.triton('triton_per_fused__native_batch_norm_legit_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_7(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 64.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.22351616621017456
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rt/crti6y5b23m4ksjt24ggssvg6e2ytmzsvabzhckh4fasjppcpxqw.py
# Topologically Sorted Source Nodes: [batch_norm_66], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_66 => add_79, mul_312, mul_313, rsqrt_66, sub_66, var_mean_66
# Graph fragment:
#   %var_mean_66 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_199, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_199, %getitem_133), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_132, 1e-05), kwargs = {})
#   %rsqrt_66 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_79,), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %rsqrt_66), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_312, %unsqueeze_66), kwargs = {})
triton_per_fused__native_batch_norm_legit_8 = async_compile.triton('triton_per_fused__native_batch_norm_legit_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_8(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp22 = 0.11175808310508728
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ag/cagfpxfpr3c4oqpzj337x3qc3oqllse47zu7io2xd4wmjelj6x5e.py
# Topologically Sorted Source Nodes: [batch_norm_67], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_67 => add_80, mul_315, mul_316, rsqrt_67, sub_67, var_mean_67
# Graph fragment:
#   %var_mean_67 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_202, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_202, %getitem_135), kwargs = {})
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_134, 1e-05), kwargs = {})
#   %rsqrt_67 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_80,), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %rsqrt_67), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_315, %unsqueeze_67), kwargs = {})
triton_per_fused__native_batch_norm_legit_9 = async_compile.triton('triton_per_fused__native_batch_norm_legit_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_9(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp22 = 0.11175808310508728
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/du/cdu6gpyiueo3a6qyvuil7gxvqcmtakx6if7dqpfuxmkgwee6fyxc.py
# Topologically Sorted Source Nodes: [batch_norm_70], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_70 => add_83, mul_327, mul_328, rsqrt_70, sub_70, var_mean_70
# Graph fragment:
#   %var_mean_70 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_211, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_211, %getitem_141), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_140, 1e-05), kwargs = {})
#   %rsqrt_70 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_83,), kwargs = {})
#   %mul_327 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %rsqrt_70), kwargs = {})
#   %mul_328 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_327, %unsqueeze_70), kwargs = {})
triton_per_fused__native_batch_norm_legit_10 = async_compile.triton('triton_per_fused__native_batch_norm_legit_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_10(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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
    tmp25 = 0.1580497968320339
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zq/czqqi374gezqjdpctpn5iktaqbfgf2qjdspilcuzsmdo7d5rxtno.py
# Topologically Sorted Source Nodes: [batch_norm_71], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_71 => add_85, mul_335, mul_336, rsqrt_71, sub_71, var_mean_71
# Graph fragment:
#   %var_mean_71 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_214, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_214, %getitem_143), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_142, 1e-05), kwargs = {})
#   %rsqrt_71 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_85,), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %rsqrt_71), kwargs = {})
#   %mul_336 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_335, %unsqueeze_71), kwargs = {})
triton_per_fused__native_batch_norm_legit_11 = async_compile.triton('triton_per_fused__native_batch_norm_legit_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_11(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp22 = 0.07902489841601695
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5d/c5dxhm4vqrf6f2lqom4sqtzspos5ks7qkqwutqwv2leby2pijj5a.py
# Topologically Sorted Source Nodes: [batch_norm_75], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_75 => add_90, mul_355, mul_356, rsqrt_75, sub_75, var_mean_75
# Graph fragment:
#   %var_mean_75 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_226, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_226, %getitem_151), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_150, 1e-05), kwargs = {})
#   %rsqrt_75 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_90,), kwargs = {})
#   %mul_355 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_75, %rsqrt_75), kwargs = {})
#   %mul_356 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_355, %unsqueeze_75), kwargs = {})
triton_per_fused__native_batch_norm_legit_12 = async_compile.triton('triton_per_fused__native_batch_norm_legit_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_12(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp22 = 0.07902489841601695
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eb/ceb5s5ml726ot4kuzgzm4y3efyj62fcx2kagxo6elrls472ze6qs.py
# Topologically Sorted Source Nodes: [batch_norm_76], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_76 => add_91, mul_358, mul_359, rsqrt_76, sub_76, var_mean_76
# Graph fragment:
#   %var_mean_76 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_229, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_229, %getitem_153), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_152, 1e-05), kwargs = {})
#   %rsqrt_76 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_91,), kwargs = {})
#   %mul_358 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %rsqrt_76), kwargs = {})
#   %mul_359 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_358, %unsqueeze_76), kwargs = {})
triton_per_fused__native_batch_norm_legit_13 = async_compile.triton('triton_per_fused__native_batch_norm_legit_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_13(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 384
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
    tmp22 = 0.07902489841601695
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7g/c7gildzjmyoqh6ndkr2yn6ap4roqidhxyqaase3n6zmdlz4a4nxx.py
# Topologically Sorted Source Nodes: [batch_norm_77], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_77 => var_mean_77
# Graph fragment:
#   %var_mean_77 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_232, [0, 2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused__native_batch_norm_legit_14 = async_compile.triton('triton_per_fused__native_batch_norm_legit_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_14(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 384
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
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ig/cigeytwl7vorw7inema5muzgxpubuqezfkfxkgzlr2figaeswfcv.py
# Topologically Sorted Source Nodes: [batch_norm_79], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_79 => add_94, mul_370, mul_371, rsqrt_79, sub_79, var_mean_79
# Graph fragment:
#   %var_mean_79 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_238, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_238, %getitem_159), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_158, 1e-05), kwargs = {})
#   %rsqrt_79 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_94,), kwargs = {})
#   %mul_370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %rsqrt_79), kwargs = {})
#   %mul_371 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_370, %unsqueeze_79), kwargs = {})
triton_per_fused__native_batch_norm_legit_15 = async_compile.triton('triton_per_fused__native_batch_norm_legit_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_15(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1536
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 384, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 384.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.09125009274634042
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp27, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/um/cumdxg2cfz2chbczfveremlzq5gqnpfaazia6gz7ys6wcu6wkmnn.py
# Topologically Sorted Source Nodes: [batch_norm_80], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_80 => add_96, mul_378, mul_379, rsqrt_80, sub_80, var_mean_80
# Graph fragment:
#   %var_mean_80 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_241, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_241, %getitem_161), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_160, 1e-05), kwargs = {})
#   %rsqrt_80 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_96,), kwargs = {})
#   %mul_378 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %rsqrt_80), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_378, %unsqueeze_80), kwargs = {})
triton_red_fused__native_batch_norm_legit_16 = async_compile.triton('triton_red_fused__native_batch_norm_legit_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_16(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp14 = 0.04562504637317021
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp16, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kg/ckgzun4enn6v2brx6yqj2hd4qa6aao3nmoe3742rm6dyhcyu6kjo.py
# Topologically Sorted Source Nodes: [batch_norm_100], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_100 => add_121, mul_478, mul_479, rsqrt_100, sub_100, var_mean_100
# Graph fragment:
#   %var_mean_100 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_301, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_301, %getitem_201), kwargs = {})
#   %add_121 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_200, 1e-05), kwargs = {})
#   %rsqrt_100 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_121,), kwargs = {})
#   %mul_478 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %rsqrt_100), kwargs = {})
#   %mul_479 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_478, %unsqueeze_100), kwargs = {})
triton_red_fused__native_batch_norm_legit_17 = async_compile.triton('triton_red_fused__native_batch_norm_legit_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_17(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp14 = 0.04562504637317021
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp16, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/j6/cj6eibfyxordl2aoniyycymtiaiwgrvspsg4m35natvy4g7pltbd.py
# Topologically Sorted Source Nodes: [batch_norm_113], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_113 => add_137, mul_539, mul_540, rsqrt_113, sub_113, var_mean_113
# Graph fragment:
#   %var_mean_113 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_340, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_340, %getitem_227), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_226, 1e-05), kwargs = {})
#   %rsqrt_113 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_137,), kwargs = {})
#   %mul_539 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %rsqrt_113), kwargs = {})
#   %mul_540 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_539, %unsqueeze_113), kwargs = {})
triton_red_fused__native_batch_norm_legit_18 = async_compile.triton('triton_red_fused__native_batch_norm_legit_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_18(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2304
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
        tmp14 = 0.04562504637317021
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp16, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bk/cbklraagnbjg3jzdz3bmcqazlvnym3f4bi2jeg5kyufpvyoobj2i.py
# Topologically Sorted Source Nodes: [batch_norm_57, input_8], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_57 => add_69, mul_272, mul_273, rsqrt_57, sub_57, var_mean_57
#   input_8 => convolution_81
# Graph fragment:
#   %var_mean_57 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_172, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_172, %getitem_115), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_114, 1e-05), kwargs = {})
#   %rsqrt_57 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_69,), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %rsqrt_57), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_272, %unsqueeze_57), kwargs = {})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg3_1, %view_174, %arg2_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_convolution_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_convolution_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[64, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_convolution_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y1 = (yindex // 3)
    y0 = yindex % 3
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y1), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 27.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = 0.34412564994580647
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 * tmp12
    tl.store(out_ptr1 + (y0 + (3*x2) + (27*y1)), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sm/csmh6z2uyyv3fwwdnnrdaowwcmk5f7yvpcwg2h7ab3wmrlectp33.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_8 => convolution_81
# Graph fragment:
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg3_1, %view_174, %arg2_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_20 = async_compile.triton('triton_poi_fused_convolution_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32, 131072], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 82944
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
    tmp0 = tl.load(in_ptr0 + (x2 + (82944*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (248832*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dq/cdqk226hpgeyeviyi7fi54njhkjctvp36hjknmbmlivd3mm6gij5.py
# Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.convolution, aten.silu]
# Source node to ATen node mapping:
#   input_8 => convolution_81
#   input_9 => mul_274, sigmoid_64
# Graph fragment:
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg3_1, %view_174, %arg2_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_64 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_81,), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, %sigmoid_64), kwargs = {})
triton_poi_fused_convolution_silu_21 = async_compile.triton('triton_poi_fused_convolution_silu_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5i/c5iecaeggnifasexisdcjndnfsfphpnvt4seqksviqtykzudzure.py
# Topologically Sorted Source Nodes: [batch_norm_58, input_8, input_9, input_10], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
# Source node to ATen node mapping:
#   batch_norm_58 => add_70, mul_276, mul_277, rsqrt_58, sub_58, var_mean_58
#   input_10 => convolution_82
#   input_8 => convolution_81
#   input_9 => mul_274, sigmoid_64
# Graph fragment:
#   %var_mean_58 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_175, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg3_1, %view_174, %arg2_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_64 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_81,), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, %sigmoid_64), kwargs = {})
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_175, %getitem_117), kwargs = {})
#   %add_70 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_116, 1e-05), kwargs = {})
#   %rsqrt_58 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_70,), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %rsqrt_58), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_276, %unsqueeze_58), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_274, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_convolution_silu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_convolution_silu_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_convolution_silu_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_silu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y1 = (yindex // 16)
    y0 = yindex % 16
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y1), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 144.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = 0.1490107774734497
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 * tmp12
    tl.store(out_ptr1 + (y0 + (16*x2) + (144*y1)), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n6/cn6gbhywrrqz3wteebemnzwrut4egv6zwgmfelh6a6mgk5f4h2sq.py
# Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.silu]
# Source node to ATen node mapping:
#   input_10 => convolution_82
#   input_11 => mul_278, sigmoid_65
#   input_8 => convolution_81
#   input_9 => mul_274, sigmoid_64
# Graph fragment:
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg3_1, %view_174, %arg2_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_64 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_81,), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, %sigmoid_64), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_274, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_65 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_82,), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, %sigmoid_65), kwargs = {})
triton_poi_fused_convolution_silu_23 = async_compile.triton('triton_poi_fused_convolution_silu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_23(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3b/c3bwfkkwo62smozl2bp7wytc3jz5xw2huiymkf4lgfh2ovig2zn4.py
# Topologically Sorted Source Nodes: [batch_norm_59, input_8, input_9, input_10, input_11, input_12], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
# Source node to ATen node mapping:
#   batch_norm_59 => add_71, mul_280, mul_281, rsqrt_59, sub_59, var_mean_59
#   input_10 => convolution_82
#   input_11 => mul_278, sigmoid_65
#   input_12 => convolution_83
#   input_8 => convolution_81
#   input_9 => mul_274, sigmoid_64
# Graph fragment:
#   %var_mean_59 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_178, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg3_1, %view_174, %arg2_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_64 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_81,), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, %sigmoid_64), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_274, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_65 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_82,), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, %sigmoid_65), kwargs = {})
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_178, %getitem_119), kwargs = {})
#   %add_71 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_118, 1e-05), kwargs = {})
#   %rsqrt_59 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_71,), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %rsqrt_59), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_280, %unsqueeze_59), kwargs = {})
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_278, %view_180, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_convolution_silu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_convolution_silu_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_convolution_silu_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_silu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y1 = (yindex // 32)
    y0 = yindex % 32
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 288.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = 0.10536653122135592
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 * tmp12
    tl.store(out_ptr1 + (y0 + (32*x2) + (288*y1)), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fx/cfxpx3ywl33dtytz7l2pxjx2wjm7jwrjqkqks45y36gfpcfnxsud.py
# Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11, input_12, input_13], Original ATen: [aten.convolution, aten.silu]
# Source node to ATen node mapping:
#   input_10 => convolution_82
#   input_11 => mul_278, sigmoid_65
#   input_12 => convolution_83
#   input_13 => mul_282, sigmoid_66
#   input_8 => convolution_81
#   input_9 => mul_274, sigmoid_64
# Graph fragment:
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg3_1, %view_174, %arg2_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_64 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_81,), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, %sigmoid_64), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_274, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_65 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_82,), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, %sigmoid_65), kwargs = {})
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_278, %view_180, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_66 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_83,), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_83, %sigmoid_66), kwargs = {})
triton_poi_fused_convolution_silu_25 = async_compile.triton('triton_poi_fused_convolution_silu_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_25(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10616832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sf/csf4iqdw7qnc7ldewht36ltjtitxmkkzrltatvrmp2dtjlis5fv6.py
# Topologically Sorted Source Nodes: [batch_norm_60, input_8, input_9, input_10, input_11, input_12, input_13, input_14], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
# Source node to ATen node mapping:
#   batch_norm_60 => add_72, mul_284, mul_285, rsqrt_60, sub_60, var_mean_60
#   input_10 => convolution_82
#   input_11 => mul_278, sigmoid_65
#   input_12 => convolution_83
#   input_13 => mul_282, sigmoid_66
#   input_14 => convolution_84
#   input_8 => convolution_81
#   input_9 => mul_274, sigmoid_64
# Graph fragment:
#   %var_mean_60 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_181, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg3_1, %view_174, %arg2_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_64 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_81,), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, %sigmoid_64), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_274, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_65 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_82,), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, %sigmoid_65), kwargs = {})
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_278, %view_180, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_66 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_83,), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_83, %sigmoid_66), kwargs = {})
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_181, %getitem_121), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_120, 1e-05), kwargs = {})
#   %rsqrt_60 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_72,), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %rsqrt_60), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_284, %unsqueeze_60), kwargs = {})
#   %convolution_84 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_282, %view_183, %arg12_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_convolution_silu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_convolution_silu_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_convolution_silu_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_silu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y1 = (yindex // 64)
    y0 = yindex % 64
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 576.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = 0.07450538873672485
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 * tmp12
    tl.store(out_ptr1 + (y0 + (64*x2) + (576*y1)), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g7/cg7m6xd4ueuhqqfqw7dmckwptynyrd6txyrj32wdxu77llgnifx3.py
# Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11, input_12, input_13, input_14, silu_55, out_84], Original ATen: [aten.convolution, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   input_10 => convolution_82
#   input_11 => mul_278, sigmoid_65
#   input_12 => convolution_83
#   input_13 => mul_282, sigmoid_66
#   input_14 => convolution_84
#   input_8 => convolution_81
#   input_9 => mul_274, sigmoid_64
#   out_84 => mul_287
#   silu_55 => mul_286, sigmoid_67
# Graph fragment:
#   %convolution_81 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg3_1, %view_174, %arg2_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_64 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_81,), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, %sigmoid_64), kwargs = {})
#   %convolution_82 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_274, %view_177, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_65 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_82,), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, %sigmoid_65), kwargs = {})
#   %convolution_83 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_278, %view_180, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_66 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_83,), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_83, %sigmoid_66), kwargs = {})
#   %convolution_84 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_282, %view_183, %arg12_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_67 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_84,), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_84, %sigmoid_67), kwargs = {})
#   %mul_287 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_286, 1.0), kwargs = {})
triton_poi_fused_convolution_mul_silu_27 = async_compile.triton('triton_poi_fused_convolution_mul_silu_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_silu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_silu_27(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qt/cqtpv4nyioyp5g7rc4loynlrllmhqt363wtpizvzh2axv7ie2dxa.py
# Topologically Sorted Source Nodes: [out_85, silu_56], Original ATen: [aten.convolution, aten.silu]
# Source node to ATen node mapping:
#   out_85 => convolution_86
#   silu_56 => mul_294, sigmoid_68
# Graph fragment:
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_287, %view_189, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_68 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_86,), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, %sigmoid_68), kwargs = {})
triton_poi_fused_convolution_silu_28 = async_compile.triton('triton_poi_fused_convolution_silu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_28(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vc/cvcxdhsfqatqnbeqg2x64zrtlwyzl36q7thnrnljefjkj6lsmyho.py
# Topologically Sorted Source Nodes: [batch_norm_63, out_85, silu_56, out_86], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
# Source node to ATen node mapping:
#   batch_norm_63 => add_75, mul_296, mul_297, rsqrt_63, sub_63, var_mean_63
#   out_85 => convolution_86
#   out_86 => convolution_87
#   silu_56 => mul_294, sigmoid_68
# Graph fragment:
#   %var_mean_63 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_190, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_287, %view_189, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_68 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_86,), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, %sigmoid_68), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_190, %getitem_127), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_126, 1e-05), kwargs = {})
#   %rsqrt_63 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_75,), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %rsqrt_63), kwargs = {})
#   %mul_297 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_296, %unsqueeze_63), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_294, %view_192, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_convolution_silu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_convolution_silu_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_convolution_silu_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_silu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y1 = (yindex // 64)
    y0 = yindex % 64
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 576.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = 0.07450538873672485
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 * tmp12
    tl.store(out_ptr1 + (y0 + (64*x2) + (576*y1)), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qo/cqofpkytfjnxdg7hokgmlh4x4trmenzuqta2qbim4njvkiss6grg.py
# Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87, silu_58, out_88, x_se_48], Original ATen: [aten.convolution, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   out_85 => convolution_86
#   out_86 => convolution_87
#   out_87 => convolution_88
#   out_88 => convolution_89
#   silu_56 => mul_294, sigmoid_68
#   silu_57 => mul_298, sigmoid_69
#   silu_58 => mul_302, sigmoid_70
#   x_se_48 => mean_13
# Graph fragment:
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_287, %view_189, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_68 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_86,), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, %sigmoid_68), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_294, %view_192, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_69 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_87,), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, %sigmoid_69), kwargs = {})
#   %convolution_88 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_298, %view_195, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_70 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_88,), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, %sigmoid_70), kwargs = {})
#   %convolution_89 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_302, %view_198, %arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_89, [2, 3], True), kwargs = {})
triton_red_fused_convolution_mean_silu_30 = async_compile.triton('triton_red_fused_convolution_mean_silu_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_silu_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_mean_silu_30(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 83968
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256) % 41
    x0 = xindex % 256
    x2 = (xindex // 10496)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (127*x1)
        tmp1 = tl.full([1, 1], 5184, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r3 + (127*x1)) % 5184)) + (1327104*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2m/c2mpf2qtc4b6g3paylt7i4siw6tozj73ta6juwxarz7byazym6jw.py
# Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87, silu_58, out_88, x_se_48], Original ATen: [aten.convolution, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   out_85 => convolution_86
#   out_86 => convolution_87
#   out_87 => convolution_88
#   out_88 => convolution_89
#   silu_56 => mul_294, sigmoid_68
#   silu_57 => mul_298, sigmoid_69
#   silu_58 => mul_302, sigmoid_70
#   x_se_48 => mean_13
# Graph fragment:
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_287, %view_189, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_68 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_86,), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, %sigmoid_68), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_294, %view_192, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_69 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_87,), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, %sigmoid_69), kwargs = {})
#   %convolution_88 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_298, %view_195, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_70 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_88,), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, %sigmoid_70), kwargs = {})
#   %convolution_89 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_302, %view_198, %arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_89, [2, 3], True), kwargs = {})
triton_per_fused_convolution_mean_silu_31 = async_compile.triton('triton_per_fused_convolution_mean_silu_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_silu_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_mean_silu_31(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 41
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (10496*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 5184.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ab/cabucgyz2ddykypjtl2bepq5u4xhgu4hf6tsplokshnvps4cv2fo.py
# Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87, silu_58, out_88, x_se_48, x_se_49, x_se_50], Original ATen: [aten.convolution, aten.silu, aten.mean, aten.relu]
# Source node to ATen node mapping:
#   out_85 => convolution_86
#   out_86 => convolution_87
#   out_87 => convolution_88
#   out_88 => convolution_89
#   silu_56 => mul_294, sigmoid_68
#   silu_57 => mul_298, sigmoid_69
#   silu_58 => mul_302, sigmoid_70
#   x_se_48 => mean_13
#   x_se_49 => convolution_90
#   x_se_50 => relu_12
# Graph fragment:
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_287, %view_189, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_68 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_86,), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, %sigmoid_68), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_294, %view_192, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_69 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_87,), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, %sigmoid_69), kwargs = {})
#   %convolution_88 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_298, %view_195, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_70 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_88,), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, %sigmoid_70), kwargs = {})
#   %convolution_89 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_302, %view_198, %arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_89, [2, 3], True), kwargs = {})
#   %convolution_90 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_13, %arg28_1, %arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_90,), kwargs = {})
triton_poi_fused_convolution_mean_relu_silu_32 = async_compile.triton('triton_poi_fused_convolution_mean_relu_silu_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_silu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_silu_32(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xh/cxhr5ise2snepkwzgx76vabp2e2pb6pt64k5wqmhavwezmmaxadb.py
# Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87, silu_58, out_88, x_se_48, x_se_49, x_se_50, x_se_51, sigmoid_12, mul_115, out_89, mul_117, shortcut_4, out_90, silu_59, out_91], Original ATen: [aten.convolution, aten.silu, aten.mean, aten.relu, aten.sigmoid, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_115 => mul_306
#   mul_117 => mul_308
#   out_85 => convolution_86
#   out_86 => convolution_87
#   out_87 => convolution_88
#   out_88 => convolution_89
#   out_89 => mul_307
#   out_90 => add_78
#   out_91 => mul_310
#   shortcut_4 => convolution_85
#   sigmoid_12 => sigmoid_71
#   silu_56 => mul_294, sigmoid_68
#   silu_57 => mul_298, sigmoid_69
#   silu_58 => mul_302, sigmoid_70
#   silu_59 => mul_309, sigmoid_72
#   x_se_48 => mean_13
#   x_se_49 => convolution_90
#   x_se_50 => relu_12
#   x_se_51 => convolution_91
# Graph fragment:
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_287, %view_189, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_68 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_86,), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, %sigmoid_68), kwargs = {})
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_294, %view_192, %arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_69 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_87,), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, %sigmoid_69), kwargs = {})
#   %convolution_88 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_298, %view_195, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_70 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_88,), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, %sigmoid_70), kwargs = {})
#   %convolution_89 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_302, %view_198, %arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_89, [2, 3], True), kwargs = {})
#   %convolution_90 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_13, %arg28_1, %arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_90,), kwargs = {})
#   %convolution_91 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg30_1, %arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_71 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_91,), kwargs = {})
#   %mul_306 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_89, %sigmoid_71), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_306, 2.0), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_307, 0.2), kwargs = {})
#   %convolution_85 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_287, %view_186, %arg15_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_78 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_308, %convolution_85), kwargs = {})
#   %sigmoid_72 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_72), kwargs = {})
#   %mul_310 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_309, 0.9805806756909201), kwargs = {})
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_33 = async_compile.triton('triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10616832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 1327104)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), None)
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = 0.9805806756909201
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vq/cvqna5cpj5b2mxrpsqpmobrxyzatho3ilehwbtq5rgi6ruhxt6zw.py
# Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60], Original ATen: [aten.silu, aten.mul, aten.convolution]
# Source node to ATen node mapping:
#   out_91 => mul_310
#   out_92 => convolution_93
#   silu_59 => mul_309, sigmoid_72
#   silu_60 => mul_317, sigmoid_73
# Graph fragment:
#   %sigmoid_72 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_72), kwargs = {})
#   %mul_310 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_309, 0.9805806756909201), kwargs = {})
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_310, %view_204, %arg37_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_73 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_93,), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, %sigmoid_73), kwargs = {})
triton_poi_fused_convolution_mul_silu_34 = async_compile.triton('triton_poi_fused_convolution_mul_silu_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_silu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_silu_34(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ch/cchgy5jsytubltujr3xrmhpke4te4mdpzjtni2p7ch4ykz4h6hms.py
# Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61], Original ATen: [aten.silu, aten.mul, aten.convolution]
# Source node to ATen node mapping:
#   out_91 => mul_310
#   out_92 => convolution_93
#   out_93 => convolution_94
#   silu_59 => mul_309, sigmoid_72
#   silu_60 => mul_317, sigmoid_73
#   silu_61 => mul_321, sigmoid_74
# Graph fragment:
#   %sigmoid_72 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_72), kwargs = {})
#   %mul_310 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_309, 0.9805806756909201), kwargs = {})
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_310, %view_204, %arg37_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_73 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_93,), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, %sigmoid_73), kwargs = {})
#   %convolution_94 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_317, %view_207, %arg40_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %sigmoid_74 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_94,), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, %sigmoid_74), kwargs = {})
triton_poi_fused_convolution_mul_silu_35 = async_compile.triton('triton_poi_fused_convolution_mul_silu_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_silu_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_silu_35(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1327104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6s/c6s5lj67jfsqx5rdye5mfoeo3skxnbk5luszikgo7cetsltvb6el.py
# Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94, silu_62, out_95, x_se_52], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
# Source node to ATen node mapping:
#   out_91 => mul_310
#   out_92 => convolution_93
#   out_93 => convolution_94
#   out_94 => convolution_95
#   out_95 => convolution_96
#   silu_59 => mul_309, sigmoid_72
#   silu_60 => mul_317, sigmoid_73
#   silu_61 => mul_321, sigmoid_74
#   silu_62 => mul_325, sigmoid_75
#   x_se_52 => mean_14
# Graph fragment:
#   %sigmoid_72 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_72), kwargs = {})
#   %mul_310 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_309, 0.9805806756909201), kwargs = {})
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_310, %view_204, %arg37_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_73 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_93,), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, %sigmoid_73), kwargs = {})
#   %convolution_94 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_317, %view_207, %arg40_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %sigmoid_74 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_94,), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, %sigmoid_74), kwargs = {})
#   %convolution_95 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_321, %view_210, %arg43_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %sigmoid_75 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_95,), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_95, %sigmoid_75), kwargs = {})
#   %convolution_96 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_325, %view_213, %arg46_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_96, [2, 3], True), kwargs = {})
triton_red_fused_convolution_mean_mul_silu_36 = async_compile.triton('triton_red_fused_convolution_mean_mul_silu_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_mul_silu_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_mean_mul_silu_36(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 45056
    rnumel = 118
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512) % 11
    x0 = xindex % 512
    x2 = (xindex // 5632)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (118*x1)
        tmp1 = tl.full([1, 1], 1296, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r3 + (118*x1)) % 1296)) + (663552*x2)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sa/csafj2rjcv7234caqgithl74ymme63injgmqqkfizrgwjsoqhvu3.py
# Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94, silu_62, out_95, x_se_52], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
# Source node to ATen node mapping:
#   out_91 => mul_310
#   out_92 => convolution_93
#   out_93 => convolution_94
#   out_94 => convolution_95
#   out_95 => convolution_96
#   silu_59 => mul_309, sigmoid_72
#   silu_60 => mul_317, sigmoid_73
#   silu_61 => mul_321, sigmoid_74
#   silu_62 => mul_325, sigmoid_75
#   x_se_52 => mean_14
# Graph fragment:
#   %sigmoid_72 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_72), kwargs = {})
#   %mul_310 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_309, 0.9805806756909201), kwargs = {})
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_310, %view_204, %arg37_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_73 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_93,), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, %sigmoid_73), kwargs = {})
#   %convolution_94 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_317, %view_207, %arg40_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %sigmoid_74 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_94,), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, %sigmoid_74), kwargs = {})
#   %convolution_95 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_321, %view_210, %arg43_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %sigmoid_75 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_95,), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_95, %sigmoid_75), kwargs = {})
#   %convolution_96 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_325, %view_213, %arg46_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_96, [2, 3], True), kwargs = {})
triton_per_fused_convolution_mean_mul_silu_37 = async_compile.triton('triton_per_fused_convolution_mean_mul_silu_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_mul_silu_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_mean_mul_silu_37(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 11
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (5632*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 1296.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dh/cdh5h3v42jb3x7wepuwmkqvwgac6u76ie5326ncqupyx7ssl44qu.py
# Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94, silu_62, out_95, x_se_52, x_se_53, x_se_54], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
# Source node to ATen node mapping:
#   out_91 => mul_310
#   out_92 => convolution_93
#   out_93 => convolution_94
#   out_94 => convolution_95
#   out_95 => convolution_96
#   silu_59 => mul_309, sigmoid_72
#   silu_60 => mul_317, sigmoid_73
#   silu_61 => mul_321, sigmoid_74
#   silu_62 => mul_325, sigmoid_75
#   x_se_52 => mean_14
#   x_se_53 => convolution_97
#   x_se_54 => relu_13
# Graph fragment:
#   %sigmoid_72 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_72), kwargs = {})
#   %mul_310 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_309, 0.9805806756909201), kwargs = {})
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_310, %view_204, %arg37_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_73 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_93,), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, %sigmoid_73), kwargs = {})
#   %convolution_94 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_317, %view_207, %arg40_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %sigmoid_74 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_94,), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, %sigmoid_74), kwargs = {})
#   %convolution_95 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_321, %view_210, %arg43_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %sigmoid_75 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_95,), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_95, %sigmoid_75), kwargs = {})
#   %convolution_96 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_325, %view_213, %arg46_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_96, [2, 3], True), kwargs = {})
#   %convolution_97 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_14, %arg47_1, %arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_97,), kwargs = {})
triton_poi_fused_convolution_mean_mul_relu_silu_38 = async_compile.triton('triton_poi_fused_convolution_mean_mul_relu_silu_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_silu_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_relu_silu_38(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ku/ckutdqedanlblbuube43mnt3zs6ac2ln22hoscfdba3ybre7z3xc.py
# Topologically Sorted Source Nodes: [avg_pool2d_3], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_3 => avg_pool2d_3
# Graph fragment:
#   %avg_pool2d_3 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_310, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
triton_poi_fused_avg_pool2d_39 = async_compile.triton('triton_poi_fused_avg_pool2d_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256) % 36
    x2 = (xindex // 9216)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (36864*x2)), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (36864*x2)), None)
    tmp3 = tl.load(in_ptr0 + (18432 + x0 + (512*x1) + (36864*x2)), None)
    tmp5 = tl.load(in_ptr0 + (18688 + x0 + (512*x1) + (36864*x2)), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kb/ckbkcqsdlcvfumipv2jcamjt2u6qpoekfmibhek3ahuuw4foqvhk.py
# Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94, silu_62, out_95, x_se_52, x_se_53, x_se_54, x_se_55, sigmoid_13, mul_124, out_96, mul_126, avg_pool2d_3, shortcut_5, out_97, silu_63, out_98], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.avg_pool2d, aten.add]
# Source node to ATen node mapping:
#   avg_pool2d_3 => avg_pool2d_3
#   mul_124 => mul_329
#   mul_126 => mul_331
#   out_91 => mul_310
#   out_92 => convolution_93
#   out_93 => convolution_94
#   out_94 => convolution_95
#   out_95 => convolution_96
#   out_96 => mul_330
#   out_97 => add_84
#   out_98 => mul_333
#   shortcut_5 => convolution_92
#   sigmoid_13 => sigmoid_76
#   silu_59 => mul_309, sigmoid_72
#   silu_60 => mul_317, sigmoid_73
#   silu_61 => mul_321, sigmoid_74
#   silu_62 => mul_325, sigmoid_75
#   silu_63 => mul_332, sigmoid_77
#   x_se_52 => mean_14
#   x_se_53 => convolution_97
#   x_se_54 => relu_13
#   x_se_55 => convolution_98
# Graph fragment:
#   %sigmoid_72 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_72), kwargs = {})
#   %mul_310 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_309, 0.9805806756909201), kwargs = {})
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_310, %view_204, %arg37_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_73 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_93,), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, %sigmoid_73), kwargs = {})
#   %convolution_94 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_317, %view_207, %arg40_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %sigmoid_74 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_94,), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, %sigmoid_74), kwargs = {})
#   %convolution_95 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_321, %view_210, %arg43_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %sigmoid_75 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_95,), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_95, %sigmoid_75), kwargs = {})
#   %convolution_96 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_325, %view_213, %arg46_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_96, [2, 3], True), kwargs = {})
#   %convolution_97 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_14, %arg47_1, %arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_97,), kwargs = {})
#   %convolution_98 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg49_1, %arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_76 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_98,), kwargs = {})
#   %mul_329 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_96, %sigmoid_76), kwargs = {})
#   %mul_330 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_329, 2.0), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_330, 0.2), kwargs = {})
#   %avg_pool2d_3 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_310, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_92 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_3, %view_201, %arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_84 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_331, %convolution_92), kwargs = {})
#   %sigmoid_77 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_84,), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_84, %sigmoid_77), kwargs = {})
#   %mul_333 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_332, 0.9805806756909201), kwargs = {})
triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_40 = async_compile.triton('triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_40', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 663552)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), None)
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = 0.9805806756909201
    tmp19 = tmp17 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2g/c2gt6cfh5q76r4uqoojh643sp5ytgxdmyoyq3rltg5ndrv6dvn3i.py
# Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64, out_100, silu_65, out_101, silu_66, out_102, x_se_56, x_se_57, x_se_58, x_se_59, sigmoid_14, mul_132, out_103, mul_134, out_104, silu_67, out_105], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   mul_132 => mul_349
#   mul_134 => mul_351
#   out_100 => convolution_100
#   out_101 => convolution_101
#   out_102 => convolution_102
#   out_103 => mul_350
#   out_104 => add_89
#   out_105 => mul_353
#   out_98 => mul_333
#   out_99 => convolution_99
#   sigmoid_14 => sigmoid_81
#   silu_63 => mul_332, sigmoid_77
#   silu_64 => mul_337, sigmoid_78
#   silu_65 => mul_341, sigmoid_79
#   silu_66 => mul_345, sigmoid_80
#   silu_67 => mul_352, sigmoid_82
#   x_se_56 => mean_15
#   x_se_57 => convolution_103
#   x_se_58 => relu_14
#   x_se_59 => convolution_104
# Graph fragment:
#   %sigmoid_77 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_84,), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_84, %sigmoid_77), kwargs = {})
#   %mul_333 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_332, 0.9805806756909201), kwargs = {})
#   %convolution_99 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_333, %view_216, %arg53_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_78 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_99,), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_99, %sigmoid_78), kwargs = {})
#   %convolution_100 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_337, %view_219, %arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %sigmoid_79 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_100,), kwargs = {})
#   %mul_341 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_100, %sigmoid_79), kwargs = {})
#   %convolution_101 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_341, %view_222, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
#   %sigmoid_80 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_101,), kwargs = {})
#   %mul_345 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_101, %sigmoid_80), kwargs = {})
#   %convolution_102 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_345, %view_225, %arg62_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_102, [2, 3], True), kwargs = {})
#   %convolution_103 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_15, %arg63_1, %arg64_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_103,), kwargs = {})
#   %convolution_104 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg65_1, %arg66_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_81 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_104,), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_102, %sigmoid_81), kwargs = {})
#   %mul_350 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_349, 2.0), kwargs = {})
#   %mul_351 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_350, 0.2), kwargs = {})
#   %add_89 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_351, %add_84), kwargs = {})
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_82), kwargs = {})
#   %mul_353 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, 0.9622504486493761), kwargs = {})
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_41 = async_compile.triton('triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 663552)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.9622504486493761
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ok/cok4klufkrcnieahzvs6k6akfp77udy6qnwmz4uuusb7wkcs7tke.py
# Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68], Original ATen: [aten.silu, aten.mul, aten.convolution]
# Source node to ATen node mapping:
#   out_105 => mul_353
#   out_106 => convolution_106
#   silu_67 => mul_352, sigmoid_82
#   silu_68 => mul_360, sigmoid_83
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_82), kwargs = {})
#   %mul_353 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, 0.9622504486493761), kwargs = {})
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_353, %view_231, %arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_106,), kwargs = {})
#   %mul_360 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, %sigmoid_83), kwargs = {})
triton_poi_fused_convolution_mul_silu_42 = async_compile.triton('triton_poi_fused_convolution_mul_silu_42', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_silu_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_silu_42(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3k/c3kdrzw5jfl6y5j4upovizd63hkvntel25wlamtqvtnfqqeaygmd.py
# Topologically Sorted Source Nodes: [batch_norm_77, silu_67, out_105, out_106, silu_68, out_107], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_77 => add_92, mul_362, mul_363, rsqrt_77, sub_77, var_mean_77
#   out_105 => mul_353
#   out_106 => convolution_106
#   out_107 => convolution_107
#   silu_67 => mul_352, sigmoid_82
#   silu_68 => mul_360, sigmoid_83
# Graph fragment:
#   %var_mean_77 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_232, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_82), kwargs = {})
#   %mul_353 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, 0.9622504486493761), kwargs = {})
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_353, %view_231, %arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_106,), kwargs = {})
#   %mul_360 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, %sigmoid_83), kwargs = {})
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_232, %getitem_155), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_154, 1e-05), kwargs = {})
#   %rsqrt_77 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_92,), kwargs = {})
#   %mul_362 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %rsqrt_77), kwargs = {})
#   %mul_363 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_362, %unsqueeze_77), kwargs = {})
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_360, %view_234, %arg75_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y1 = (yindex // 64)
    y0 = yindex % 64
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 576.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = 0.07450538873672485
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 * tmp12
    tl.store(out_ptr1 + (y0 + (64*x2) + (576*y1)), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zn/cznqxcgi7tgqepnc2j7hc3qtmvtzeyr6gnxbxcqlzgkt4oydvqts.py
# Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69], Original ATen: [aten.silu, aten.mul, aten.convolution]
# Source node to ATen node mapping:
#   out_105 => mul_353
#   out_106 => convolution_106
#   out_107 => convolution_107
#   silu_67 => mul_352, sigmoid_82
#   silu_68 => mul_360, sigmoid_83
#   silu_69 => mul_364, sigmoid_84
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_82), kwargs = {})
#   %mul_353 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, 0.9622504486493761), kwargs = {})
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_353, %view_231, %arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_106,), kwargs = {})
#   %mul_360 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, %sigmoid_83), kwargs = {})
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_360, %view_234, %arg75_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_84 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_107,), kwargs = {})
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, %sigmoid_84), kwargs = {})
triton_poi_fused_convolution_mul_silu_44 = async_compile.triton('triton_poi_fused_convolution_mul_silu_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_silu_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_silu_44(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 995328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ra/craidpngje3fdopd45n35qgx4c34356tuygxmwgimvvjii5oskhv.py
# Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69, out_108, silu_70, out_109, x_se_60], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
# Source node to ATen node mapping:
#   out_105 => mul_353
#   out_106 => convolution_106
#   out_107 => convolution_107
#   out_108 => convolution_108
#   out_109 => convolution_109
#   silu_67 => mul_352, sigmoid_82
#   silu_68 => mul_360, sigmoid_83
#   silu_69 => mul_364, sigmoid_84
#   silu_70 => mul_368, sigmoid_85
#   x_se_60 => mean_16
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_82), kwargs = {})
#   %mul_353 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, 0.9622504486493761), kwargs = {})
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_353, %view_231, %arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_106,), kwargs = {})
#   %mul_360 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, %sigmoid_83), kwargs = {})
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_360, %view_234, %arg75_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_84 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_107,), kwargs = {})
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, %sigmoid_84), kwargs = {})
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_364, %view_237, %arg78_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_85 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_108,), kwargs = {})
#   %mul_368 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, %sigmoid_85), kwargs = {})
#   %convolution_109 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_368, %view_240, %arg81_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_109, [2, 3], True), kwargs = {})
triton_red_fused_convolution_mean_mul_silu_45 = async_compile.triton('triton_red_fused_convolution_mean_mul_silu_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_mul_silu_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_mean_mul_silu_45(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 324
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1536
    x1 = (xindex // 1536)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1536*r2) + (497664*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 324.0
    tmp7 = tmp4 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/na/cnaob4zqpqk6npofbyynb7vdrw24bfw27yxhvv4al53fdbjci6bp.py
# Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69, out_108, silu_70, out_109, x_se_60, x_se_61, x_se_62], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
# Source node to ATen node mapping:
#   out_105 => mul_353
#   out_106 => convolution_106
#   out_107 => convolution_107
#   out_108 => convolution_108
#   out_109 => convolution_109
#   silu_67 => mul_352, sigmoid_82
#   silu_68 => mul_360, sigmoid_83
#   silu_69 => mul_364, sigmoid_84
#   silu_70 => mul_368, sigmoid_85
#   x_se_60 => mean_16
#   x_se_61 => convolution_110
#   x_se_62 => relu_15
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_82), kwargs = {})
#   %mul_353 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, 0.9622504486493761), kwargs = {})
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_353, %view_231, %arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_106,), kwargs = {})
#   %mul_360 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, %sigmoid_83), kwargs = {})
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_360, %view_234, %arg75_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_84 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_107,), kwargs = {})
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, %sigmoid_84), kwargs = {})
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_364, %view_237, %arg78_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_85 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_108,), kwargs = {})
#   %mul_368 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, %sigmoid_85), kwargs = {})
#   %convolution_109 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_368, %view_240, %arg81_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_109, [2, 3], True), kwargs = {})
#   %convolution_110 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_16, %arg82_1, %arg83_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_110,), kwargs = {})
triton_poi_fused_convolution_mean_mul_relu_silu_46 = async_compile.triton('triton_poi_fused_convolution_mean_mul_relu_silu_46', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_silu_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_relu_silu_46(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wd/cwd5x4kufq25uipbxwftmcsiek32p3ecmfxzd7b7ey3yafigmhzr.py
# Topologically Sorted Source Nodes: [avg_pool2d_4], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_4 => avg_pool2d_4
# Graph fragment:
#   %avg_pool2d_4 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_353, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
triton_poi_fused_avg_pool2d_47 = async_compile.triton('triton_poi_fused_avg_pool2d_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1327104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512) % 18
    x2 = (xindex // 9216)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x1) + (36864*x2)), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + (1024*x1) + (36864*x2)), None)
    tmp3 = tl.load(in_ptr0 + (18432 + x0 + (1024*x1) + (36864*x2)), None)
    tmp5 = tl.load(in_ptr0 + (18944 + x0 + (1024*x1) + (36864*x2)), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ro/croyv7bfg5vdeiachakia6mhfoln2omjl5rpbevuuprlnezbol4r.py
# Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69, out_108, silu_70, out_109, x_se_60, x_se_61, x_se_62, x_se_63, sigmoid_15, mul_141, out_110, mul_143, avg_pool2d_4, shortcut_6, out_111, silu_71, out_112], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.avg_pool2d, aten.add]
# Source node to ATen node mapping:
#   avg_pool2d_4 => avg_pool2d_4
#   mul_141 => mul_372
#   mul_143 => mul_374
#   out_105 => mul_353
#   out_106 => convolution_106
#   out_107 => convolution_107
#   out_108 => convolution_108
#   out_109 => convolution_109
#   out_110 => mul_373
#   out_111 => add_95
#   out_112 => mul_376
#   shortcut_6 => convolution_105
#   sigmoid_15 => sigmoid_86
#   silu_67 => mul_352, sigmoid_82
#   silu_68 => mul_360, sigmoid_83
#   silu_69 => mul_364, sigmoid_84
#   silu_70 => mul_368, sigmoid_85
#   silu_71 => mul_375, sigmoid_87
#   x_se_60 => mean_16
#   x_se_61 => convolution_110
#   x_se_62 => relu_15
#   x_se_63 => convolution_111
# Graph fragment:
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_82), kwargs = {})
#   %mul_353 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, 0.9622504486493761), kwargs = {})
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_353, %view_231, %arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_106,), kwargs = {})
#   %mul_360 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_106, %sigmoid_83), kwargs = {})
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_360, %view_234, %arg75_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_84 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_107,), kwargs = {})
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_107, %sigmoid_84), kwargs = {})
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_364, %view_237, %arg78_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_85 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_108,), kwargs = {})
#   %mul_368 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, %sigmoid_85), kwargs = {})
#   %convolution_109 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_368, %view_240, %arg81_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_109, [2, 3], True), kwargs = {})
#   %convolution_110 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_16, %arg82_1, %arg83_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_110,), kwargs = {})
#   %convolution_111 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg84_1, %arg85_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_86 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_111,), kwargs = {})
#   %mul_372 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_109, %sigmoid_86), kwargs = {})
#   %mul_373 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_372, 2.0), kwargs = {})
#   %mul_374 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_373, 0.2), kwargs = {})
#   %avg_pool2d_4 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_353, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_105 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_4, %view_228, %arg69_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_95 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_374, %convolution_105), kwargs = {})
#   %sigmoid_87 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_95,), kwargs = {})
#   %mul_375 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_95, %sigmoid_87), kwargs = {})
#   %mul_376 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_375, 0.9805806756909201), kwargs = {})
triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_48 = async_compile.triton('triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 497664)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), None)
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = 0.9805806756909201
    tmp19 = tmp17 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/k7/ck7dassyu56kmlohkk337pfs6bn5rplxsa5p6gbaieern4bppsxz.py
# Topologically Sorted Source Nodes: [silu_71, out_112, out_113, silu_72, out_114, silu_73, out_115, silu_74, out_116, x_se_64, x_se_65, x_se_66, x_se_67, sigmoid_16, mul_149, out_117, mul_151, out_118, silu_75, out_119], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   mul_149 => mul_392
#   mul_151 => mul_394
#   out_112 => mul_376
#   out_113 => convolution_112
#   out_114 => convolution_113
#   out_115 => convolution_114
#   out_116 => convolution_115
#   out_117 => mul_393
#   out_118 => add_100
#   out_119 => mul_396
#   sigmoid_16 => sigmoid_91
#   silu_71 => mul_375, sigmoid_87
#   silu_72 => mul_380, sigmoid_88
#   silu_73 => mul_384, sigmoid_89
#   silu_74 => mul_388, sigmoid_90
#   silu_75 => mul_395, sigmoid_92
#   x_se_64 => mean_17
#   x_se_65 => convolution_116
#   x_se_66 => relu_16
#   x_se_67 => convolution_117
# Graph fragment:
#   %sigmoid_87 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_95,), kwargs = {})
#   %mul_375 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_95, %sigmoid_87), kwargs = {})
#   %mul_376 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_375, 0.9805806756909201), kwargs = {})
#   %convolution_112 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_376, %view_243, %arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_88 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_112,), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_112, %sigmoid_88), kwargs = {})
#   %convolution_113 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_380, %view_246, %arg91_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_89 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_113,), kwargs = {})
#   %mul_384 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_113, %sigmoid_89), kwargs = {})
#   %convolution_114 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_384, %view_249, %arg94_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_90 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_114,), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_114, %sigmoid_90), kwargs = {})
#   %convolution_115 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_388, %view_252, %arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_115, [2, 3], True), kwargs = {})
#   %convolution_116 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_17, %arg98_1, %arg99_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_116,), kwargs = {})
#   %convolution_117 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_16, %arg100_1, %arg101_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_91 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_117,), kwargs = {})
#   %mul_392 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_115, %sigmoid_91), kwargs = {})
#   %mul_393 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_392, 2.0), kwargs = {})
#   %mul_394 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_393, 0.2), kwargs = {})
#   %add_100 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_394, %add_95), kwargs = {})
#   %sigmoid_92 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_100,), kwargs = {})
#   %mul_395 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_100, %sigmoid_92), kwargs = {})
#   %mul_396 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_395, 0.9622504486493761), kwargs = {})
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_49 = async_compile.triton('triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 497664)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.9622504486493761
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qt/cqtfenrb6idfrmp25zehg6cmcg2jqx2e6cty2mjktqe37wosqh75.py
# Topologically Sorted Source Nodes: [silu_75, out_119, out_120, silu_76, out_121, silu_77, out_122, silu_78, out_123, x_se_68, x_se_69, x_se_70, x_se_71, sigmoid_17, mul_157, out_124, mul_159, out_125, silu_79, out_126], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   mul_157 => mul_412
#   mul_159 => mul_414
#   out_119 => mul_396
#   out_120 => convolution_118
#   out_121 => convolution_119
#   out_122 => convolution_120
#   out_123 => convolution_121
#   out_124 => mul_413
#   out_125 => add_105
#   out_126 => mul_416
#   sigmoid_17 => sigmoid_96
#   silu_75 => mul_395, sigmoid_92
#   silu_76 => mul_400, sigmoid_93
#   silu_77 => mul_404, sigmoid_94
#   silu_78 => mul_408, sigmoid_95
#   silu_79 => mul_415, sigmoid_97
#   x_se_68 => mean_18
#   x_se_69 => convolution_122
#   x_se_70 => relu_17
#   x_se_71 => convolution_123
# Graph fragment:
#   %sigmoid_92 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_100,), kwargs = {})
#   %mul_395 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_100, %sigmoid_92), kwargs = {})
#   %mul_396 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_395, 0.9622504486493761), kwargs = {})
#   %convolution_118 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_396, %view_255, %arg104_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_93 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_118,), kwargs = {})
#   %mul_400 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_118, %sigmoid_93), kwargs = {})
#   %convolution_119 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_400, %view_258, %arg107_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_94 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_119,), kwargs = {})
#   %mul_404 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_119, %sigmoid_94), kwargs = {})
#   %convolution_120 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_404, %view_261, %arg110_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_95 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_120,), kwargs = {})
#   %mul_408 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_120, %sigmoid_95), kwargs = {})
#   %convolution_121 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_408, %view_264, %arg113_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_121, [2, 3], True), kwargs = {})
#   %convolution_122 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_18, %arg114_1, %arg115_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_122,), kwargs = {})
#   %convolution_123 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg116_1, %arg117_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_96 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_123,), kwargs = {})
#   %mul_412 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_121, %sigmoid_96), kwargs = {})
#   %mul_413 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_412, 2.0), kwargs = {})
#   %mul_414 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_413, 0.2), kwargs = {})
#   %add_105 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_414, %add_100), kwargs = {})
#   %sigmoid_97 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_105,), kwargs = {})
#   %mul_415 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_105, %sigmoid_97), kwargs = {})
#   %mul_416 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_415, 0.9449111825230679), kwargs = {})
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_50 = async_compile.triton('triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 497664)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.9449111825230679
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6b/c6bhazeq6t7qqfjryalh63q37g6md2zriqcmelegui3wgoiktxcd.py
# Topologically Sorted Source Nodes: [silu_79, out_126, out_127, silu_80, out_128, silu_81, out_129, silu_82, out_130, x_se_72, x_se_73, x_se_74, x_se_75, sigmoid_18, mul_165, out_131, mul_167, out_132, silu_83, out_133], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   mul_165 => mul_432
#   mul_167 => mul_434
#   out_126 => mul_416
#   out_127 => convolution_124
#   out_128 => convolution_125
#   out_129 => convolution_126
#   out_130 => convolution_127
#   out_131 => mul_433
#   out_132 => add_110
#   out_133 => mul_436
#   sigmoid_18 => sigmoid_101
#   silu_79 => mul_415, sigmoid_97
#   silu_80 => mul_420, sigmoid_98
#   silu_81 => mul_424, sigmoid_99
#   silu_82 => mul_428, sigmoid_100
#   silu_83 => mul_435, sigmoid_102
#   x_se_72 => mean_19
#   x_se_73 => convolution_128
#   x_se_74 => relu_18
#   x_se_75 => convolution_129
# Graph fragment:
#   %sigmoid_97 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_105,), kwargs = {})
#   %mul_415 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_105, %sigmoid_97), kwargs = {})
#   %mul_416 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_415, 0.9449111825230679), kwargs = {})
#   %convolution_124 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_416, %view_267, %arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_98 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_124,), kwargs = {})
#   %mul_420 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_124, %sigmoid_98), kwargs = {})
#   %convolution_125 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_420, %view_270, %arg123_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_99 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_125,), kwargs = {})
#   %mul_424 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_125, %sigmoid_99), kwargs = {})
#   %convolution_126 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_424, %view_273, %arg126_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_100 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_126,), kwargs = {})
#   %mul_428 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_126, %sigmoid_100), kwargs = {})
#   %convolution_127 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_428, %view_276, %arg129_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_127, [2, 3], True), kwargs = {})
#   %convolution_128 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_19, %arg130_1, %arg131_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_128,), kwargs = {})
#   %convolution_129 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %arg132_1, %arg133_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_101 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_129,), kwargs = {})
#   %mul_432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_127, %sigmoid_101), kwargs = {})
#   %mul_433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_432, 2.0), kwargs = {})
#   %mul_434 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_433, 0.2), kwargs = {})
#   %add_110 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_434, %add_105), kwargs = {})
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_110,), kwargs = {})
#   %mul_435 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_110, %sigmoid_102), kwargs = {})
#   %mul_436 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_435, 0.9284766908852592), kwargs = {})
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_51 = async_compile.triton('triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 497664)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.9284766908852592
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4x/c4xy4mjxtybywdxsgzffwhwhsv6x3hxfs62wqpmvrre5pfsyog64.py
# Topologically Sorted Source Nodes: [silu_83, out_133, out_134, silu_84, out_135, silu_85, out_136, silu_86, out_137, x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, mul_173, out_138, mul_175, out_139, silu_87, out_140], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   mul_173 => mul_452
#   mul_175 => mul_454
#   out_133 => mul_436
#   out_134 => convolution_130
#   out_135 => convolution_131
#   out_136 => convolution_132
#   out_137 => convolution_133
#   out_138 => mul_453
#   out_139 => add_115
#   out_140 => mul_456
#   sigmoid_19 => sigmoid_106
#   silu_83 => mul_435, sigmoid_102
#   silu_84 => mul_440, sigmoid_103
#   silu_85 => mul_444, sigmoid_104
#   silu_86 => mul_448, sigmoid_105
#   silu_87 => mul_455, sigmoid_107
#   x_se_76 => mean_20
#   x_se_77 => convolution_134
#   x_se_78 => relu_19
#   x_se_79 => convolution_135
# Graph fragment:
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_110,), kwargs = {})
#   %mul_435 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_110, %sigmoid_102), kwargs = {})
#   %mul_436 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_435, 0.9284766908852592), kwargs = {})
#   %convolution_130 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_436, %view_279, %arg136_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_103 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_130,), kwargs = {})
#   %mul_440 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_130, %sigmoid_103), kwargs = {})
#   %convolution_131 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_440, %view_282, %arg139_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_104 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_131,), kwargs = {})
#   %mul_444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_131, %sigmoid_104), kwargs = {})
#   %convolution_132 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_444, %view_285, %arg142_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_105 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_132,), kwargs = {})
#   %mul_448 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_132, %sigmoid_105), kwargs = {})
#   %convolution_133 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_448, %view_288, %arg145_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_133, [2, 3], True), kwargs = {})
#   %convolution_134 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_20, %arg146_1, %arg147_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_134,), kwargs = {})
#   %convolution_135 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg148_1, %arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_106 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_135,), kwargs = {})
#   %mul_452 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_133, %sigmoid_106), kwargs = {})
#   %mul_453 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_452, 2.0), kwargs = {})
#   %mul_454 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_453, 0.2), kwargs = {})
#   %add_115 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_454, %add_110), kwargs = {})
#   %sigmoid_107 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_115,), kwargs = {})
#   %mul_455 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_115, %sigmoid_107), kwargs = {})
#   %mul_456 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_455, 0.9128709291752768), kwargs = {})
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_52 = async_compile.triton('triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_52(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 497664)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.9128709291752768
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vh/cvh4ycgiqq6rwln2qgehorbcsvuzwugziapzylvve6ppgv42dvlz.py
# Topologically Sorted Source Nodes: [silu_87, out_140, out_141, silu_88, out_142, silu_89, out_143, silu_90, out_144, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, mul_181, out_145, mul_183, out_146, silu_91, out_147], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   mul_181 => mul_472
#   mul_183 => mul_474
#   out_140 => mul_456
#   out_141 => convolution_136
#   out_142 => convolution_137
#   out_143 => convolution_138
#   out_144 => convolution_139
#   out_145 => mul_473
#   out_146 => add_120
#   out_147 => mul_476
#   sigmoid_20 => sigmoid_111
#   silu_87 => mul_455, sigmoid_107
#   silu_88 => mul_460, sigmoid_108
#   silu_89 => mul_464, sigmoid_109
#   silu_90 => mul_468, sigmoid_110
#   silu_91 => mul_475, sigmoid_112
#   x_se_80 => mean_21
#   x_se_81 => convolution_140
#   x_se_82 => relu_20
#   x_se_83 => convolution_141
# Graph fragment:
#   %sigmoid_107 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_115,), kwargs = {})
#   %mul_455 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_115, %sigmoid_107), kwargs = {})
#   %mul_456 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_455, 0.9128709291752768), kwargs = {})
#   %convolution_136 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_456, %view_291, %arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_108 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_136,), kwargs = {})
#   %mul_460 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_136, %sigmoid_108), kwargs = {})
#   %convolution_137 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_460, %view_294, %arg155_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_109 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_137,), kwargs = {})
#   %mul_464 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_137, %sigmoid_109), kwargs = {})
#   %convolution_138 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_464, %view_297, %arg158_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_110 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_138,), kwargs = {})
#   %mul_468 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_138, %sigmoid_110), kwargs = {})
#   %convolution_139 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_468, %view_300, %arg161_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_139, [2, 3], True), kwargs = {})
#   %convolution_140 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_21, %arg162_1, %arg163_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_140,), kwargs = {})
#   %convolution_141 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %arg164_1, %arg165_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_111 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_141,), kwargs = {})
#   %mul_472 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_139, %sigmoid_111), kwargs = {})
#   %mul_473 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_472, 2.0), kwargs = {})
#   %mul_474 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_473, 0.2), kwargs = {})
#   %add_120 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_474, %add_115), kwargs = {})
#   %sigmoid_112 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_120,), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_120, %sigmoid_112), kwargs = {})
#   %mul_476 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_475, 0.8980265101338745), kwargs = {})
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_53 = async_compile.triton('triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_53(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 497664)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.8980265101338745
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tc/ctcuowjxpj3feknouztfkb2dv75fsuu3ed4qkutxpb5oqoor6wns.py
# Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149, silu_93], Original ATen: [aten.silu, aten.mul, aten.convolution]
# Source node to ATen node mapping:
#   out_147 => mul_476
#   out_148 => convolution_143
#   out_149 => convolution_144
#   silu_91 => mul_475, sigmoid_112
#   silu_92 => mul_483, sigmoid_113
#   silu_93 => mul_487, sigmoid_114
# Graph fragment:
#   %sigmoid_112 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_120,), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_120, %sigmoid_112), kwargs = {})
#   %mul_476 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_475, 0.8980265101338745), kwargs = {})
#   %convolution_143 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_476, %view_306, %arg171_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_113 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_143,), kwargs = {})
#   %mul_483 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, %sigmoid_113), kwargs = {})
#   %convolution_144 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_483, %view_309, %arg174_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_114 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_144,), kwargs = {})
#   %mul_487 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_144, %sigmoid_114), kwargs = {})
triton_poi_fused_convolution_mul_silu_54 = async_compile.triton('triton_poi_fused_convolution_mul_silu_54', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_silu_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_silu_54(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 248832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rj/crjivhb6a3xiltottfbajam5xzcjbzngzixzfhihq3qouvv3y432.py
# Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149, silu_93, out_150, silu_94, out_151, x_se_84], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
# Source node to ATen node mapping:
#   out_147 => mul_476
#   out_148 => convolution_143
#   out_149 => convolution_144
#   out_150 => convolution_145
#   out_151 => convolution_146
#   silu_91 => mul_475, sigmoid_112
#   silu_92 => mul_483, sigmoid_113
#   silu_93 => mul_487, sigmoid_114
#   silu_94 => mul_491, sigmoid_115
#   x_se_84 => mean_22
# Graph fragment:
#   %sigmoid_112 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_120,), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_120, %sigmoid_112), kwargs = {})
#   %mul_476 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_475, 0.8980265101338745), kwargs = {})
#   %convolution_143 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_476, %view_306, %arg171_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_113 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_143,), kwargs = {})
#   %mul_483 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, %sigmoid_113), kwargs = {})
#   %convolution_144 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_483, %view_309, %arg174_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_114 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_144,), kwargs = {})
#   %mul_487 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_144, %sigmoid_114), kwargs = {})
#   %convolution_145 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_487, %view_312, %arg177_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_115 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_145,), kwargs = {})
#   %mul_491 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_145, %sigmoid_115), kwargs = {})
#   %convolution_146 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_491, %view_315, %arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_146, [2, 3], True), kwargs = {})
triton_red_fused_convolution_mean_mul_silu_55 = async_compile.triton('triton_red_fused_convolution_mean_mul_silu_55', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_mul_silu_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_mean_mul_silu_55(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 81
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1536
    x1 = (xindex // 1536)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1536*r2) + (124416*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 81.0
    tmp7 = tmp4 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mg/cmga22m5cbczb5dbuyvalh56nryeyzzhv5msrkajuhiwisbajxef.py
# Topologically Sorted Source Nodes: [avg_pool2d_5], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_5 => avg_pool2d_5
# Graph fragment:
#   %avg_pool2d_5 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_476, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
triton_poi_fused_avg_pool2d_56 = async_compile.triton('triton_poi_fused_avg_pool2d_56', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_56(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 995328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 1536
    x1 = (xindex // 1536) % 9
    x2 = (xindex // 13824)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*x1) + (55296*x2)), None)
    tmp1 = tl.load(in_ptr0 + (1536 + x0 + (3072*x1) + (55296*x2)), None)
    tmp3 = tl.load(in_ptr0 + (27648 + x0 + (3072*x1) + (55296*x2)), None)
    tmp5 = tl.load(in_ptr0 + (29184 + x0 + (3072*x1) + (55296*x2)), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7x/c7xy2y3wuze3dnrfsp75uldsjqsgyt4spcx4psx55iiyfhfeovjo.py
# Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149, silu_93, out_150, silu_94, out_151, x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, mul_190, out_152, mul_192, avg_pool2d_5, shortcut_7, out_153, silu_95, out_154], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.avg_pool2d, aten.add]
# Source node to ATen node mapping:
#   avg_pool2d_5 => avg_pool2d_5
#   mul_190 => mul_495
#   mul_192 => mul_497
#   out_147 => mul_476
#   out_148 => convolution_143
#   out_149 => convolution_144
#   out_150 => convolution_145
#   out_151 => convolution_146
#   out_152 => mul_496
#   out_153 => add_126
#   out_154 => mul_499
#   shortcut_7 => convolution_142
#   sigmoid_21 => sigmoid_116
#   silu_91 => mul_475, sigmoid_112
#   silu_92 => mul_483, sigmoid_113
#   silu_93 => mul_487, sigmoid_114
#   silu_94 => mul_491, sigmoid_115
#   silu_95 => mul_498, sigmoid_117
#   x_se_84 => mean_22
#   x_se_85 => convolution_147
#   x_se_86 => relu_21
#   x_se_87 => convolution_148
# Graph fragment:
#   %sigmoid_112 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_120,), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_120, %sigmoid_112), kwargs = {})
#   %mul_476 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_475, 0.8980265101338745), kwargs = {})
#   %convolution_143 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_476, %view_306, %arg171_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_113 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_143,), kwargs = {})
#   %mul_483 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, %sigmoid_113), kwargs = {})
#   %convolution_144 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_483, %view_309, %arg174_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_114 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_144,), kwargs = {})
#   %mul_487 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_144, %sigmoid_114), kwargs = {})
#   %convolution_145 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_487, %view_312, %arg177_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_115 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_145,), kwargs = {})
#   %mul_491 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_145, %sigmoid_115), kwargs = {})
#   %convolution_146 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_491, %view_315, %arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_146, [2, 3], True), kwargs = {})
#   %convolution_147 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_22, %arg181_1, %arg182_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_147,), kwargs = {})
#   %convolution_148 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg183_1, %arg184_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_116 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_148,), kwargs = {})
#   %mul_495 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_146, %sigmoid_116), kwargs = {})
#   %mul_496 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_495, 2.0), kwargs = {})
#   %mul_497 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_496, 0.2), kwargs = {})
#   %avg_pool2d_5 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_476, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_142 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_5, %view_303, %arg168_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_126 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_497, %convolution_142), kwargs = {})
#   %sigmoid_117 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_126,), kwargs = {})
#   %mul_498 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_126, %sigmoid_117), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_498, 0.9805806756909201), kwargs = {})
triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_57 = async_compile.triton('triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_57', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_57', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_57(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 995328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 124416)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), None)
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = 0.9805806756909201
    tmp19 = tmp17 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bz/cbzoxvwsvpul5tiy3j6ad6g3su32f3onotjpnpelnbtsourzisho.py
# Topologically Sorted Source Nodes: [silu_95, out_154, out_155, silu_96, out_156, silu_97, out_157, silu_98, out_158, x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, mul_198, out_159, mul_200, out_160, silu_99, out_161], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   mul_198 => mul_515
#   mul_200 => mul_517
#   out_154 => mul_499
#   out_155 => convolution_149
#   out_156 => convolution_150
#   out_157 => convolution_151
#   out_158 => convolution_152
#   out_159 => mul_516
#   out_160 => add_131
#   out_161 => mul_519
#   sigmoid_22 => sigmoid_121
#   silu_95 => mul_498, sigmoid_117
#   silu_96 => mul_503, sigmoid_118
#   silu_97 => mul_507, sigmoid_119
#   silu_98 => mul_511, sigmoid_120
#   silu_99 => mul_518, sigmoid_122
#   x_se_88 => mean_23
#   x_se_89 => convolution_153
#   x_se_90 => relu_22
#   x_se_91 => convolution_154
# Graph fragment:
#   %sigmoid_117 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_126,), kwargs = {})
#   %mul_498 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_126, %sigmoid_117), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_498, 0.9805806756909201), kwargs = {})
#   %convolution_149 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_499, %view_318, %arg187_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_118 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_149,), kwargs = {})
#   %mul_503 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_149, %sigmoid_118), kwargs = {})
#   %convolution_150 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_503, %view_321, %arg190_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_119 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_150,), kwargs = {})
#   %mul_507 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_150, %sigmoid_119), kwargs = {})
#   %convolution_151 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_507, %view_324, %arg193_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_120 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_151,), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_151, %sigmoid_120), kwargs = {})
#   %convolution_152 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_511, %view_327, %arg196_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_23 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_152, [2, 3], True), kwargs = {})
#   %convolution_153 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_23, %arg197_1, %arg198_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_153,), kwargs = {})
#   %convolution_154 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %arg199_1, %arg200_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_121 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_154,), kwargs = {})
#   %mul_515 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_152, %sigmoid_121), kwargs = {})
#   %mul_516 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_515, 2.0), kwargs = {})
#   %mul_517 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_516, 0.2), kwargs = {})
#   %add_131 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_517, %add_126), kwargs = {})
#   %sigmoid_122 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_131,), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_131, %sigmoid_122), kwargs = {})
#   %mul_519 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_518, 0.9622504486493761), kwargs = {})
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_58 = async_compile.triton('triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_58', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 995328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 124416)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.9622504486493761
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2b/c2b4v6avt7k2eoyrwtvwhy6uuemd5wjsc7xzpjry576s7tpisp32.py
# Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164, silu_102, out_165, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, mul_206, out_166, mul_208, out_167], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   mul_206 => mul_535
#   mul_208 => mul_537
#   out_161 => mul_519
#   out_162 => convolution_155
#   out_163 => convolution_156
#   out_164 => convolution_157
#   out_165 => convolution_158
#   out_166 => mul_536
#   out_167 => add_136
#   sigmoid_23 => sigmoid_126
#   silu_100 => mul_523, sigmoid_123
#   silu_101 => mul_527, sigmoid_124
#   silu_102 => mul_531, sigmoid_125
#   silu_99 => mul_518, sigmoid_122
#   x_se_92 => mean_24
#   x_se_93 => convolution_159
#   x_se_94 => relu_23
#   x_se_95 => convolution_160
# Graph fragment:
#   %sigmoid_122 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_131,), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_131, %sigmoid_122), kwargs = {})
#   %mul_519 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_518, 0.9622504486493761), kwargs = {})
#   %convolution_155 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_519, %view_330, %arg203_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_123 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_155,), kwargs = {})
#   %mul_523 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_155, %sigmoid_123), kwargs = {})
#   %convolution_156 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_523, %view_333, %arg206_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_124 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_156,), kwargs = {})
#   %mul_527 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_156, %sigmoid_124), kwargs = {})
#   %convolution_157 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_527, %view_336, %arg209_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_125 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_157,), kwargs = {})
#   %mul_531 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_157, %sigmoid_125), kwargs = {})
#   %convolution_158 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_531, %view_339, %arg212_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_158, [2, 3], True), kwargs = {})
#   %convolution_159 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_24, %arg213_1, %arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_159,), kwargs = {})
#   %convolution_160 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg215_1, %arg216_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_126 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_160,), kwargs = {})
#   %mul_535 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_158, %sigmoid_126), kwargs = {})
#   %mul_536 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_535, 2.0), kwargs = {})
#   %mul_537 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_536, 0.2), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_537, %add_131), kwargs = {})
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_59 = async_compile.triton('triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_59', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 995328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 124416)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pl/cplyincqzi7psnxlfnpk5vdu3z6uwmuiqqkeworgdueh5z3plr5f.py
# Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164, silu_102, out_165, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, mul_206, out_166, mul_208, out_167, x_6, x_7, x_8], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   mul_206 => mul_535
#   mul_208 => mul_537
#   out_161 => mul_519
#   out_162 => convolution_155
#   out_163 => convolution_156
#   out_164 => convolution_157
#   out_165 => convolution_158
#   out_166 => mul_536
#   out_167 => add_136
#   sigmoid_23 => sigmoid_126
#   silu_100 => mul_523, sigmoid_123
#   silu_101 => mul_527, sigmoid_124
#   silu_102 => mul_531, sigmoid_125
#   silu_99 => mul_518, sigmoid_122
#   x_6 => convolution_161
#   x_7 => mul_541, sigmoid_127
#   x_8 => mean_25
#   x_se_92 => mean_24
#   x_se_93 => convolution_159
#   x_se_94 => relu_23
#   x_se_95 => convolution_160
# Graph fragment:
#   %sigmoid_122 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_131,), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_131, %sigmoid_122), kwargs = {})
#   %mul_519 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_518, 0.9622504486493761), kwargs = {})
#   %convolution_155 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_519, %view_330, %arg203_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_123 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_155,), kwargs = {})
#   %mul_523 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_155, %sigmoid_123), kwargs = {})
#   %convolution_156 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_523, %view_333, %arg206_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_124 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_156,), kwargs = {})
#   %mul_527 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_156, %sigmoid_124), kwargs = {})
#   %convolution_157 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_527, %view_336, %arg209_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6), kwargs = {})
#   %sigmoid_125 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_157,), kwargs = {})
#   %mul_531 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_157, %sigmoid_125), kwargs = {})
#   %convolution_158 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_531, %view_339, %arg212_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_158, [2, 3], True), kwargs = {})
#   %convolution_159 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_24, %arg213_1, %arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_159,), kwargs = {})
#   %convolution_160 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg215_1, %arg216_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_126 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_160,), kwargs = {})
#   %mul_535 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_158, %sigmoid_126), kwargs = {})
#   %mul_536 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_535, 2.0), kwargs = {})
#   %mul_537 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_536, 0.2), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_537, %add_131), kwargs = {})
#   %convolution_161 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_136, %view_342, %arg219_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_127 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_161,), kwargs = {})
#   %mul_541 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_161, %sigmoid_127), kwargs = {})
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_541, [-1, -2], True), kwargs = {})
triton_red_fused_add_convolution_mean_mul_relu_sigmoid_silu_60 = async_compile.triton('triton_red_fused_add_convolution_mean_mul_relu_sigmoid_silu_60', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_mul_relu_sigmoid_silu_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_mul_relu_sigmoid_silu_60(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 81
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2304
    x1 = (xindex // 2304)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2304*r2) + (186624*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.sigmoid(tmp2)
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = 81.0
    tmp9 = tmp6 / tmp8
    tl.store(out_ptr1 + (x3), tmp9, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (8, 3, 288, 288), (248832, 82944, 288, 1))
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
    assert_size_stride(arg16_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg17_1, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg19_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg20_1, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg21_1, (64, ), (1, ))
    assert_size_stride(arg22_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg23_1, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg26_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg33_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg36_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg39_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg42_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg45_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg48_1, (128, ), (1, ))
    assert_size_stride(arg49_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg52_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg55_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg56_1, (128, ), (1, ))
    assert_size_stride(arg57_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg58_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg61_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg68_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg69_1, (1536, ), (1, ))
    assert_size_stride(arg70_1, (384, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg71_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg72_1, (384, ), (1, ))
    assert_size_stride(arg73_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg74_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg77_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg78_1, (384, ), (1, ))
    assert_size_stride(arg79_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg80_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg81_1, (1536, ), (1, ))
    assert_size_stride(arg82_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg85_1, (1536, ), (1, ))
    assert_size_stride(arg86_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg87_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg90_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg91_1, (384, ), (1, ))
    assert_size_stride(arg92_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg93_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg96_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg97_1, (1536, ), (1, ))
    assert_size_stride(arg98_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg101_1, (1536, ), (1, ))
    assert_size_stride(arg102_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg103_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg104_1, (384, ), (1, ))
    assert_size_stride(arg105_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg106_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg107_1, (384, ), (1, ))
    assert_size_stride(arg108_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg109_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg112_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg113_1, (1536, ), (1, ))
    assert_size_stride(arg114_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg115_1, (384, ), (1, ))
    assert_size_stride(arg116_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg117_1, (1536, ), (1, ))
    assert_size_stride(arg118_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg119_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg120_1, (384, ), (1, ))
    assert_size_stride(arg121_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg122_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg125_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg126_1, (384, ), (1, ))
    assert_size_stride(arg127_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg128_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg129_1, (1536, ), (1, ))
    assert_size_stride(arg130_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg131_1, (384, ), (1, ))
    assert_size_stride(arg132_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg133_1, (1536, ), (1, ))
    assert_size_stride(arg134_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg135_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg136_1, (384, ), (1, ))
    assert_size_stride(arg137_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg138_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg139_1, (384, ), (1, ))
    assert_size_stride(arg140_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg141_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg142_1, (384, ), (1, ))
    assert_size_stride(arg143_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg144_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg145_1, (1536, ), (1, ))
    assert_size_stride(arg146_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg147_1, (384, ), (1, ))
    assert_size_stride(arg148_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg149_1, (1536, ), (1, ))
    assert_size_stride(arg150_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg151_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg152_1, (384, ), (1, ))
    assert_size_stride(arg153_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg154_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg155_1, (384, ), (1, ))
    assert_size_stride(arg156_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg157_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg158_1, (384, ), (1, ))
    assert_size_stride(arg159_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg160_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg161_1, (1536, ), (1, ))
    assert_size_stride(arg162_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg163_1, (384, ), (1, ))
    assert_size_stride(arg164_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg165_1, (1536, ), (1, ))
    assert_size_stride(arg166_1, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg167_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg168_1, (1536, ), (1, ))
    assert_size_stride(arg169_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg170_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg171_1, (384, ), (1, ))
    assert_size_stride(arg172_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg173_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg176_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg179_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg180_1, (1536, ), (1, ))
    assert_size_stride(arg181_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg182_1, (384, ), (1, ))
    assert_size_stride(arg183_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg184_1, (1536, ), (1, ))
    assert_size_stride(arg185_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg186_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg187_1, (384, ), (1, ))
    assert_size_stride(arg188_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg189_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg190_1, (384, ), (1, ))
    assert_size_stride(arg191_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg192_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg193_1, (384, ), (1, ))
    assert_size_stride(arg194_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg195_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg196_1, (1536, ), (1, ))
    assert_size_stride(arg197_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg198_1, (384, ), (1, ))
    assert_size_stride(arg199_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg200_1, (1536, ), (1, ))
    assert_size_stride(arg201_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg202_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg203_1, (384, ), (1, ))
    assert_size_stride(arg204_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg205_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg206_1, (384, ), (1, ))
    assert_size_stride(arg207_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg208_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg209_1, (384, ), (1, ))
    assert_size_stride(arg210_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg211_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg212_1, (1536, ), (1, ))
    assert_size_stride(arg213_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg214_1, (384, ), (1, ))
    assert_size_stride(arg215_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg216_1, (1536, ), (1, ))
    assert_size_stride(arg217_1, (2304, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg218_1, (2304, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg219_1, (2304, ), (1, ))
    assert_size_stride(arg220_1, (1000, 2304), (2304, 1))
    assert_size_stride(arg221_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 16, 1), (16, 1, 16), torch.float32)
        buf1 = empty_strided_cuda((1, 16, 1), (16, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_57], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_0.run(arg0_1, buf0, buf1, 16, 27, grid=grid(16), stream=stream0)
        buf3 = empty_strided_cuda((1, 32, 1), (32, 1, 32), torch.float32)
        buf4 = empty_strided_cuda((1, 32, 1), (32, 1, 32), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_58], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_1.run(arg4_1, buf3, buf4, 32, 144, grid=grid(32), stream=stream0)
        buf6 = empty_strided_cuda((1, 64, 1), (64, 1, 64), torch.float32)
        buf7 = empty_strided_cuda((1, 64, 1), (64, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_59], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_2.run(arg7_1, buf6, buf7, 64, 288, grid=grid(64), stream=stream0)
        buf9 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf10 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_60], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_3.run(arg10_1, buf9, buf10, 128, 576, grid=grid(128), stream=stream0)
        buf207 = empty_strided_cuda((1, 256, 128), (32768, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_61], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_4.run(arg13_1, arg14_1, buf207, 256, 128, grid=grid(256), stream=stream0)
        del arg13_1
        del arg14_1
        buf188 = empty_strided_cuda((1, 64, 128), (8192, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_62], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_5.run(arg16_1, arg17_1, buf188, 64, 128, grid=grid(64), stream=stream0)
        del arg16_1
        del arg17_1
        buf18 = empty_strided_cuda((1, 64, 1), (64, 1, 64), torch.float32)
        buf19 = empty_strided_cuda((1, 64, 1), (64, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_63], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_6.run(arg19_1, buf18, buf19, 64, 576, grid=grid(64), stream=stream0)
        buf21 = empty_strided_cuda((1, 64, 1), (64, 1, 64), torch.float32)
        buf22 = empty_strided_cuda((1, 64, 1), (64, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_64], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_6.run(arg22_1, buf21, buf22, 64, 576, grid=grid(64), stream=stream0)
        buf199 = empty_strided_cuda((1, 256, 64), (16384, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_65], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_7.run(arg25_1, arg26_1, buf199, 256, 64, grid=grid(256), stream=stream0)
        del arg25_1
        del arg26_1
        buf231 = empty_strided_cuda((1, 512, 256), (131072, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_66], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_8.run(arg32_1, arg33_1, buf231, 512, 256, grid=grid(512), stream=stream0)
        del arg32_1
        del arg33_1
        buf211 = empty_strided_cuda((1, 128, 256), (32768, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_67], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_9.run(arg35_1, arg36_1, buf211, 128, 256, grid=grid(128), stream=stream0)
        del arg35_1
        del arg36_1
        buf33 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf34 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_68], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_3.run(arg38_1, buf33, buf34, 128, 576, grid=grid(128), stream=stream0)
        buf36 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf37 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_69], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_3.run(arg41_1, buf36, buf37, 128, 576, grid=grid(128), stream=stream0)
        buf222 = empty_strided_cuda((1, 512, 128), (65536, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_70], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_10.run(arg44_1, arg45_1, buf222, 512, 128, grid=grid(512), stream=stream0)
        del arg44_1
        del arg45_1
        buf235 = empty_strided_cuda((1, 128, 512), (65536, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_71], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_11.run(arg51_1, arg52_1, buf235, 128, 512, grid=grid(128), stream=stream0)
        del arg51_1
        del arg52_1
        buf45 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf46 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_72], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_3.run(arg54_1, buf45, buf46, 128, 576, grid=grid(128), stream=stream0)
        buf48 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf49 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_73], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_3.run(arg57_1, buf48, buf49, 128, 576, grid=grid(128), stream=stream0)
        buf246 = empty_strided_cuda((1, 512, 128), (65536, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_74], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_10.run(arg60_1, arg61_1, buf246, 512, 128, grid=grid(512), stream=stream0)
        del arg60_1
        del arg61_1
        buf275 = empty_strided_cuda((1, 1536, 512), (786432, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_75], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_12.run(arg67_1, arg68_1, buf275, 1536, 512, grid=grid(1536), stream=stream0)
        del arg67_1
        del arg68_1
        buf256 = empty_strided_cuda((1, 384, 512), (196608, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_76], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_13.run(arg70_1, arg71_1, buf256, 384, 512, grid=grid(384), stream=stream0)
        del arg70_1
        del arg71_1
        buf60 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf61 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_77], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg73_1, buf60, buf61, 384, 576, grid=grid(384), stream=stream0)
        buf63 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf64 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_78], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg76_1, buf63, buf64, 384, 576, grid=grid(384), stream=stream0)
        buf267 = empty_strided_cuda((1, 1536, 384), (589824, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_79], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_15.run(arg79_1, arg80_1, buf267, 1536, 384, grid=grid(1536), stream=stream0)
        del arg79_1
        del arg80_1
        buf279 = empty_strided_cuda((1, 384, 1536), (589824, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_80], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_16.run(arg86_1, arg87_1, buf279, 384, 1536, grid=grid(384), stream=stream0)
        del arg86_1
        del arg87_1
        buf72 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf73 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_81], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg89_1, buf72, buf73, 384, 576, grid=grid(384), stream=stream0)
        buf75 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf76 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_82], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg92_1, buf75, buf76, 384, 576, grid=grid(384), stream=stream0)
        buf290 = empty_strided_cuda((1, 1536, 384), (589824, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_83], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_15.run(arg95_1, arg96_1, buf290, 1536, 384, grid=grid(1536), stream=stream0)
        del arg95_1
        del arg96_1
        buf299 = empty_strided_cuda((1, 384, 1536), (589824, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_84], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_16.run(arg102_1, arg103_1, buf299, 384, 1536, grid=grid(384), stream=stream0)
        del arg102_1
        del arg103_1
        buf84 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf85 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_85], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg105_1, buf84, buf85, 384, 576, grid=grid(384), stream=stream0)
        buf87 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf88 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_86], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg108_1, buf87, buf88, 384, 576, grid=grid(384), stream=stream0)
        buf310 = empty_strided_cuda((1, 1536, 384), (589824, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_87], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_15.run(arg111_1, arg112_1, buf310, 1536, 384, grid=grid(1536), stream=stream0)
        del arg111_1
        del arg112_1
        buf319 = empty_strided_cuda((1, 384, 1536), (589824, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_88], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_16.run(arg118_1, arg119_1, buf319, 384, 1536, grid=grid(384), stream=stream0)
        del arg118_1
        del arg119_1
        buf96 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf97 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_89], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg121_1, buf96, buf97, 384, 576, grid=grid(384), stream=stream0)
        buf99 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf100 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_90], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg124_1, buf99, buf100, 384, 576, grid=grid(384), stream=stream0)
        buf330 = empty_strided_cuda((1, 1536, 384), (589824, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_91], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_15.run(arg127_1, arg128_1, buf330, 1536, 384, grid=grid(1536), stream=stream0)
        del arg127_1
        del arg128_1
        buf339 = empty_strided_cuda((1, 384, 1536), (589824, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_92], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_16.run(arg134_1, arg135_1, buf339, 384, 1536, grid=grid(384), stream=stream0)
        del arg134_1
        del arg135_1
        buf108 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf109 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_93], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg137_1, buf108, buf109, 384, 576, grid=grid(384), stream=stream0)
        buf111 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf112 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_94], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg140_1, buf111, buf112, 384, 576, grid=grid(384), stream=stream0)
        buf350 = empty_strided_cuda((1, 1536, 384), (589824, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_95], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_15.run(arg143_1, arg144_1, buf350, 1536, 384, grid=grid(1536), stream=stream0)
        del arg143_1
        del arg144_1
        buf359 = empty_strided_cuda((1, 384, 1536), (589824, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_96], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_16.run(arg150_1, arg151_1, buf359, 384, 1536, grid=grid(384), stream=stream0)
        del arg150_1
        del arg151_1
        buf120 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf121 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_97], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg153_1, buf120, buf121, 384, 576, grid=grid(384), stream=stream0)
        buf123 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf124 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_98], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg156_1, buf123, buf124, 384, 576, grid=grid(384), stream=stream0)
        buf370 = empty_strided_cuda((1, 1536, 384), (589824, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_99], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_15.run(arg159_1, arg160_1, buf370, 1536, 384, grid=grid(1536), stream=stream0)
        del arg159_1
        del arg160_1
        buf398 = empty_strided_cuda((1, 1536, 1536), (2359296, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_100], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_17.run(arg166_1, arg167_1, buf398, 1536, 1536, grid=grid(1536), stream=stream0)
        del arg166_1
        del arg167_1
        buf379 = empty_strided_cuda((1, 384, 1536), (589824, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_101], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_16.run(arg169_1, arg170_1, buf379, 384, 1536, grid=grid(384), stream=stream0)
        del arg169_1
        del arg170_1
        buf135 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf136 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_102], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg172_1, buf135, buf136, 384, 576, grid=grid(384), stream=stream0)
        buf138 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf139 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_103], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg175_1, buf138, buf139, 384, 576, grid=grid(384), stream=stream0)
        buf390 = empty_strided_cuda((1, 1536, 384), (589824, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_104], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_15.run(arg178_1, arg179_1, buf390, 1536, 384, grid=grid(1536), stream=stream0)
        del arg178_1
        del arg179_1
        buf402 = empty_strided_cuda((1, 384, 1536), (589824, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_105], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_16.run(arg185_1, arg186_1, buf402, 384, 1536, grid=grid(384), stream=stream0)
        del arg185_1
        del arg186_1
        buf147 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf148 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_106], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg188_1, buf147, buf148, 384, 576, grid=grid(384), stream=stream0)
        buf150 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf151 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_107], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg191_1, buf150, buf151, 384, 576, grid=grid(384), stream=stream0)
        buf413 = empty_strided_cuda((1, 1536, 384), (589824, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_108], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_15.run(arg194_1, arg195_1, buf413, 1536, 384, grid=grid(1536), stream=stream0)
        del arg194_1
        del arg195_1
        buf422 = empty_strided_cuda((1, 384, 1536), (589824, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_109], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_16.run(arg201_1, arg202_1, buf422, 384, 1536, grid=grid(384), stream=stream0)
        del arg201_1
        del arg202_1
        buf159 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf160 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_110], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg204_1, buf159, buf160, 384, 576, grid=grid(384), stream=stream0)
        buf162 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        buf163 = empty_strided_cuda((1, 384, 1), (384, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_111], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_14.run(arg207_1, buf162, buf163, 384, 576, grid=grid(384), stream=stream0)
        buf433 = empty_strided_cuda((1, 1536, 384), (589824, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_112], Original ATen: [aten._native_batch_norm_legit]
        triton_per_fused__native_batch_norm_legit_15.run(arg210_1, arg211_1, buf433, 1536, 384, grid=grid(1536), stream=stream0)
        del arg210_1
        del arg211_1
        buf441 = empty_strided_cuda((1, 2304, 1536), (3538944, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_113], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_18.run(arg217_1, arg218_1, buf441, 2304, 1536, grid=grid(2304), stream=stream0)
        del arg217_1
        del arg218_1
        buf173 = empty_strided_cuda((16, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_57, input_8], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_19.run(arg0_1, buf0, buf1, arg1_1, buf173, 48, 9, grid=grid(48, 9), stream=stream0)
        del arg0_1
        del arg1_1
        del buf0
        del buf1
        buf172 = empty_strided_cuda((8, 3, 288, 288), (248832, 1, 864, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg3_1, buf172, 24, 82944, grid=grid(24, 82944), stream=stream0)
        del arg3_1
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf172, buf173, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (8, 16, 144, 144), (331776, 1, 2304, 16))
        del buf172
        del buf173
        buf175 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_21.run(buf175, arg2_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg2_1
        buf177 = empty_strided_cuda((32, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_58, input_8, input_9, input_10], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_convolution_silu_22.run(arg4_1, buf3, buf4, arg5_1, buf177, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg4_1
        del arg5_1
        del buf3
        del buf4
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10], Original ATen: [aten.convolution, aten.silu]
        buf178 = extern_kernels.convolution(buf175, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 32, 144, 144), (663552, 1, 4608, 32))
        del buf175
        del buf177
        buf179 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_23.run(buf179, arg6_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg6_1
        buf181 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_59, input_8, input_9, input_10, input_11, input_12], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_convolution_silu_24.run(arg7_1, buf6, buf7, arg8_1, buf181, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg7_1
        del arg8_1
        del buf6
        del buf7
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11, input_12], Original ATen: [aten.convolution, aten.silu]
        buf182 = extern_kernels.convolution(buf179, buf181, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 64, 144, 144), (1327104, 1, 9216, 64))
        del buf179
        buf183 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11, input_12, input_13], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_25.run(buf183, arg9_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg9_1
        buf185 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_60, input_8, input_9, input_10, input_11, input_12, input_13, input_14], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_convolution_silu_26.run(arg10_1, buf9, buf10, arg11_1, buf185, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg10_1
        del arg11_1
        del buf10
        del buf9
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11, input_12, input_13, input_14], Original ATen: [aten.convolution, aten.silu]
        buf186 = extern_kernels.convolution(buf183, buf185, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 128, 72, 72), (663552, 1, 9216, 128))
        buf187 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11, input_12, input_13, input_14, silu_55, out_84], Original ATen: [aten.convolution, aten.silu, aten.mul]
        triton_poi_fused_convolution_mul_silu_27.run(buf187, arg12_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg12_1
        # Topologically Sorted Source Nodes: [out_85], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf187, reinterpret_tensor(buf188, (64, 128, 1, 1), (128, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 64, 72, 72), (331776, 1, 4608, 64))
        del buf188
        buf190 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [out_85, silu_56], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_28.run(buf190, arg18_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg18_1
        buf192 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_63, out_85, silu_56, out_86], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_convolution_silu_29.run(arg19_1, buf18, buf19, arg20_1, buf192, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg19_1
        del arg20_1
        del buf18
        del buf19
        # Topologically Sorted Source Nodes: [out_85, silu_56, out_86], Original ATen: [aten.convolution, aten.silu]
        buf193 = extern_kernels.convolution(buf190, buf192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 64, 72, 72), (331776, 1, 4608, 64))
        del buf190
        buf194 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_28.run(buf194, arg21_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg21_1
        buf196 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_64, out_85, silu_56, out_86, silu_57, out_87], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_convolution_silu_29.run(arg22_1, buf21, buf22, arg23_1, buf196, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg22_1
        del arg23_1
        del buf21
        del buf22
        # Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87], Original ATen: [aten.convolution, aten.silu]
        buf197 = extern_kernels.convolution(buf194, buf196, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 64, 72, 72), (331776, 1, 4608, 64))
        del buf194
        del buf196
        buf198 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87, silu_58], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_28.run(buf198, arg24_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg24_1
        # Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87, silu_58, out_88], Original ATen: [aten.convolution, aten.silu]
        buf200 = extern_kernels.convolution(buf198, reinterpret_tensor(buf199, (256, 64, 1, 1), (64, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 256, 72, 72), (1327104, 1, 18432, 256))
        del buf199
        buf201 = empty_strided_cuda((8, 256, 1, 1, 41), (10496, 1, 83968, 83968, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87, silu_58, out_88, x_se_48], Original ATen: [aten.convolution, aten.silu, aten.mean]
        triton_red_fused_convolution_mean_silu_30.run(buf200, arg27_1, buf201, 83968, 127, grid=grid(83968), stream=stream0)
        buf203 = empty_strided_cuda((8, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87, silu_58, out_88, x_se_48], Original ATen: [aten.convolution, aten.silu, aten.mean]
        triton_per_fused_convolution_mean_silu_31.run(buf201, buf203, 2048, 41, grid=grid(2048), stream=stream0)
        del buf201
        # Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87, silu_58, out_88, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.silu, aten.mean]
        buf204 = extern_kernels.convolution(buf203, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg28_1
        del buf203
        buf205 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87, silu_58, out_88, x_se_48, x_se_49, x_se_50], Original ATen: [aten.convolution, aten.silu, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_silu_32.run(buf205, arg29_1, 512, grid=grid(512), stream=stream0)
        del arg29_1
        # Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87, silu_58, out_88, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.silu, aten.mean, aten.relu]
        buf206 = extern_kernels.convolution(buf205, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg30_1
        del buf205
        # Topologically Sorted Source Nodes: [shortcut_4], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf187, reinterpret_tensor(buf207, (256, 128, 1, 1), (128, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 256, 72, 72), (1327104, 1, 18432, 256))
        del buf187
        del buf207
        buf209 = buf200; del buf200  # reuse
        buf210 = reinterpret_tensor(buf183, (8, 256, 72, 72), (1327104, 1, 18432, 256), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [out_85, silu_56, out_86, silu_57, out_87, silu_58, out_88, x_se_48, x_se_49, x_se_50, x_se_51, sigmoid_12, mul_115, out_89, mul_117, shortcut_4, out_90, silu_59, out_91], Original ATen: [aten.convolution, aten.silu, aten.mean, aten.relu, aten.sigmoid, aten.mul, aten.add]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_33.run(buf209, arg27_1, buf206, arg31_1, buf208, arg15_1, buf210, 10616832, grid=grid(10616832), stream=stream0)
        del arg15_1
        del arg27_1
        del arg31_1
        del buf206
        del buf208
        del buf209
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf212 = extern_kernels.convolution(buf210, reinterpret_tensor(buf211, (128, 256, 1, 1), (256, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (8, 128, 72, 72), (663552, 1, 9216, 128))
        del buf211
        buf213 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_34.run(buf213, arg37_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg37_1
        buf215 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_68, silu_59, out_91, out_92, silu_60, out_93], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_silu_26.run(arg38_1, buf33, buf34, arg39_1, buf215, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg38_1
        del arg39_1
        del buf33
        del buf34
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf216 = extern_kernels.convolution(buf213, buf215, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf216, (8, 128, 36, 36), (165888, 1, 4608, 128))
        buf217 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_35.run(buf217, arg40_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg40_1
        buf219 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_69, silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_silu_26.run(arg41_1, buf36, buf37, arg42_1, buf219, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg41_1
        del arg42_1
        del buf36
        del buf37
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf220 = extern_kernels.convolution(buf217, buf219, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf220, (8, 128, 36, 36), (165888, 1, 4608, 128))
        del buf217
        buf221 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94, silu_62], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_35.run(buf221, arg43_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg43_1
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94, silu_62, out_95], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf223 = extern_kernels.convolution(buf221, reinterpret_tensor(buf222, (512, 128, 1, 1), (128, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 512, 36, 36), (663552, 1, 18432, 512))
        del buf221
        del buf222
        buf224 = empty_strided_cuda((8, 512, 1, 1, 11), (5632, 1, 45056, 45056, 512), torch.float32)
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94, silu_62, out_95, x_se_52], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_mul_silu_36.run(buf223, arg46_1, buf224, 45056, 118, grid=grid(45056), stream=stream0)
        buf226 = empty_strided_cuda((8, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94, silu_62, out_95, x_se_52], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_mul_silu_37.run(buf224, buf226, 4096, 11, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94, silu_62, out_95, x_se_52, x_se_53], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        buf227 = extern_kernels.convolution(buf226, arg47_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg47_1
        del buf226
        buf228 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94, silu_62, out_95, x_se_52, x_se_53, x_se_54], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_mul_relu_silu_38.run(buf228, arg48_1, 1024, grid=grid(1024), stream=stream0)
        del arg48_1
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94, silu_62, out_95, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf229 = extern_kernels.convolution(buf228, arg49_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg49_1
        del buf228
        buf230 = reinterpret_tensor(buf198, (8, 256, 36, 36), (331776, 1, 9216, 256), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [avg_pool2d_3], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_39.run(buf210, buf230, 2654208, grid=grid(2654208), stream=stream0)
        del buf210
        # Topologically Sorted Source Nodes: [avg_pool2d_3, shortcut_5], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf232 = extern_kernels.convolution(buf230, reinterpret_tensor(buf231, (512, 256, 1, 1), (256, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (8, 512, 36, 36), (663552, 1, 18432, 512))
        del buf230
        del buf231
        buf233 = buf223; del buf223  # reuse
        buf234 = reinterpret_tensor(buf213, (8, 512, 36, 36), (663552, 1, 18432, 512), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [silu_59, out_91, out_92, silu_60, out_93, silu_61, out_94, silu_62, out_95, x_se_52, x_se_53, x_se_54, x_se_55, sigmoid_13, mul_124, out_96, mul_126, avg_pool2d_3, shortcut_5, out_97, silu_63, out_98], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.avg_pool2d, aten.add]
        triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_40.run(buf233, arg46_1, buf229, arg50_1, buf232, arg34_1, buf234, 5308416, grid=grid(5308416), stream=stream0)
        del arg34_1
        del arg46_1
        del arg50_1
        del buf232
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf236 = extern_kernels.convolution(buf234, reinterpret_tensor(buf235, (128, 512, 1, 1), (512, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (8, 128, 36, 36), (165888, 1, 4608, 128))
        del buf235
        buf237 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_35.run(buf237, arg53_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg53_1
        buf239 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_72, silu_63, out_98, out_99, silu_64, out_100], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_silu_26.run(arg54_1, buf45, buf46, arg55_1, buf239, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg54_1
        del arg55_1
        del buf45
        del buf46
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64, out_100], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf240 = extern_kernels.convolution(buf237, buf239, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf240, (8, 128, 36, 36), (165888, 1, 4608, 128))
        del buf237
        buf241 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64, out_100, silu_65], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_35.run(buf241, arg56_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg56_1
        buf243 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_73, silu_63, out_98, out_99, silu_64, out_100, silu_65, out_101], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_silu_26.run(arg57_1, buf48, buf49, arg58_1, buf243, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg57_1
        del arg58_1
        del buf48
        del buf49
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64, out_100, silu_65, out_101], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf244 = extern_kernels.convolution(buf241, buf243, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf244, (8, 128, 36, 36), (165888, 1, 4608, 128))
        del buf241
        del buf243
        buf245 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64, out_100, silu_65, out_101, silu_66], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_35.run(buf245, arg59_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg59_1
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64, out_100, silu_65, out_101, silu_66, out_102], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf247 = extern_kernels.convolution(buf245, reinterpret_tensor(buf246, (512, 128, 1, 1), (128, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (8, 512, 36, 36), (663552, 1, 18432, 512))
        del buf246
        buf248 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64, out_100, silu_65, out_101, silu_66, out_102, x_se_56], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_mul_silu_36.run(buf247, arg62_1, buf248, 45056, 118, grid=grid(45056), stream=stream0)
        buf250 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64, out_100, silu_65, out_101, silu_66, out_102, x_se_56], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_mul_silu_37.run(buf248, buf250, 4096, 11, grid=grid(4096), stream=stream0)
        del buf248
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64, out_100, silu_65, out_101, silu_66, out_102, x_se_56, x_se_57], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        buf251 = extern_kernels.convolution(buf250, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg63_1
        del buf250
        buf252 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64, out_100, silu_65, out_101, silu_66, out_102, x_se_56, x_se_57, x_se_58], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_mul_relu_silu_38.run(buf252, arg64_1, 1024, grid=grid(1024), stream=stream0)
        del arg64_1
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64, out_100, silu_65, out_101, silu_66, out_102, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf253 = extern_kernels.convolution(buf252, arg65_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg65_1
        del buf252
        buf254 = buf233; del buf233  # reuse
        buf255 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [silu_63, out_98, out_99, silu_64, out_100, silu_65, out_101, silu_66, out_102, x_se_56, x_se_57, x_se_58, x_se_59, sigmoid_14, mul_132, out_103, mul_134, out_104, silu_67, out_105], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_41.run(buf254, buf247, arg62_1, buf253, arg66_1, buf255, 5308416, grid=grid(5308416), stream=stream0)
        del arg62_1
        del arg66_1
        del buf247
        del buf253
        del buf254
        # Topologically Sorted Source Nodes: [silu_67, out_105, out_106], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf257 = extern_kernels.convolution(buf255, reinterpret_tensor(buf256, (384, 512, 1, 1), (512, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (8, 384, 36, 36), (497664, 1, 13824, 384))
        del buf256
        buf258 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_42.run(buf258, arg72_1, 3981312, grid=grid(3981312), stream=stream0)
        del arg72_1
        buf260 = empty_strided_cuda((384, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_77, silu_67, out_105, out_106, silu_68, out_107], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg73_1, buf60, buf61, arg74_1, buf260, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg73_1
        del arg74_1
        del buf60
        del buf61
        # Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf261 = extern_kernels.convolution(buf258, buf260, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf261, (8, 384, 18, 18), (124416, 1, 6912, 384))
        buf262 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf262, arg75_1, 995328, grid=grid(995328), stream=stream0)
        del arg75_1
        buf264 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_78, silu_67, out_105, out_106, silu_68, out_107, silu_69, out_108], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg76_1, buf63, buf64, arg77_1, buf264, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg76_1
        del arg77_1
        del buf63
        del buf64
        # Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69, out_108], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf265 = extern_kernels.convolution(buf262, buf264, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf265, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf262
        buf266 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69, out_108, silu_70], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf266, arg78_1, 995328, grid=grid(995328), stream=stream0)
        del arg78_1
        # Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69, out_108, silu_70, out_109], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf268 = extern_kernels.convolution(buf266, reinterpret_tensor(buf267, (1536, 384, 1, 1), (384, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (8, 1536, 18, 18), (497664, 1, 27648, 1536))
        del buf266
        del buf267
        buf270 = empty_strided_cuda((8, 1536, 1, 1), (1536, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69, out_108, silu_70, out_109, x_se_60], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_mul_silu_45.run(buf268, arg81_1, buf270, 12288, 324, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69, out_108, silu_70, out_109, x_se_60, x_se_61], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        buf271 = extern_kernels.convolution(buf270, arg82_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg82_1
        del buf270
        buf272 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69, out_108, silu_70, out_109, x_se_60, x_se_61, x_se_62], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_mul_relu_silu_46.run(buf272, arg83_1, 3072, grid=grid(3072), stream=stream0)
        del arg83_1
        # Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69, out_108, silu_70, out_109, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf273 = extern_kernels.convolution(buf272, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg84_1
        del buf272
        buf274 = reinterpret_tensor(buf245, (8, 512, 18, 18), (165888, 1, 9216, 512), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [avg_pool2d_4], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_47.run(buf255, buf274, 1327104, grid=grid(1327104), stream=stream0)
        del buf255
        # Topologically Sorted Source Nodes: [avg_pool2d_4, shortcut_6], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf276 = extern_kernels.convolution(buf274, reinterpret_tensor(buf275, (1536, 512, 1, 1), (512, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 1536, 18, 18), (497664, 1, 27648, 1536))
        del buf274
        del buf275
        buf277 = buf268; del buf268  # reuse
        buf278 = reinterpret_tensor(buf258, (8, 1536, 18, 18), (497664, 1, 27648, 1536), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [silu_67, out_105, out_106, silu_68, out_107, silu_69, out_108, silu_70, out_109, x_se_60, x_se_61, x_se_62, x_se_63, sigmoid_15, mul_141, out_110, mul_143, avg_pool2d_4, shortcut_6, out_111, silu_71, out_112], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.avg_pool2d, aten.add]
        triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_48.run(buf277, arg81_1, buf273, arg85_1, buf276, arg69_1, buf278, 3981312, grid=grid(3981312), stream=stream0)
        del arg69_1
        del arg81_1
        del arg85_1
        del buf276
        # Topologically Sorted Source Nodes: [silu_71, out_112, out_113], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf280 = extern_kernels.convolution(buf278, reinterpret_tensor(buf279, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf279
        buf281 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [silu_71, out_112, out_113, silu_72], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf281, arg88_1, 995328, grid=grid(995328), stream=stream0)
        del arg88_1
        buf283 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_81, silu_71, out_112, out_113, silu_72, out_114], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg89_1, buf72, buf73, arg90_1, buf283, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg89_1
        del arg90_1
        del buf72
        del buf73
        # Topologically Sorted Source Nodes: [silu_71, out_112, out_113, silu_72, out_114], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf284 = extern_kernels.convolution(buf281, buf283, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf284, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf281
        buf285 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [silu_71, out_112, out_113, silu_72, out_114, silu_73], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf285, arg91_1, 995328, grid=grid(995328), stream=stream0)
        del arg91_1
        buf287 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_82, silu_71, out_112, out_113, silu_72, out_114, silu_73, out_115], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg92_1, buf75, buf76, arg93_1, buf287, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg92_1
        del arg93_1
        del buf75
        del buf76
        # Topologically Sorted Source Nodes: [silu_71, out_112, out_113, silu_72, out_114, silu_73, out_115], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf288 = extern_kernels.convolution(buf285, buf287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf288, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf285
        buf289 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [silu_71, out_112, out_113, silu_72, out_114, silu_73, out_115, silu_74], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf289, arg94_1, 995328, grid=grid(995328), stream=stream0)
        del arg94_1
        # Topologically Sorted Source Nodes: [silu_71, out_112, out_113, silu_72, out_114, silu_73, out_115, silu_74, out_116], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf291 = extern_kernels.convolution(buf289, reinterpret_tensor(buf290, (1536, 384, 1, 1), (384, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 1536, 18, 18), (497664, 1, 27648, 1536))
        del buf289
        del buf290
        buf293 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [silu_71, out_112, out_113, silu_72, out_114, silu_73, out_115, silu_74, out_116, x_se_64], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_mul_silu_45.run(buf291, arg97_1, buf293, 12288, 324, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [silu_71, out_112, out_113, silu_72, out_114, silu_73, out_115, silu_74, out_116, x_se_64, x_se_65], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        buf294 = extern_kernels.convolution(buf293, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg98_1
        del buf293
        buf295 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [silu_71, out_112, out_113, silu_72, out_114, silu_73, out_115, silu_74, out_116, x_se_64, x_se_65, x_se_66], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_mul_relu_silu_46.run(buf295, arg99_1, 3072, grid=grid(3072), stream=stream0)
        del arg99_1
        # Topologically Sorted Source Nodes: [silu_71, out_112, out_113, silu_72, out_114, silu_73, out_115, silu_74, out_116, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf296 = extern_kernels.convolution(buf295, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg100_1
        del buf295
        buf297 = buf277; del buf277  # reuse
        buf298 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [silu_71, out_112, out_113, silu_72, out_114, silu_73, out_115, silu_74, out_116, x_se_64, x_se_65, x_se_66, x_se_67, sigmoid_16, mul_149, out_117, mul_151, out_118, silu_75, out_119], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_49.run(buf297, buf291, arg97_1, buf296, arg101_1, buf298, 3981312, grid=grid(3981312), stream=stream0)
        del arg101_1
        del arg97_1
        del buf291
        # Topologically Sorted Source Nodes: [silu_75, out_119, out_120], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf300 = extern_kernels.convolution(buf298, reinterpret_tensor(buf299, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf299
        buf301 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [silu_75, out_119, out_120, silu_76], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf301, arg104_1, 995328, grid=grid(995328), stream=stream0)
        del arg104_1
        buf303 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_85, silu_75, out_119, out_120, silu_76, out_121], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg105_1, buf84, buf85, arg106_1, buf303, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg105_1
        del arg106_1
        del buf84
        del buf85
        # Topologically Sorted Source Nodes: [silu_75, out_119, out_120, silu_76, out_121], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf304 = extern_kernels.convolution(buf301, buf303, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf304, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf301
        buf305 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [silu_75, out_119, out_120, silu_76, out_121, silu_77], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf305, arg107_1, 995328, grid=grid(995328), stream=stream0)
        del arg107_1
        buf307 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_86, silu_75, out_119, out_120, silu_76, out_121, silu_77, out_122], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg108_1, buf87, buf88, arg109_1, buf307, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg108_1
        del arg109_1
        del buf87
        del buf88
        # Topologically Sorted Source Nodes: [silu_75, out_119, out_120, silu_76, out_121, silu_77, out_122], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf308 = extern_kernels.convolution(buf305, buf307, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf308, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf305
        buf309 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [silu_75, out_119, out_120, silu_76, out_121, silu_77, out_122, silu_78], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf309, arg110_1, 995328, grid=grid(995328), stream=stream0)
        del arg110_1
        # Topologically Sorted Source Nodes: [silu_75, out_119, out_120, silu_76, out_121, silu_77, out_122, silu_78, out_123], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf311 = extern_kernels.convolution(buf309, reinterpret_tensor(buf310, (1536, 384, 1, 1), (384, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (8, 1536, 18, 18), (497664, 1, 27648, 1536))
        del buf309
        del buf310
        buf313 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [silu_75, out_119, out_120, silu_76, out_121, silu_77, out_122, silu_78, out_123, x_se_68], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_mul_silu_45.run(buf311, arg113_1, buf313, 12288, 324, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [silu_75, out_119, out_120, silu_76, out_121, silu_77, out_122, silu_78, out_123, x_se_68, x_se_69], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        buf314 = extern_kernels.convolution(buf313, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg114_1
        del buf313
        buf315 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [silu_75, out_119, out_120, silu_76, out_121, silu_77, out_122, silu_78, out_123, x_se_68, x_se_69, x_se_70], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_mul_relu_silu_46.run(buf315, arg115_1, 3072, grid=grid(3072), stream=stream0)
        del arg115_1
        # Topologically Sorted Source Nodes: [silu_75, out_119, out_120, silu_76, out_121, silu_77, out_122, silu_78, out_123, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf316 = extern_kernels.convolution(buf315, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg116_1
        del buf315
        buf317 = buf297; del buf297  # reuse
        buf318 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [silu_75, out_119, out_120, silu_76, out_121, silu_77, out_122, silu_78, out_123, x_se_68, x_se_69, x_se_70, x_se_71, sigmoid_17, mul_157, out_124, mul_159, out_125, silu_79, out_126], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_50.run(buf317, buf311, arg113_1, buf316, arg117_1, buf318, 3981312, grid=grid(3981312), stream=stream0)
        del arg113_1
        del arg117_1
        del buf311
        # Topologically Sorted Source Nodes: [silu_79, out_126, out_127], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf320 = extern_kernels.convolution(buf318, reinterpret_tensor(buf319, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf319
        buf321 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [silu_79, out_126, out_127, silu_80], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf321, arg120_1, 995328, grid=grid(995328), stream=stream0)
        del arg120_1
        buf323 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_89, silu_79, out_126, out_127, silu_80, out_128], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg121_1, buf96, buf97, arg122_1, buf323, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg121_1
        del arg122_1
        del buf96
        del buf97
        # Topologically Sorted Source Nodes: [silu_79, out_126, out_127, silu_80, out_128], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf324 = extern_kernels.convolution(buf321, buf323, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf324, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf321
        buf325 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [silu_79, out_126, out_127, silu_80, out_128, silu_81], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf325, arg123_1, 995328, grid=grid(995328), stream=stream0)
        del arg123_1
        buf327 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_90, silu_79, out_126, out_127, silu_80, out_128, silu_81, out_129], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg124_1, buf99, buf100, arg125_1, buf327, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg124_1
        del arg125_1
        del buf100
        del buf99
        # Topologically Sorted Source Nodes: [silu_79, out_126, out_127, silu_80, out_128, silu_81, out_129], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf328 = extern_kernels.convolution(buf325, buf327, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf328, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf325
        buf329 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [silu_79, out_126, out_127, silu_80, out_128, silu_81, out_129, silu_82], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf329, arg126_1, 995328, grid=grid(995328), stream=stream0)
        del arg126_1
        # Topologically Sorted Source Nodes: [silu_79, out_126, out_127, silu_80, out_128, silu_81, out_129, silu_82, out_130], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf331 = extern_kernels.convolution(buf329, reinterpret_tensor(buf330, (1536, 384, 1, 1), (384, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (8, 1536, 18, 18), (497664, 1, 27648, 1536))
        del buf329
        del buf330
        buf333 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [silu_79, out_126, out_127, silu_80, out_128, silu_81, out_129, silu_82, out_130, x_se_72], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_mul_silu_45.run(buf331, arg129_1, buf333, 12288, 324, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [silu_79, out_126, out_127, silu_80, out_128, silu_81, out_129, silu_82, out_130, x_se_72, x_se_73], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        buf334 = extern_kernels.convolution(buf333, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg130_1
        del buf333
        buf335 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [silu_79, out_126, out_127, silu_80, out_128, silu_81, out_129, silu_82, out_130, x_se_72, x_se_73, x_se_74], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_mul_relu_silu_46.run(buf335, arg131_1, 3072, grid=grid(3072), stream=stream0)
        del arg131_1
        # Topologically Sorted Source Nodes: [silu_79, out_126, out_127, silu_80, out_128, silu_81, out_129, silu_82, out_130, x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf336 = extern_kernels.convolution(buf335, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg132_1
        del buf335
        buf337 = buf317; del buf317  # reuse
        buf338 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [silu_79, out_126, out_127, silu_80, out_128, silu_81, out_129, silu_82, out_130, x_se_72, x_se_73, x_se_74, x_se_75, sigmoid_18, mul_165, out_131, mul_167, out_132, silu_83, out_133], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_51.run(buf337, buf331, arg129_1, buf336, arg133_1, buf338, 3981312, grid=grid(3981312), stream=stream0)
        del arg129_1
        del arg133_1
        del buf331
        # Topologically Sorted Source Nodes: [silu_83, out_133, out_134], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf340 = extern_kernels.convolution(buf338, reinterpret_tensor(buf339, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf339
        buf341 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [silu_83, out_133, out_134, silu_84], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf341, arg136_1, 995328, grid=grid(995328), stream=stream0)
        del arg136_1
        buf343 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_93, silu_83, out_133, out_134, silu_84, out_135], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg137_1, buf108, buf109, arg138_1, buf343, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg137_1
        del arg138_1
        del buf108
        del buf109
        # Topologically Sorted Source Nodes: [silu_83, out_133, out_134, silu_84, out_135], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf344 = extern_kernels.convolution(buf341, buf343, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf344, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf341
        buf345 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [silu_83, out_133, out_134, silu_84, out_135, silu_85], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf345, arg139_1, 995328, grid=grid(995328), stream=stream0)
        del arg139_1
        buf347 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_94, silu_83, out_133, out_134, silu_84, out_135, silu_85, out_136], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg140_1, buf111, buf112, arg141_1, buf347, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg140_1
        del arg141_1
        del buf111
        del buf112
        # Topologically Sorted Source Nodes: [silu_83, out_133, out_134, silu_84, out_135, silu_85, out_136], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf348 = extern_kernels.convolution(buf345, buf347, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf348, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf345
        buf349 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [silu_83, out_133, out_134, silu_84, out_135, silu_85, out_136, silu_86], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf349, arg142_1, 995328, grid=grid(995328), stream=stream0)
        del arg142_1
        # Topologically Sorted Source Nodes: [silu_83, out_133, out_134, silu_84, out_135, silu_85, out_136, silu_86, out_137], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf351 = extern_kernels.convolution(buf349, reinterpret_tensor(buf350, (1536, 384, 1, 1), (384, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (8, 1536, 18, 18), (497664, 1, 27648, 1536))
        del buf349
        del buf350
        buf353 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [silu_83, out_133, out_134, silu_84, out_135, silu_85, out_136, silu_86, out_137, x_se_76], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_mul_silu_45.run(buf351, arg145_1, buf353, 12288, 324, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [silu_83, out_133, out_134, silu_84, out_135, silu_85, out_136, silu_86, out_137, x_se_76, x_se_77], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        buf354 = extern_kernels.convolution(buf353, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg146_1
        del buf353
        buf355 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [silu_83, out_133, out_134, silu_84, out_135, silu_85, out_136, silu_86, out_137, x_se_76, x_se_77, x_se_78], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_mul_relu_silu_46.run(buf355, arg147_1, 3072, grid=grid(3072), stream=stream0)
        del arg147_1
        # Topologically Sorted Source Nodes: [silu_83, out_133, out_134, silu_84, out_135, silu_85, out_136, silu_86, out_137, x_se_76, x_se_77, x_se_78, x_se_79], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf356 = extern_kernels.convolution(buf355, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg148_1
        del buf355
        buf357 = buf337; del buf337  # reuse
        buf358 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [silu_83, out_133, out_134, silu_84, out_135, silu_85, out_136, silu_86, out_137, x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, mul_173, out_138, mul_175, out_139, silu_87, out_140], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_52.run(buf357, buf351, arg145_1, buf356, arg149_1, buf358, 3981312, grid=grid(3981312), stream=stream0)
        del arg145_1
        del arg149_1
        del buf351
        # Topologically Sorted Source Nodes: [silu_87, out_140, out_141], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf360 = extern_kernels.convolution(buf358, reinterpret_tensor(buf359, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf359
        buf361 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [silu_87, out_140, out_141, silu_88], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf361, arg152_1, 995328, grid=grid(995328), stream=stream0)
        del arg152_1
        buf363 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_97, silu_87, out_140, out_141, silu_88, out_142], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg153_1, buf120, buf121, arg154_1, buf363, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg153_1
        del arg154_1
        del buf120
        del buf121
        # Topologically Sorted Source Nodes: [silu_87, out_140, out_141, silu_88, out_142], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf364 = extern_kernels.convolution(buf361, buf363, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf364, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf361
        buf365 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [silu_87, out_140, out_141, silu_88, out_142, silu_89], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf365, arg155_1, 995328, grid=grid(995328), stream=stream0)
        del arg155_1
        buf367 = buf363; del buf363  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_98, silu_87, out_140, out_141, silu_88, out_142, silu_89, out_143], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg156_1, buf123, buf124, arg157_1, buf367, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg156_1
        del arg157_1
        del buf123
        del buf124
        # Topologically Sorted Source Nodes: [silu_87, out_140, out_141, silu_88, out_142, silu_89, out_143], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf368 = extern_kernels.convolution(buf365, buf367, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf368, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf365
        buf369 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [silu_87, out_140, out_141, silu_88, out_142, silu_89, out_143, silu_90], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf369, arg158_1, 995328, grid=grid(995328), stream=stream0)
        del arg158_1
        # Topologically Sorted Source Nodes: [silu_87, out_140, out_141, silu_88, out_142, silu_89, out_143, silu_90, out_144], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf371 = extern_kernels.convolution(buf369, reinterpret_tensor(buf370, (1536, 384, 1, 1), (384, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (8, 1536, 18, 18), (497664, 1, 27648, 1536))
        del buf369
        del buf370
        buf373 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [silu_87, out_140, out_141, silu_88, out_142, silu_89, out_143, silu_90, out_144, x_se_80], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_mul_silu_45.run(buf371, arg161_1, buf373, 12288, 324, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [silu_87, out_140, out_141, silu_88, out_142, silu_89, out_143, silu_90, out_144, x_se_80, x_se_81], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        buf374 = extern_kernels.convolution(buf373, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg162_1
        del buf373
        buf375 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [silu_87, out_140, out_141, silu_88, out_142, silu_89, out_143, silu_90, out_144, x_se_80, x_se_81, x_se_82], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_mul_relu_silu_46.run(buf375, arg163_1, 3072, grid=grid(3072), stream=stream0)
        del arg163_1
        # Topologically Sorted Source Nodes: [silu_87, out_140, out_141, silu_88, out_142, silu_89, out_143, silu_90, out_144, x_se_80, x_se_81, x_se_82, x_se_83], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf376 = extern_kernels.convolution(buf375, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg164_1
        del buf375
        buf377 = buf357; del buf357  # reuse
        buf378 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [silu_87, out_140, out_141, silu_88, out_142, silu_89, out_143, silu_90, out_144, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, mul_181, out_145, mul_183, out_146, silu_91, out_147], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_53.run(buf377, buf371, arg161_1, buf376, arg165_1, buf378, 3981312, grid=grid(3981312), stream=stream0)
        del arg161_1
        del arg165_1
        del buf371
        del buf377
        # Topologically Sorted Source Nodes: [silu_91, out_147, out_148], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf380 = extern_kernels.convolution(buf378, reinterpret_tensor(buf379, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (8, 384, 18, 18), (124416, 1, 6912, 384))
        del buf379
        buf381 = buf380; del buf380  # reuse
        # Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_44.run(buf381, arg171_1, 995328, grid=grid(995328), stream=stream0)
        del arg171_1
        buf383 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_102, silu_91, out_147, out_148, silu_92, out_149], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg172_1, buf135, buf136, arg173_1, buf383, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg172_1
        del arg173_1
        del buf135
        del buf136
        # Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf384 = extern_kernels.convolution(buf381, buf383, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf384, (8, 384, 9, 9), (31104, 1, 3456, 384))
        buf385 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149, silu_93], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_54.run(buf385, arg174_1, 248832, grid=grid(248832), stream=stream0)
        del arg174_1
        buf387 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_103, silu_91, out_147, out_148, silu_92, out_149, silu_93, out_150], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg175_1, buf138, buf139, arg176_1, buf387, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg175_1
        del arg176_1
        del buf138
        del buf139
        # Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149, silu_93, out_150], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf388 = extern_kernels.convolution(buf385, buf387, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf388, (8, 384, 9, 9), (31104, 1, 3456, 384))
        del buf385
        buf389 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149, silu_93, out_150, silu_94], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_54.run(buf389, arg177_1, 248832, grid=grid(248832), stream=stream0)
        del arg177_1
        # Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149, silu_93, out_150, silu_94, out_151], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf391 = extern_kernels.convolution(buf389, reinterpret_tensor(buf390, (1536, 384, 1, 1), (384, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf391, (8, 1536, 9, 9), (124416, 1, 13824, 1536))
        del buf389
        del buf390
        buf393 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149, silu_93, out_150, silu_94, out_151, x_se_84], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_mul_silu_55.run(buf391, arg180_1, buf393, 12288, 81, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149, silu_93, out_150, silu_94, out_151, x_se_84, x_se_85], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        buf394 = extern_kernels.convolution(buf393, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg181_1
        del buf393
        buf395 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149, silu_93, out_150, silu_94, out_151, x_se_84, x_se_85, x_se_86], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_mul_relu_silu_46.run(buf395, arg182_1, 3072, grid=grid(3072), stream=stream0)
        del arg182_1
        # Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149, silu_93, out_150, silu_94, out_151, x_se_84, x_se_85, x_se_86, x_se_87], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf396 = extern_kernels.convolution(buf395, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg183_1
        del buf395
        buf397 = reinterpret_tensor(buf381, (8, 1536, 9, 9), (124416, 1, 13824, 1536), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [avg_pool2d_5], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_56.run(buf378, buf397, 995328, grid=grid(995328), stream=stream0)
        del buf378
        # Topologically Sorted Source Nodes: [avg_pool2d_5, shortcut_7], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf399 = extern_kernels.convolution(buf397, reinterpret_tensor(buf398, (1536, 1536, 1, 1), (1536, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (8, 1536, 9, 9), (124416, 1, 13824, 1536))
        del buf398
        buf400 = buf391; del buf391  # reuse
        buf401 = buf397; del buf397  # reuse
        # Topologically Sorted Source Nodes: [silu_91, out_147, out_148, silu_92, out_149, silu_93, out_150, silu_94, out_151, x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, mul_190, out_152, mul_192, avg_pool2d_5, shortcut_7, out_153, silu_95, out_154], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.avg_pool2d, aten.add]
        triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_57.run(buf400, arg180_1, buf396, arg184_1, buf399, arg168_1, buf401, 995328, grid=grid(995328), stream=stream0)
        del arg168_1
        del arg180_1
        del arg184_1
        del buf399
        # Topologically Sorted Source Nodes: [silu_95, out_154, out_155], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf403 = extern_kernels.convolution(buf401, reinterpret_tensor(buf402, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (8, 384, 9, 9), (31104, 1, 3456, 384))
        del buf402
        buf404 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [silu_95, out_154, out_155, silu_96], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_54.run(buf404, arg187_1, 248832, grid=grid(248832), stream=stream0)
        del arg187_1
        buf406 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_106, silu_95, out_154, out_155, silu_96, out_156], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg188_1, buf147, buf148, arg189_1, buf406, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg188_1
        del arg189_1
        del buf147
        del buf148
        # Topologically Sorted Source Nodes: [silu_95, out_154, out_155, silu_96, out_156], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf407 = extern_kernels.convolution(buf404, buf406, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf407, (8, 384, 9, 9), (31104, 1, 3456, 384))
        del buf404
        buf408 = buf407; del buf407  # reuse
        # Topologically Sorted Source Nodes: [silu_95, out_154, out_155, silu_96, out_156, silu_97], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_54.run(buf408, arg190_1, 248832, grid=grid(248832), stream=stream0)
        del arg190_1
        buf410 = buf406; del buf406  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_107, silu_95, out_154, out_155, silu_96, out_156, silu_97, out_157], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg191_1, buf150, buf151, arg192_1, buf410, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg191_1
        del arg192_1
        del buf150
        del buf151
        # Topologically Sorted Source Nodes: [silu_95, out_154, out_155, silu_96, out_156, silu_97, out_157], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf411 = extern_kernels.convolution(buf408, buf410, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf411, (8, 384, 9, 9), (31104, 1, 3456, 384))
        del buf408
        buf412 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [silu_95, out_154, out_155, silu_96, out_156, silu_97, out_157, silu_98], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_54.run(buf412, arg193_1, 248832, grid=grid(248832), stream=stream0)
        del arg193_1
        # Topologically Sorted Source Nodes: [silu_95, out_154, out_155, silu_96, out_156, silu_97, out_157, silu_98, out_158], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf414 = extern_kernels.convolution(buf412, reinterpret_tensor(buf413, (1536, 384, 1, 1), (384, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (8, 1536, 9, 9), (124416, 1, 13824, 1536))
        del buf412
        del buf413
        buf416 = buf396; del buf396  # reuse
        # Topologically Sorted Source Nodes: [silu_95, out_154, out_155, silu_96, out_156, silu_97, out_157, silu_98, out_158, x_se_88], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_mul_silu_55.run(buf414, arg196_1, buf416, 12288, 81, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [silu_95, out_154, out_155, silu_96, out_156, silu_97, out_157, silu_98, out_158, x_se_88, x_se_89], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        buf417 = extern_kernels.convolution(buf416, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf417, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg197_1
        del buf416
        buf418 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [silu_95, out_154, out_155, silu_96, out_156, silu_97, out_157, silu_98, out_158, x_se_88, x_se_89, x_se_90], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_mul_relu_silu_46.run(buf418, arg198_1, 3072, grid=grid(3072), stream=stream0)
        del arg198_1
        # Topologically Sorted Source Nodes: [silu_95, out_154, out_155, silu_96, out_156, silu_97, out_157, silu_98, out_158, x_se_88, x_se_89, x_se_90, x_se_91], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf419 = extern_kernels.convolution(buf418, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg199_1
        del buf418
        buf420 = buf400; del buf400  # reuse
        buf421 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [silu_95, out_154, out_155, silu_96, out_156, silu_97, out_157, silu_98, out_158, x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, mul_198, out_159, mul_200, out_160, silu_99, out_161], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_58.run(buf420, buf414, arg196_1, buf419, arg200_1, buf421, 995328, grid=grid(995328), stream=stream0)
        del arg196_1
        del arg200_1
        del buf414
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf423 = extern_kernels.convolution(buf421, reinterpret_tensor(buf422, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf423, (8, 384, 9, 9), (31104, 1, 3456, 384))
        del buf421
        del buf422
        buf424 = buf423; del buf423  # reuse
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_54.run(buf424, arg203_1, 248832, grid=grid(248832), stream=stream0)
        del arg203_1
        buf426 = buf410; del buf410  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_110, silu_99, out_161, out_162, silu_100, out_163], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg204_1, buf159, buf160, arg205_1, buf426, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg204_1
        del arg205_1
        del buf159
        del buf160
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf427 = extern_kernels.convolution(buf424, buf426, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf427, (8, 384, 9, 9), (31104, 1, 3456, 384))
        del buf424
        buf428 = buf427; del buf427  # reuse
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_54.run(buf428, arg206_1, 248832, grid=grid(248832), stream=stream0)
        del arg206_1
        buf430 = buf426; del buf426  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_111, silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164], Original ATen: [aten._native_batch_norm_legit, aten.silu, aten.mul, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_43.run(arg207_1, buf162, buf163, arg208_1, buf430, 24576, 9, grid=grid(24576, 9), stream=stream0)
        del arg207_1
        del arg208_1
        del buf162
        del buf163
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf431 = extern_kernels.convolution(buf428, buf430, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf431, (8, 384, 9, 9), (31104, 1, 3456, 384))
        del buf428
        del buf430
        buf432 = buf431; del buf431  # reuse
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164, silu_102], Original ATen: [aten.silu, aten.mul, aten.convolution]
        triton_poi_fused_convolution_mul_silu_54.run(buf432, arg209_1, 248832, grid=grid(248832), stream=stream0)
        del arg209_1
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164, silu_102, out_165], Original ATen: [aten.silu, aten.mul, aten.convolution]
        buf434 = extern_kernels.convolution(buf432, reinterpret_tensor(buf433, (1536, 384, 1, 1), (384, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (8, 1536, 9, 9), (124416, 1, 13824, 1536))
        del buf432
        del buf433
        buf436 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164, silu_102, out_165, x_se_92], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_mul_silu_55.run(buf434, arg212_1, buf436, 12288, 81, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164, silu_102, out_165, x_se_92, x_se_93], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean]
        buf437 = extern_kernels.convolution(buf436, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg213_1
        del buf436
        buf438 = buf437; del buf437  # reuse
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164, silu_102, out_165, x_se_92, x_se_93, x_se_94], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_mul_relu_silu_46.run(buf438, arg214_1, 3072, grid=grid(3072), stream=stream0)
        del arg214_1
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164, silu_102, out_165, x_se_92, x_se_93, x_se_94, x_se_95], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu]
        buf439 = extern_kernels.convolution(buf438, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg215_1
        del buf438
        buf440 = buf420; del buf420  # reuse
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164, silu_102, out_165, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, mul_206, out_166, mul_208, out_167], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_59.run(buf440, buf434, arg212_1, buf439, arg216_1, 995328, grid=grid(995328), stream=stream0)
        del arg212_1
        del arg216_1
        del buf434
        del buf439
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164, silu_102, out_165, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, mul_206, out_166, mul_208, out_167, x_6], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        buf442 = extern_kernels.convolution(buf440, reinterpret_tensor(buf441, (2304, 1536, 1, 1), (1536, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (8, 2304, 9, 9), (186624, 1, 20736, 2304))
        del buf440
        del buf441
        buf444 = reinterpret_tensor(buf181, (8, 2304, 1, 1), (2304, 1, 18432, 18432), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [silu_99, out_161, out_162, silu_100, out_163, silu_101, out_164, silu_102, out_165, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, mul_206, out_166, mul_208, out_167, x_6, x_7, x_8], Original ATen: [aten.silu, aten.mul, aten.convolution, aten.mean, aten.relu, aten.sigmoid, aten.add]
        triton_red_fused_add_convolution_mean_mul_relu_sigmoid_silu_60.run(buf442, arg219_1, buf444, 18432, 81, grid=grid(18432), stream=stream0)
        del arg219_1
        del buf442
        buf445 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg221_1, reinterpret_tensor(buf444, (8, 2304), (2304, 1), 0), reinterpret_tensor(arg220_1, (2304, 1000), (1, 2304), 0), alpha=1, beta=1, out=buf445)
        del arg220_1
        del arg221_1
        del buf444
    return (buf445, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((8, 3, 288, 288), (248832, 82944, 288, 1), device='cuda:0', dtype=torch.float32)
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
    arg16_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((2304, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((2304, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1000, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('nfnet_l0', benchmark_compiled_module)
