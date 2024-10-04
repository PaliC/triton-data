
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long
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

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_sahanp/yf/cyfccspwkop74bs7ndhmfydbnqgjsz6b7d2cj7mgipho4qyxer6u.py
# Source Nodes: [embeddings_1, embeddings_2, l__self___vit_encoder_layer_0_layernorm_before], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# embeddings_1 => cat
# embeddings_2 => add
# l__self___vit_encoder_layer_0_layernorm_before => var_mean
triton_red_fused_add_cat_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '3944FAAD9D18E987535CF029C4441E186640DD453AE5AE6831EDF736C09E0A47', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1182
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 6)
    x0 = xindex % 6
    x3 = xindex
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp17 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r2 + (128*x0)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 197, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((196*r2) + (25088*x0) + (((-1) + x1) % 196)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r2 + (128*x0)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp8, tmp13, tmp14)
        tmp16 = tl.where(tmp4, tmp7, tmp15)
        tmp18 = tmp16 + tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight, roffset == 0
        )
        tmp20_mean = tl.where(rmask & xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask & xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask & xmask, tmp20_weight_next, tmp20_weight)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
    tl.store(out_ptr1 + (x3), tmp21, xmask)
    tl.store(out_ptr2 + (x3), tmp22, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_sahanp/cm/ccmacdduz7qztz2sogxqgmxdirrulyv24gunc5xlg3pdqan3dv4j.py
# Source Nodes: [embeddings_1, embeddings_2, l__self___vit_encoder_layer_0_layernorm_before], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# embeddings_1 => cat
# embeddings_2 => add
# l__self___vit_encoder_layer_0_layernorm_before => var_mean
triton_per_fused_add_cat_native_layer_norm_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '3944FAAD9D18E987535CF029C4441E186640DD453AE5AE6831EDF736C09E0A47', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 197
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cj/ccjhsdhedhjd4ibvlicc35g7wcbikndwix5t2q23ikhj3bico3a4.py
# Source Nodes: [embeddings_1, embeddings_2, l__self___vit_encoder_layer_0_layernorm_before], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# embeddings_1 => cat
# embeddings_2 => add
# l__self___vit_encoder_layer_0_layernorm_before => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
triton_poi_fused_add_cat_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_layer_norm_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '3944FAAD9D18E987535CF029C4441E186640DD453AE5AE6831EDF736C09E0A47', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 151296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x0 = xindex % 768
    x2 = xindex
    tmp17 = tl.load(in_ptr3 + (x2), xmask)
    tmp19 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((196*x0) + (((-1) + x1) % 196)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 - tmp19
    tmp22 = 768.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp20 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2m/c2mhenjy54f6sekfnrd5ituqfbawvqvgkot6dt2ylgz5aln3tsyz.py
# Source Nodes: [embeddings_1, embeddings_2, hidden_states_2, layer_output], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# embeddings_1 => cat
# embeddings_2 => add
# hidden_states_2 => add_3
# layer_output => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
triton_red_fused_add_cat_native_layer_norm_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '3944FAAD9D18E987535CF029C4441E186640DD453AE5AE6831EDF736C09E0A47', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 197
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp24_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = x0
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1], 1, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = tl.load(in_ptr1 + (tl.broadcast_to(r1, [XBLOCK, RBLOCK])), rmask & tmp7 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
        tmp10 = tl.where(tmp7, tmp8, tmp9)
        tmp11 = tmp3 >= tmp6
        tmp12 = tl.full([1, 1], 197, tl.int64)
        tmp13 = tmp3 < tmp12
        tmp14 = tl.load(in_ptr2 + ((196*r1) + (((-1) + x0) % 196)), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr3 + (tl.broadcast_to(r1, [XBLOCK, RBLOCK])), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp14 + tmp15
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp11, tmp16, tmp17)
        tmp19 = tl.where(tmp7, tmp10, tmp18)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp2 + tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp24_mean_next, tmp24_m2_next, tmp24_weight_next = triton_helpers.welford_reduce(
            tmp23, tmp24_mean, tmp24_m2, tmp24_weight, roffset == 0
        )
        tmp24_mean = tl.where(rmask & xmask, tmp24_mean_next, tmp24_mean)
        tmp24_m2 = tl.where(rmask & xmask, tmp24_m2_next, tmp24_m2)
        tmp24_weight = tl.where(rmask & xmask, tmp24_weight_next, tmp24_weight)
        tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp22, rmask & xmask)
    tmp24_tmp, tmp25_tmp, tmp26_tmp = triton_helpers.welford(
        tmp24_mean, tmp24_m2, tmp24_weight, 1
    )
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp27 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tmp27 - tmp24
        tmp29 = 768.0
        tmp30 = tmp25 / tmp29
        tmp31 = 1e-12
        tmp32 = tmp30 + tmp31
        tmp33 = libdevice.rsqrt(tmp32)
        tmp34 = tmp28 * tmp33
        tmp36 = tmp34 * tmp35
        tmp38 = tmp36 + tmp37
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp38, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/td/ctd43ifm4crmrqw3mbnb3dn4fhcdj4opwx4kulxk2h7xx3srygpv.py
# Source Nodes: [hidden_states_4], Original ATen: [aten.gelu]
# hidden_states_4 => add_6, erf, mul_4, mul_5, mul_6
triton_poi_fused_gelu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '3944FAAD9D18E987535CF029C4441E186640DD453AE5AE6831EDF736C09E0A47', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ty/ctybtzqdv5mzq62jlyjezynhzioaupjy2prf6xz6hnhpuwhvpwny.py
# Source Nodes: [hidden_states_7, l__self___vit_encoder_layer_1_layernorm_before], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_7 => add_7
# l__self___vit_encoder_layer_1_layernorm_before => add_8, add_9, mul_7, mul_8, rsqrt_2, sub_2, var_mean_2
triton_per_fused_add_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_5', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '3944FAAD9D18E987535CF029C4441E186640DD453AE5AE6831EDF736C09E0A47', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 197
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4v/c4vcoup2f54nlzfiiyyxcedxvn4rxnwpbh6eitxkgj23blyvegb6.py
# Source Nodes: [hidden_states_10, hidden_states_7, layer_output_1], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_10 => add_10
# hidden_states_7 => add_7
# layer_output_1 => add_11, add_12, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '3944FAAD9D18E987535CF029C4441E186640DD453AE5AE6831EDF736C09E0A47', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 197
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-12
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fj/cfjj7gqhwuxejryt5wh6tb76kfm4pjyvyc26zvbc4vz35yauk4wh.py
# Source Nodes: [hidden_states_95, sequence_output], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_95 => add_84
# sequence_output => add_85, add_86, mul_84, mul_85, rsqrt_24, sub_24, var_mean_24
triton_per_fused_add_native_layer_norm_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '3944FAAD9D18E987535CF029C4441E186640DD453AE5AE6831EDF736C09E0A47', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 197
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp4, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg1_1, (1, 197, 768), (151296, 768, 1))
    assert_size_stride(arg2_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, 768), (768, 1))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, 768), (768, 1))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, 768), (768, 1))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, 768), (768, 1))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (3072, 768), (768, 1))
    assert_size_stride(arg17_1, (3072, ), (1, ))
    assert_size_stride(arg18_1, (768, 3072), (3072, 1))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, 768), (768, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, 768), (768, 1))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, 768), (768, 1))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, 768), (768, 1))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (3072, 768), (768, 1))
    assert_size_stride(arg33_1, (3072, ), (1, ))
    assert_size_stride(arg34_1, (768, 3072), (3072, 1))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, 768), (768, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, 768), (768, 1))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, 768), (768, 1))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, 768), (768, 1))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (3072, 768), (768, 1))
    assert_size_stride(arg49_1, (3072, ), (1, ))
    assert_size_stride(arg50_1, (768, 3072), (3072, 1))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, 768), (768, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, 768), (768, 1))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, 768), (768, 1))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (3072, 768), (768, 1))
    assert_size_stride(arg65_1, (3072, ), (1, ))
    assert_size_stride(arg66_1, (768, 3072), (3072, 1))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, 768), (768, 1))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, 768), (768, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, 768), (768, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, 768), (768, 1))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (3072, 768), (768, 1))
    assert_size_stride(arg81_1, (3072, ), (1, ))
    assert_size_stride(arg82_1, (768, 3072), (3072, 1))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, 768), (768, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, 768), (768, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, 768), (768, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, 768), (768, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (3072, 768), (768, 1))
    assert_size_stride(arg97_1, (3072, ), (1, ))
    assert_size_stride(arg98_1, (768, 3072), (3072, 1))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, 768), (768, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, 768), (768, 1))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, 768), (768, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, 768), (768, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (3072, 768), (768, 1))
    assert_size_stride(arg113_1, (3072, ), (1, ))
    assert_size_stride(arg114_1, (768, 3072), (3072, 1))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, 768), (768, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, 768), (768, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, 768), (768, 1))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, 768), (768, 1))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (3072, 768), (768, 1))
    assert_size_stride(arg129_1, (3072, ), (1, ))
    assert_size_stride(arg130_1, (768, 3072), (3072, 1))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, 768), (768, 1))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, 768), (768, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, 768), (768, 1))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, 768), (768, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (3072, 768), (768, 1))
    assert_size_stride(arg145_1, (3072, ), (1, ))
    assert_size_stride(arg146_1, (768, 3072), (3072, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, 768), (768, 1))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (768, 768), (768, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, 768), (768, 1))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, 768), (768, 1))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (3072, 768), (768, 1))
    assert_size_stride(arg161_1, (3072, ), (1, ))
    assert_size_stride(arg162_1, (768, 3072), (3072, 1))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, 768), (768, 1))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, 768), (768, 1))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, 768), (768, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, 768), (768, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (3072, 768), (768, 1))
    assert_size_stride(arg177_1, (3072, ), (1, ))
    assert_size_stride(arg178_1, (768, 3072), (3072, 1))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, 768), (768, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, 768), (768, 1))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, 768), (768, 1))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, 768), (768, 1))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (3072, 768), (768, 1))
    assert_size_stride(arg193_1, (3072, ), (1, ))
    assert_size_stride(arg194_1, (768, 3072), (3072, 1))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (1000, 768), (768, 1))
    assert_size_stride(arg199_1, (1000, ), (1, ))
    assert_size_stride(arg200_1, (1, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Source Nodes: [l__self___vit_embeddings_patch_embeddings_projection], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg200_1, arg2_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 768, 14, 14), (150528, 196, 14, 1))
        del arg200_1
        del arg2_1
        buf1 = empty_strided_cuda((1, 197, 1, 6), (1184, 6, 1184, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 197, 1, 6), (1184, 6, 1184, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 197, 1, 6), (1184, 6, 1184, 1), torch.float32)
        # Source Nodes: [embeddings_1, embeddings_2, l__self___vit_encoder_layer_0_layernorm_before], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_native_layer_norm_0.run(arg0_1, buf0, arg3_1, arg1_1, buf1, buf2, buf3, 1182, 128, grid=grid(1182), stream=stream0)
        buf4 = empty_strided_cuda((1, 197, 1), (197, 1, 197), torch.float32)
        buf5 = empty_strided_cuda((1, 197, 1), (197, 1, 197), torch.float32)
        # Source Nodes: [embeddings_1, embeddings_2, l__self___vit_encoder_layer_0_layernorm_before], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 197, 6, grid=grid(197), stream=stream0)
        del buf1
        del buf2
        del buf3
        buf7 = empty_strided_cuda((1, 197, 768), (151296, 768, 1), torch.float32)
        buf8 = buf7; del buf7  # reuse
        # Source Nodes: [embeddings_1, embeddings_2, l__self___vit_encoder_layer_0_layernorm_before], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_poi_fused_add_cat_native_layer_norm_2.run(buf8, arg0_1, buf0, arg3_1, arg1_1, buf4, buf5, arg4_1, arg5_1, 151296, grid=grid(151296), stream=stream0)
        del arg4_1
        del arg5_1
        del buf4
        del buf5
        buf9 = empty_strided_cuda((197, 768), (768, 1), torch.float32)
        # Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg7_1, reinterpret_tensor(buf8, (197, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf9)
        del arg6_1
        del arg7_1
        buf10 = empty_strided_cuda((197, 768), (768, 1), torch.float32)
        # Source Nodes: [l__self___vit_encoder_layer_0_attention_attention_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg9_1, reinterpret_tensor(buf8, (197, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf10)
        del arg8_1
        del arg9_1
        buf11 = empty_strided_cuda((197, 768), (768, 1), torch.float32)
        # Source Nodes: [l__self___vit_encoder_layer_0_attention_attention_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf8, (197, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf11)
        del arg10_1
        del arg11_1
        # Source Nodes: [context_layer], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf12 = aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf9, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf10, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf11, (1, 12, 197, 64), (151296, 64, 768, 1), 0), None, False)
        buf13 = buf12[0]
        del buf12
        buf17 = buf9; del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf13, (197, 768), (768, 1), 0), reinterpret_tensor(arg12_1, (768, 768), (1, 768), 0), out=buf17)
        del arg12_1
        buf18 = reinterpret_tensor(buf17, (1, 197, 768), (151296, 768, 1), 0); del buf17  # reuse
        buf22 = reinterpret_tensor(buf13, (1, 197, 768), (151296, 768, 1), 0); del buf13  # reuse
        # Source Nodes: [embeddings_1, embeddings_2, hidden_states_2, layer_output], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_red_fused_add_cat_native_layer_norm_3.run(buf18, arg13_1, arg0_1, buf0, arg3_1, arg1_1, arg14_1, arg15_1, buf22, 197, 768, grid=grid(197), stream=stream0)
        del arg0_1
        del arg13_1
        del arg14_1
        del arg15_1
        del arg1_1
        del arg3_1
        del buf0
        buf23 = empty_strided_cuda((197, 3072), (3072, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (197, 768), (768, 1), 0), reinterpret_tensor(arg16_1, (768, 3072), (1, 768), 0), out=buf23)
        del arg16_1
        buf24 = reinterpret_tensor(buf23, (1, 197, 3072), (605184, 3072, 1), 0); del buf23  # reuse
        # Source Nodes: [hidden_states_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf24, arg17_1, 605184, grid=grid(605184), stream=stream0)
        del arg17_1
        buf25 = reinterpret_tensor(buf22, (197, 768), (768, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf24, (197, 3072), (3072, 1), 0), reinterpret_tensor(arg18_1, (3072, 768), (1, 3072), 0), out=buf25)
        del arg18_1
        buf29 = reinterpret_tensor(buf11, (1, 197, 768), (151296, 768, 1), 0); del buf11  # reuse
        # Source Nodes: [hidden_states_7, l__self___vit_encoder_layer_1_layernorm_before], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf25, arg19_1, buf18, arg20_1, arg21_1, buf29, 197, 768, grid=grid(197), stream=stream0)
        del arg20_1
        del arg21_1
        buf30 = buf10; del buf10  # reuse
        # Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg23_1, reinterpret_tensor(buf29, (197, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf30)
        del arg22_1
        del arg23_1
        buf31 = reinterpret_tensor(buf8, (197, 768), (768, 1), 0); del buf8  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_1_attention_attention_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg25_1, reinterpret_tensor(buf29, (197, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf31)
        del arg24_1
        del arg25_1
        buf32 = empty_strided_cuda((197, 768), (768, 1), torch.float32)
        # Source Nodes: [l__self___vit_encoder_layer_1_attention_attention_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf29, (197, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf32)
        del arg26_1
        del arg27_1
        del buf29
        # Source Nodes: [context_layer_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf33 = aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf30, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf31, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf32, (1, 12, 197, 64), (151296, 64, 768, 1), 0), None, False)
        buf34 = buf33[0]
        del buf33
        buf38 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (197, 768), (768, 1), 0), reinterpret_tensor(arg28_1, (768, 768), (1, 768), 0), out=buf38)
        del arg28_1
        buf39 = reinterpret_tensor(buf38, (1, 197, 768), (151296, 768, 1), 0); del buf38  # reuse
        buf43 = reinterpret_tensor(buf34, (1, 197, 768), (151296, 768, 1), 0); del buf34  # reuse
        # Source Nodes: [hidden_states_10, hidden_states_7, layer_output_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf39, arg29_1, buf25, arg19_1, buf18, arg30_1, arg31_1, buf43, 197, 768, grid=grid(197), stream=stream0)
        del arg19_1
        del arg29_1
        del arg30_1
        del arg31_1
        buf44 = reinterpret_tensor(buf24, (197, 3072), (3072, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf43, (197, 768), (768, 1), 0), reinterpret_tensor(arg32_1, (768, 3072), (1, 768), 0), out=buf44)
        del arg32_1
        buf45 = reinterpret_tensor(buf44, (1, 197, 3072), (605184, 3072, 1), 0); del buf44  # reuse
        # Source Nodes: [hidden_states_12], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf45, arg33_1, 605184, grid=grid(605184), stream=stream0)
        del arg33_1
        buf46 = reinterpret_tensor(buf43, (197, 768), (768, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf45, (197, 3072), (3072, 1), 0), reinterpret_tensor(arg34_1, (3072, 768), (1, 3072), 0), out=buf46)
        del arg34_1
        buf50 = reinterpret_tensor(buf25, (1, 197, 768), (151296, 768, 1), 0); del buf25  # reuse
        # Source Nodes: [hidden_states_15, l__self___vit_encoder_layer_2_layernorm_before], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf46, arg35_1, buf39, arg36_1, arg37_1, buf50, 197, 768, grid=grid(197), stream=stream0)
        del arg36_1
        del arg37_1
        buf51 = reinterpret_tensor(buf18, (197, 768), (768, 1), 0); del buf18  # reuse
        # Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg39_1, reinterpret_tensor(buf50, (197, 768), (768, 1), 0), reinterpret_tensor(arg38_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf51)
        del arg38_1
        del arg39_1
        buf52 = buf31; del buf31  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_2_attention_attention_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg41_1, reinterpret_tensor(buf50, (197, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf52)
        del arg40_1
        del arg41_1
        buf53 = buf30; del buf30  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_2_attention_attention_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg43_1, reinterpret_tensor(buf50, (197, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf53)
        del arg42_1
        del arg43_1
        del buf50
        # Source Nodes: [context_layer_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf54 = aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf51, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf52, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf53, (1, 12, 197, 64), (151296, 64, 768, 1), 0), None, False)
        buf55 = buf54[0]
        del buf54
        buf59 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf55, (197, 768), (768, 1), 0), reinterpret_tensor(arg44_1, (768, 768), (1, 768), 0), out=buf59)
        del arg44_1
        buf60 = reinterpret_tensor(buf59, (1, 197, 768), (151296, 768, 1), 0); del buf59  # reuse
        buf64 = reinterpret_tensor(buf55, (1, 197, 768), (151296, 768, 1), 0); del buf55  # reuse
        # Source Nodes: [hidden_states_15, hidden_states_18, layer_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf60, arg45_1, buf46, arg35_1, buf39, arg46_1, arg47_1, buf64, 197, 768, grid=grid(197), stream=stream0)
        del arg35_1
        del arg45_1
        del arg46_1
        del arg47_1
        buf65 = reinterpret_tensor(buf45, (197, 3072), (3072, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (197, 768), (768, 1), 0), reinterpret_tensor(arg48_1, (768, 3072), (1, 768), 0), out=buf65)
        del arg48_1
        buf66 = reinterpret_tensor(buf65, (1, 197, 3072), (605184, 3072, 1), 0); del buf65  # reuse
        # Source Nodes: [hidden_states_20], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf66, arg49_1, 605184, grid=grid(605184), stream=stream0)
        del arg49_1
        buf67 = reinterpret_tensor(buf64, (197, 768), (768, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf66, (197, 3072), (3072, 1), 0), reinterpret_tensor(arg50_1, (3072, 768), (1, 3072), 0), out=buf67)
        del arg50_1
        buf71 = reinterpret_tensor(buf46, (1, 197, 768), (151296, 768, 1), 0); del buf46  # reuse
        # Source Nodes: [hidden_states_23, l__self___vit_encoder_layer_3_layernorm_before], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf67, arg51_1, buf60, arg52_1, arg53_1, buf71, 197, 768, grid=grid(197), stream=stream0)
        del arg52_1
        del arg53_1
        buf72 = reinterpret_tensor(buf39, (197, 768), (768, 1), 0); del buf39  # reuse
        # Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg55_1, reinterpret_tensor(buf71, (197, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf72)
        del arg54_1
        del arg55_1
        buf73 = buf52; del buf52  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_3_attention_attention_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg57_1, reinterpret_tensor(buf71, (197, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf73)
        del arg56_1
        del arg57_1
        buf74 = buf51; del buf51  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_3_attention_attention_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg59_1, reinterpret_tensor(buf71, (197, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf74)
        del arg58_1
        del arg59_1
        del buf71
        # Source Nodes: [context_layer_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf75 = aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf72, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf73, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf74, (1, 12, 197, 64), (151296, 64, 768, 1), 0), None, False)
        buf76 = buf75[0]
        del buf75
        buf80 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (197, 768), (768, 1), 0), reinterpret_tensor(arg60_1, (768, 768), (1, 768), 0), out=buf80)
        del arg60_1
        buf81 = reinterpret_tensor(buf80, (1, 197, 768), (151296, 768, 1), 0); del buf80  # reuse
        buf85 = reinterpret_tensor(buf76, (1, 197, 768), (151296, 768, 1), 0); del buf76  # reuse
        # Source Nodes: [hidden_states_23, hidden_states_26, layer_output_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf81, arg61_1, buf67, arg51_1, buf60, arg62_1, arg63_1, buf85, 197, 768, grid=grid(197), stream=stream0)
        del arg51_1
        del arg61_1
        del arg62_1
        del arg63_1
        buf86 = reinterpret_tensor(buf66, (197, 3072), (3072, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf85, (197, 768), (768, 1), 0), reinterpret_tensor(arg64_1, (768, 3072), (1, 768), 0), out=buf86)
        del arg64_1
        buf87 = reinterpret_tensor(buf86, (1, 197, 3072), (605184, 3072, 1), 0); del buf86  # reuse
        # Source Nodes: [hidden_states_28], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf87, arg65_1, 605184, grid=grid(605184), stream=stream0)
        del arg65_1
        buf88 = reinterpret_tensor(buf85, (197, 768), (768, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (197, 3072), (3072, 1), 0), reinterpret_tensor(arg66_1, (3072, 768), (1, 3072), 0), out=buf88)
        del arg66_1
        buf92 = reinterpret_tensor(buf67, (1, 197, 768), (151296, 768, 1), 0); del buf67  # reuse
        # Source Nodes: [hidden_states_31, l__self___vit_encoder_layer_4_layernorm_before], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf88, arg67_1, buf81, arg68_1, arg69_1, buf92, 197, 768, grid=grid(197), stream=stream0)
        del arg68_1
        del arg69_1
        buf93 = reinterpret_tensor(buf60, (197, 768), (768, 1), 0); del buf60  # reuse
        # Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg71_1, reinterpret_tensor(buf92, (197, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf93)
        del arg70_1
        del arg71_1
        buf94 = buf73; del buf73  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_4_attention_attention_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg73_1, reinterpret_tensor(buf92, (197, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf94)
        del arg72_1
        del arg73_1
        buf95 = buf72; del buf72  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_4_attention_attention_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg75_1, reinterpret_tensor(buf92, (197, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf95)
        del arg74_1
        del arg75_1
        del buf92
        # Source Nodes: [context_layer_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf96 = aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf93, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf94, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf95, (1, 12, 197, 64), (151296, 64, 768, 1), 0), None, False)
        buf97 = buf96[0]
        del buf96
        buf101 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (197, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 768), (1, 768), 0), out=buf101)
        del arg76_1
        buf102 = reinterpret_tensor(buf101, (1, 197, 768), (151296, 768, 1), 0); del buf101  # reuse
        buf106 = reinterpret_tensor(buf97, (1, 197, 768), (151296, 768, 1), 0); del buf97  # reuse
        # Source Nodes: [hidden_states_31, hidden_states_34, layer_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf102, arg77_1, buf88, arg67_1, buf81, arg78_1, arg79_1, buf106, 197, 768, grid=grid(197), stream=stream0)
        del arg67_1
        del arg77_1
        del arg78_1
        del arg79_1
        buf107 = reinterpret_tensor(buf87, (197, 3072), (3072, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (197, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 3072), (1, 768), 0), out=buf107)
        del arg80_1
        buf108 = reinterpret_tensor(buf107, (1, 197, 3072), (605184, 3072, 1), 0); del buf107  # reuse
        # Source Nodes: [hidden_states_36], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf108, arg81_1, 605184, grid=grid(605184), stream=stream0)
        del arg81_1
        buf109 = reinterpret_tensor(buf106, (197, 768), (768, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf108, (197, 3072), (3072, 1), 0), reinterpret_tensor(arg82_1, (3072, 768), (1, 3072), 0), out=buf109)
        del arg82_1
        buf113 = reinterpret_tensor(buf88, (1, 197, 768), (151296, 768, 1), 0); del buf88  # reuse
        # Source Nodes: [hidden_states_39, l__self___vit_encoder_layer_5_layernorm_before], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf109, arg83_1, buf102, arg84_1, arg85_1, buf113, 197, 768, grid=grid(197), stream=stream0)
        del arg84_1
        del arg85_1
        buf114 = reinterpret_tensor(buf81, (197, 768), (768, 1), 0); del buf81  # reuse
        # Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg87_1, reinterpret_tensor(buf113, (197, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf114)
        del arg86_1
        del arg87_1
        buf115 = buf94; del buf94  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_5_attention_attention_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg89_1, reinterpret_tensor(buf113, (197, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf115)
        del arg88_1
        del arg89_1
        buf116 = buf93; del buf93  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_5_attention_attention_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg91_1, reinterpret_tensor(buf113, (197, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf116)
        del arg90_1
        del arg91_1
        del buf113
        # Source Nodes: [context_layer_15], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf117 = aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf114, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf115, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf116, (1, 12, 197, 64), (151296, 64, 768, 1), 0), None, False)
        buf118 = buf117[0]
        del buf117
        buf122 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (197, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 768), (1, 768), 0), out=buf122)
        del arg92_1
        buf123 = reinterpret_tensor(buf122, (1, 197, 768), (151296, 768, 1), 0); del buf122  # reuse
        buf127 = reinterpret_tensor(buf118, (1, 197, 768), (151296, 768, 1), 0); del buf118  # reuse
        # Source Nodes: [hidden_states_39, hidden_states_42, layer_output_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf123, arg93_1, buf109, arg83_1, buf102, arg94_1, arg95_1, buf127, 197, 768, grid=grid(197), stream=stream0)
        del arg83_1
        del arg93_1
        del arg94_1
        del arg95_1
        buf128 = reinterpret_tensor(buf108, (197, 3072), (3072, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (197, 768), (768, 1), 0), reinterpret_tensor(arg96_1, (768, 3072), (1, 768), 0), out=buf128)
        del arg96_1
        buf129 = reinterpret_tensor(buf128, (1, 197, 3072), (605184, 3072, 1), 0); del buf128  # reuse
        # Source Nodes: [hidden_states_44], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf129, arg97_1, 605184, grid=grid(605184), stream=stream0)
        del arg97_1
        buf130 = reinterpret_tensor(buf127, (197, 768), (768, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (197, 3072), (3072, 1), 0), reinterpret_tensor(arg98_1, (3072, 768), (1, 3072), 0), out=buf130)
        del arg98_1
        buf134 = reinterpret_tensor(buf109, (1, 197, 768), (151296, 768, 1), 0); del buf109  # reuse
        # Source Nodes: [hidden_states_47, l__self___vit_encoder_layer_6_layernorm_before], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf130, arg99_1, buf123, arg100_1, arg101_1, buf134, 197, 768, grid=grid(197), stream=stream0)
        del arg100_1
        del arg101_1
        buf135 = reinterpret_tensor(buf102, (197, 768), (768, 1), 0); del buf102  # reuse
        # Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg103_1, reinterpret_tensor(buf134, (197, 768), (768, 1), 0), reinterpret_tensor(arg102_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf135)
        del arg102_1
        del arg103_1
        buf136 = buf115; del buf115  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_6_attention_attention_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg105_1, reinterpret_tensor(buf134, (197, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf136)
        del arg104_1
        del arg105_1
        buf137 = buf114; del buf114  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_6_attention_attention_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg107_1, reinterpret_tensor(buf134, (197, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf137)
        del arg106_1
        del arg107_1
        del buf134
        # Source Nodes: [context_layer_18], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf138 = aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf135, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf136, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf137, (1, 12, 197, 64), (151296, 64, 768, 1), 0), None, False)
        buf139 = buf138[0]
        del buf138
        buf143 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf139, (197, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 768), (1, 768), 0), out=buf143)
        del arg108_1
        buf144 = reinterpret_tensor(buf143, (1, 197, 768), (151296, 768, 1), 0); del buf143  # reuse
        buf148 = reinterpret_tensor(buf139, (1, 197, 768), (151296, 768, 1), 0); del buf139  # reuse
        # Source Nodes: [hidden_states_47, hidden_states_50, layer_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf144, arg109_1, buf130, arg99_1, buf123, arg110_1, arg111_1, buf148, 197, 768, grid=grid(197), stream=stream0)
        del arg109_1
        del arg110_1
        del arg111_1
        del arg99_1
        buf149 = reinterpret_tensor(buf129, (197, 3072), (3072, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (197, 768), (768, 1), 0), reinterpret_tensor(arg112_1, (768, 3072), (1, 768), 0), out=buf149)
        del arg112_1
        buf150 = reinterpret_tensor(buf149, (1, 197, 3072), (605184, 3072, 1), 0); del buf149  # reuse
        # Source Nodes: [hidden_states_52], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf150, arg113_1, 605184, grid=grid(605184), stream=stream0)
        del arg113_1
        buf151 = reinterpret_tensor(buf148, (197, 768), (768, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf150, (197, 3072), (3072, 1), 0), reinterpret_tensor(arg114_1, (3072, 768), (1, 3072), 0), out=buf151)
        del arg114_1
        buf155 = reinterpret_tensor(buf130, (1, 197, 768), (151296, 768, 1), 0); del buf130  # reuse
        # Source Nodes: [hidden_states_55, l__self___vit_encoder_layer_7_layernorm_before], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf151, arg115_1, buf144, arg116_1, arg117_1, buf155, 197, 768, grid=grid(197), stream=stream0)
        del arg116_1
        del arg117_1
        buf156 = reinterpret_tensor(buf123, (197, 768), (768, 1), 0); del buf123  # reuse
        # Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg119_1, reinterpret_tensor(buf155, (197, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf156)
        del arg118_1
        del arg119_1
        buf157 = buf136; del buf136  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_7_attention_attention_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg121_1, reinterpret_tensor(buf155, (197, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf157)
        del arg120_1
        del arg121_1
        buf158 = buf135; del buf135  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_7_attention_attention_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg123_1, reinterpret_tensor(buf155, (197, 768), (768, 1), 0), reinterpret_tensor(arg122_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf158)
        del arg122_1
        del arg123_1
        del buf155
        # Source Nodes: [context_layer_21], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf159 = aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf156, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf157, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf158, (1, 12, 197, 64), (151296, 64, 768, 1), 0), None, False)
        buf160 = buf159[0]
        del buf159
        buf164 = buf158; del buf158  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (197, 768), (768, 1), 0), reinterpret_tensor(arg124_1, (768, 768), (1, 768), 0), out=buf164)
        del arg124_1
        buf165 = reinterpret_tensor(buf164, (1, 197, 768), (151296, 768, 1), 0); del buf164  # reuse
        buf169 = reinterpret_tensor(buf160, (1, 197, 768), (151296, 768, 1), 0); del buf160  # reuse
        # Source Nodes: [hidden_states_55, hidden_states_58, layer_output_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf165, arg125_1, buf151, arg115_1, buf144, arg126_1, arg127_1, buf169, 197, 768, grid=grid(197), stream=stream0)
        del arg115_1
        del arg125_1
        del arg126_1
        del arg127_1
        buf170 = reinterpret_tensor(buf150, (197, 3072), (3072, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (197, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 3072), (1, 768), 0), out=buf170)
        del arg128_1
        buf171 = reinterpret_tensor(buf170, (1, 197, 3072), (605184, 3072, 1), 0); del buf170  # reuse
        # Source Nodes: [hidden_states_60], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf171, arg129_1, 605184, grid=grid(605184), stream=stream0)
        del arg129_1
        buf172 = reinterpret_tensor(buf169, (197, 768), (768, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (197, 3072), (3072, 1), 0), reinterpret_tensor(arg130_1, (3072, 768), (1, 3072), 0), out=buf172)
        del arg130_1
        buf176 = reinterpret_tensor(buf151, (1, 197, 768), (151296, 768, 1), 0); del buf151  # reuse
        # Source Nodes: [hidden_states_63, l__self___vit_encoder_layer_8_layernorm_before], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf172, arg131_1, buf165, arg132_1, arg133_1, buf176, 197, 768, grid=grid(197), stream=stream0)
        del arg132_1
        del arg133_1
        buf177 = reinterpret_tensor(buf144, (197, 768), (768, 1), 0); del buf144  # reuse
        # Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg135_1, reinterpret_tensor(buf176, (197, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf177)
        del arg134_1
        del arg135_1
        buf178 = buf157; del buf157  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_8_attention_attention_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg137_1, reinterpret_tensor(buf176, (197, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf178)
        del arg136_1
        del arg137_1
        buf179 = buf156; del buf156  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_8_attention_attention_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg139_1, reinterpret_tensor(buf176, (197, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf179)
        del arg138_1
        del arg139_1
        del buf176
        # Source Nodes: [context_layer_24], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf180 = aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf177, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf178, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf179, (1, 12, 197, 64), (151296, 64, 768, 1), 0), None, False)
        buf181 = buf180[0]
        del buf180
        buf185 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf181, (197, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0), out=buf185)
        del arg140_1
        buf186 = reinterpret_tensor(buf185, (1, 197, 768), (151296, 768, 1), 0); del buf185  # reuse
        buf190 = reinterpret_tensor(buf181, (1, 197, 768), (151296, 768, 1), 0); del buf181  # reuse
        # Source Nodes: [hidden_states_63, hidden_states_66, layer_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf186, arg141_1, buf172, arg131_1, buf165, arg142_1, arg143_1, buf190, 197, 768, grid=grid(197), stream=stream0)
        del arg131_1
        del arg141_1
        del arg142_1
        del arg143_1
        buf191 = reinterpret_tensor(buf171, (197, 3072), (3072, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (197, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 3072), (1, 768), 0), out=buf191)
        del arg144_1
        buf192 = reinterpret_tensor(buf191, (1, 197, 3072), (605184, 3072, 1), 0); del buf191  # reuse
        # Source Nodes: [hidden_states_68], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf192, arg145_1, 605184, grid=grid(605184), stream=stream0)
        del arg145_1
        buf193 = reinterpret_tensor(buf190, (197, 768), (768, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf192, (197, 3072), (3072, 1), 0), reinterpret_tensor(arg146_1, (3072, 768), (1, 3072), 0), out=buf193)
        del arg146_1
        buf197 = reinterpret_tensor(buf172, (1, 197, 768), (151296, 768, 1), 0); del buf172  # reuse
        # Source Nodes: [hidden_states_71, l__self___vit_encoder_layer_9_layernorm_before], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf193, arg147_1, buf186, arg148_1, arg149_1, buf197, 197, 768, grid=grid(197), stream=stream0)
        del arg148_1
        del arg149_1
        buf198 = reinterpret_tensor(buf165, (197, 768), (768, 1), 0); del buf165  # reuse
        # Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg151_1, reinterpret_tensor(buf197, (197, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf198)
        del arg150_1
        del arg151_1
        buf199 = buf178; del buf178  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_9_attention_attention_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg153_1, reinterpret_tensor(buf197, (197, 768), (768, 1), 0), reinterpret_tensor(arg152_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf199)
        del arg152_1
        del arg153_1
        buf200 = buf177; del buf177  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_9_attention_attention_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg155_1, reinterpret_tensor(buf197, (197, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf200)
        del arg154_1
        del arg155_1
        del buf197
        # Source Nodes: [context_layer_27], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf201 = aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf198, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf199, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf200, (1, 12, 197, 64), (151296, 64, 768, 1), 0), None, False)
        buf202 = buf201[0]
        del buf201
        buf206 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf202, (197, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 768), (1, 768), 0), out=buf206)
        del arg156_1
        buf207 = reinterpret_tensor(buf206, (1, 197, 768), (151296, 768, 1), 0); del buf206  # reuse
        buf211 = reinterpret_tensor(buf202, (1, 197, 768), (151296, 768, 1), 0); del buf202  # reuse
        # Source Nodes: [hidden_states_71, hidden_states_74, layer_output_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf207, arg157_1, buf193, arg147_1, buf186, arg158_1, arg159_1, buf211, 197, 768, grid=grid(197), stream=stream0)
        del arg147_1
        del arg157_1
        del arg158_1
        del arg159_1
        buf212 = reinterpret_tensor(buf192, (197, 3072), (3072, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (197, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 3072), (1, 768), 0), out=buf212)
        del arg160_1
        buf213 = reinterpret_tensor(buf212, (1, 197, 3072), (605184, 3072, 1), 0); del buf212  # reuse
        # Source Nodes: [hidden_states_76], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf213, arg161_1, 605184, grid=grid(605184), stream=stream0)
        del arg161_1
        buf214 = reinterpret_tensor(buf211, (197, 768), (768, 1), 0); del buf211  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf213, (197, 3072), (3072, 1), 0), reinterpret_tensor(arg162_1, (3072, 768), (1, 3072), 0), out=buf214)
        del arg162_1
        buf218 = reinterpret_tensor(buf193, (1, 197, 768), (151296, 768, 1), 0); del buf193  # reuse
        # Source Nodes: [hidden_states_79, l__self___vit_encoder_layer_10_layernorm_before], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf214, arg163_1, buf207, arg164_1, arg165_1, buf218, 197, 768, grid=grid(197), stream=stream0)
        del arg164_1
        del arg165_1
        buf219 = reinterpret_tensor(buf186, (197, 768), (768, 1), 0); del buf186  # reuse
        # Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg167_1, reinterpret_tensor(buf218, (197, 768), (768, 1), 0), reinterpret_tensor(arg166_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf219)
        del arg166_1
        del arg167_1
        buf220 = buf199; del buf199  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_10_attention_attention_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg169_1, reinterpret_tensor(buf218, (197, 768), (768, 1), 0), reinterpret_tensor(arg168_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf220)
        del arg168_1
        del arg169_1
        buf221 = buf198; del buf198  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_10_attention_attention_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg171_1, reinterpret_tensor(buf218, (197, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf221)
        del arg170_1
        del arg171_1
        del buf218
        # Source Nodes: [context_layer_30], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf222 = aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf219, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf220, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf221, (1, 12, 197, 64), (151296, 64, 768, 1), 0), None, False)
        buf223 = buf222[0]
        del buf222
        buf227 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (197, 768), (768, 1), 0), reinterpret_tensor(arg172_1, (768, 768), (1, 768), 0), out=buf227)
        del arg172_1
        buf228 = reinterpret_tensor(buf227, (1, 197, 768), (151296, 768, 1), 0); del buf227  # reuse
        buf232 = reinterpret_tensor(buf223, (1, 197, 768), (151296, 768, 1), 0); del buf223  # reuse
        # Source Nodes: [hidden_states_79, hidden_states_82, layer_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf228, arg173_1, buf214, arg163_1, buf207, arg174_1, arg175_1, buf232, 197, 768, grid=grid(197), stream=stream0)
        del arg163_1
        del arg173_1
        del arg174_1
        del arg175_1
        buf233 = reinterpret_tensor(buf213, (197, 3072), (3072, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (197, 768), (768, 1), 0), reinterpret_tensor(arg176_1, (768, 3072), (1, 768), 0), out=buf233)
        del arg176_1
        buf234 = reinterpret_tensor(buf233, (1, 197, 3072), (605184, 3072, 1), 0); del buf233  # reuse
        # Source Nodes: [hidden_states_84], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf234, arg177_1, 605184, grid=grid(605184), stream=stream0)
        del arg177_1
        buf235 = reinterpret_tensor(buf232, (197, 768), (768, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf234, (197, 3072), (3072, 1), 0), reinterpret_tensor(arg178_1, (3072, 768), (1, 3072), 0), out=buf235)
        del arg178_1
        buf239 = reinterpret_tensor(buf214, (1, 197, 768), (151296, 768, 1), 0); del buf214  # reuse
        # Source Nodes: [hidden_states_87, l__self___vit_encoder_layer_11_layernorm_before], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf235, arg179_1, buf228, arg180_1, arg181_1, buf239, 197, 768, grid=grid(197), stream=stream0)
        del arg180_1
        del arg181_1
        buf240 = reinterpret_tensor(buf207, (197, 768), (768, 1), 0); del buf207  # reuse
        # Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg183_1, reinterpret_tensor(buf239, (197, 768), (768, 1), 0), reinterpret_tensor(arg182_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf240)
        del arg182_1
        del arg183_1
        buf241 = buf220; del buf220  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_11_attention_attention_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg185_1, reinterpret_tensor(buf239, (197, 768), (768, 1), 0), reinterpret_tensor(arg184_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf241)
        del arg184_1
        del arg185_1
        buf242 = buf219; del buf219  # reuse
        # Source Nodes: [l__self___vit_encoder_layer_11_attention_attention_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg187_1, reinterpret_tensor(buf239, (197, 768), (768, 1), 0), reinterpret_tensor(arg186_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf242)
        del arg186_1
        del arg187_1
        del buf239
        # Source Nodes: [context_layer_33], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf243 = aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf240, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf241, (1, 12, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf242, (1, 12, 197, 64), (151296, 64, 768, 1), 0), None, False)
        del buf240
        del buf241
        buf244 = buf243[0]
        del buf243
        buf248 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (197, 768), (768, 1), 0), reinterpret_tensor(arg188_1, (768, 768), (1, 768), 0), out=buf248)
        del arg188_1
        buf249 = reinterpret_tensor(buf248, (1, 197, 768), (151296, 768, 1), 0); del buf248  # reuse
        buf253 = reinterpret_tensor(buf244, (1, 197, 768), (151296, 768, 1), 0); del buf244  # reuse
        # Source Nodes: [hidden_states_87, hidden_states_90, layer_output_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf249, arg189_1, buf235, arg179_1, buf228, arg190_1, arg191_1, buf253, 197, 768, grid=grid(197), stream=stream0)
        del arg179_1
        del arg189_1
        del arg190_1
        del arg191_1
        del buf228
        buf254 = reinterpret_tensor(buf234, (197, 3072), (3072, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (197, 768), (768, 1), 0), reinterpret_tensor(arg192_1, (768, 3072), (1, 768), 0), out=buf254)
        del arg192_1
        buf255 = reinterpret_tensor(buf254, (1, 197, 3072), (605184, 3072, 1), 0); del buf254  # reuse
        # Source Nodes: [hidden_states_92], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf255, arg193_1, 605184, grid=grid(605184), stream=stream0)
        del arg193_1
        buf256 = reinterpret_tensor(buf253, (197, 768), (768, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf255, (197, 3072), (3072, 1), 0), reinterpret_tensor(arg194_1, (3072, 768), (1, 3072), 0), out=buf256)
        del arg194_1
        del buf255
        buf257 = reinterpret_tensor(buf256, (1, 197, 768), (151296, 768, 1), 0); del buf256  # reuse
        buf261 = reinterpret_tensor(buf235, (1, 197, 768), (151296, 768, 1), 0); del buf235  # reuse
        # Source Nodes: [hidden_states_95, sequence_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf257, arg195_1, buf249, arg196_1, arg197_1, buf261, 197, 768, grid=grid(197), stream=stream0)
        del arg195_1
        del arg196_1
        del arg197_1
        del buf249
        buf262 = empty_strided_cuda((1, 1000), (1000, 1), torch.float32)
        # Source Nodes: [logits], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg199_1, reinterpret_tensor(buf261, (1, 768), (0, 1), 0), reinterpret_tensor(arg198_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf262)
        del arg198_1
        del arg199_1
        del buf261
    return (buf262, buf257, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
