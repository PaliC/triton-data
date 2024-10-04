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


# kernel path: /tmp/torchinductor_sahanp/55/c55nv2avbx5sapcwcybtrgwcn5idrgpjql7vkd4ps3qj7pf3monw.py
# Topologically Sorted Source Nodes: [inputs_embeds, position_embeds, add, hidden_states], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add
#   hidden_states => add_2, add_3, mul, mul_1, rsqrt, sub, var_mean
#   inputs_embeds => embedding
#   position_embeds => embedding_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %unsqueeze), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg3_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg4_1), kwargs = {})
triton_red_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x0 = xindex % 512
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 50257, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert((0 <= tmp4) & (tmp4 < 50257), "index out of bounds: 0 <= tmp4 < 50257")
        tmp6 = tl.load(in_ptr1 + (r2 + (768*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(rmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask, tmp10_weight_next, tmp10_weight)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp19 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full([XBLOCK, RBLOCK], 50257, tl.int32)
        tmp14 = tmp0 + tmp13
        tmp15 = tmp0 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp0)
        tl.device_assert((0 <= tmp16) & (tmp16 < 50257), "index out of bounds: 0 <= tmp16 < 50257")
        tmp18 = tl.load(in_ptr1 + (r2 + (768*tmp16)), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20 - tmp10
        tmp22 = 768.0
        tmp23 = tmp11 / tmp22
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = libdevice.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bq/cbqcfmtunjzwxpo5mwsfatq4kcogu6hrgpvetjizms3jtwiuwpgy.py
# Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute, %permute_1, %permute_2, %expand_3, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512) % 512
    x3 = xindex
    tmp0 = x0
    tmp1 = 1 + x1
    tmp2 = tmp0 < tmp1
    tmp3 = 0.0
    tmp4 = -3.4028234663852886e+38
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/m2/cm27xef4aznwd3c46u26h5qv3j4rjpwiycznteuhrxjjxihmqhap.py
# Topologically Sorted Source Nodes: [inputs_embeds, position_embeds, add, hidden_states_1, hidden_states_2], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add
#   hidden_states_1 => add_4
#   hidden_states_2 => add_5, add_6, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   inputs_embeds => embedding
#   position_embeds => embedding_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %unsqueeze), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_9, %add), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_10), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg9_1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg10_1), kwargs = {})
triton_per_fused_add_embedding_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([RBLOCK], 50257, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert((0 <= tmp7) & (tmp7 < 50257), "index out of bounds: 0 <= tmp7 < 50257")
    tmp9 = tl.load(in_ptr2 + (r2 + (768*tmp7)), rmask, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp2 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp12, rmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp39, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ka/cka2ol2ondps4npbc5srlvsr4nuragfhotvh7vxpn6ulr5pldb55.py
# Topologically Sorted Source Nodes: [mul, pow_1, mul_1, add_3, mul_2, tanh, add_4, hidden_states_3], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_3 => add_7
#   add_4 => add_8
#   hidden_states_3 => mul_7
#   mul => mul_4
#   mul_1 => mul_5
#   mul_2 => mul_6
#   pow_1 => pow_1
#   tanh => tanh
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, 0.5), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_11, 3.0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.7978845608028654), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_6,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_8), kwargs = {})
triton_poi_fused_add_mul_pow_tanh_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = tmp2 * tmp2
    tmp6 = tmp5 * tmp2
    tmp7 = 0.044715
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp4 * tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mq/cmqrjt2jm2thdx4g235twafwgxvd3aju67fyjdqhtxof4bvqlnnn.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_5 => add_9
#   hidden_states_6 => add_10, add_11, mul_8, mul_9, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_13), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_12), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %arg15_1), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg16_1), kwargs = {})
triton_per_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_4', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 8192
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nr/cnrmcsc6nvbsn4fsall2n4szph25u3druxox2vfmxf4mpl7w6ti4.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_7, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_5 => add_9
#   hidden_states_7 => add_12
#   hidden_states_8 => add_13, add_14, mul_10, mul_11, rsqrt_3, sub_3, var_mean_3
# Graph fragment:
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_13), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_21, %add_9), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_21), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %arg21_1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %arg22_1), kwargs = {})
triton_per_fused_add_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 8192
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4d/c4d4ewqvrmw2kgzvy7mkfpclrljhltls5rpynu6zf7ibikmjkkaz.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_24, %full_default_4], 1), kwargs = {})
triton_poi_fused_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38599680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 50260
    x1 = (xindex // 50260)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 50257, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + (768*x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 50260, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x0 + (50272*x1)), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3m/c3mkoeh6s4kjr6qyxuuqtm337wrdvrlt6vocsgqbpd6e2uydlelp.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   loss => amax, exp, sub_13, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_77, [1], True), kwargs = {})
#   %sub_13 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_77, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_13,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_red_fused__log_softmax_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8176
    rnumel = 50257
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50272*(x0 % 511)) + (25739264*(x0 // 511))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (50272*(x0 % 511)) + (25739264*(x0 // 511))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pr/cprxjoslawrtzqp74k4ghdboruftdcv22l4ey7zmcmvf6gszg4s7.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type, div, full_default_3, ne_1, ne_2, neg, sum_2, sum_3, where_2
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_78, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_3), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_2,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_78, -100), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type), kwargs = {})
triton_red_fused_nll_loss_forward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {5: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 8176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (1 + (512*(r0 // 511)) + (r0 % 511)), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = tl.full([XBLOCK, RBLOCK], 50257, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 50257)) | ~(rmask), "index out of bounds: 0 <= tmp8 < 50257")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (50272*(r0 % 511)) + (25739264*(r0 // 511))), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 - tmp11
        tmp14 = tl_math.log(tmp13)
        tmp15 = tmp12 - tmp14
        tmp16 = -tmp15
        tmp17 = 0.0
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask, tmp21, _tmp20)
        tmp22 = tmp2.to(tl.int64)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp26 = tmp24.to(tl.float32)
    tmp27 = tmp20 / tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 512), (512, 1))
    assert_size_stride(arg1_1, (50257, 768), (768, 1))
    assert_size_stride(arg2_1, (1024, 768), (768, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (2304, ), (1, ))
    assert_size_stride(arg6_1, (768, 2304), (2304, 1))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, 768), (768, 1))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (3072, ), (1, ))
    assert_size_stride(arg12_1, (768, 3072), (3072, 1))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (3072, 768), (768, 1))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (2304, ), (1, ))
    assert_size_stride(arg18_1, (768, 2304), (2304, 1))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, 768), (768, 1))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (3072, ), (1, ))
    assert_size_stride(arg24_1, (768, 3072), (3072, 1))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (3072, 768), (768, 1))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (2304, ), (1, ))
    assert_size_stride(arg30_1, (768, 2304), (2304, 1))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, 768), (768, 1))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (3072, ), (1, ))
    assert_size_stride(arg36_1, (768, 3072), (3072, 1))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (3072, 768), (768, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (2304, ), (1, ))
    assert_size_stride(arg42_1, (768, 2304), (2304, 1))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, 768), (768, 1))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (3072, ), (1, ))
    assert_size_stride(arg48_1, (768, 3072), (3072, 1))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (3072, 768), (768, 1))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (2304, ), (1, ))
    assert_size_stride(arg54_1, (768, 2304), (2304, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (3072, ), (1, ))
    assert_size_stride(arg60_1, (768, 3072), (3072, 1))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (3072, 768), (768, 1))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (2304, ), (1, ))
    assert_size_stride(arg66_1, (768, 2304), (2304, 1))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, 768), (768, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (3072, ), (1, ))
    assert_size_stride(arg72_1, (768, 3072), (3072, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (3072, 768), (768, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (16, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((16, 512, 768), (393216, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, position_embeds, add, hidden_states], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_native_layer_norm_0.run(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, buf3, 8192, 768, grid=grid(8192), stream=stream0)
        del arg3_1
        del arg4_1
        buf4 = empty_strided_cuda((8192, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg5_1, reinterpret_tensor(buf3, (8192, 768), (768, 1), 0), arg6_1, alpha=1, beta=1, out=buf4)
        del arg5_1
        del arg6_1
        buf5 = empty_strided_cuda((16, 12, 512, 512), (3145728, 262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf5, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf4, (16, 12, 512, 64), (1179648, 64, 2304, 1), 0), reinterpret_tensor(buf4, (16, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf4, (16, 12, 512, 64), (1179648, 64, 2304, 1), 1536), buf5, False)
        buf7 = buf6[0]
        del buf6
        buf11 = reinterpret_tensor(buf3, (8192, 768), (768, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (8192, 768), (768, 1), 0), arg8_1, out=buf11)
        del arg8_1
        buf12 = reinterpret_tensor(buf11, (16, 512, 768), (393216, 768, 1), 0); del buf11  # reuse
        buf16 = reinterpret_tensor(buf7, (16, 512, 768), (393216, 768, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds, position_embeds, add, hidden_states_1, hidden_states_2], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
        triton_per_fused_add_embedding_native_layer_norm_2.run(buf12, arg7_1, arg0_1, arg1_1, arg2_1, arg9_1, arg10_1, buf16, 8192, 768, grid=grid(8192), stream=stream0)
        del arg0_1
        del arg10_1
        del arg2_1
        del arg7_1
        del arg9_1
        buf17 = empty_strided_cuda((8192, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf16, (8192, 768), (768, 1), 0), arg12_1, out=buf17)
        del arg12_1
        buf18 = reinterpret_tensor(buf17, (16, 512, 3072), (1572864, 3072, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [mul, pow_1, mul_1, add_3, mul_2, tanh, add_4, hidden_states_3], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf18, arg11_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg11_1
        buf19 = reinterpret_tensor(buf16, (8192, 768), (768, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (8192, 3072), (3072, 1), 0), arg14_1, out=buf19)
        del arg14_1
        buf23 = empty_strided_cuda((16, 512, 768), (393216, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf12, buf19, arg13_1, arg15_1, arg16_1, buf23, 8192, 768, grid=grid(8192), stream=stream0)
        del arg15_1
        del arg16_1
        buf24 = empty_strided_cuda((8192, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg17_1, reinterpret_tensor(buf23, (8192, 768), (768, 1), 0), arg18_1, alpha=1, beta=1, out=buf24)
        del arg17_1
        del arg18_1
        buf25 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [attn_output_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf25, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf26 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf24, (16, 12, 512, 64), (1179648, 64, 2304, 1), 0), reinterpret_tensor(buf24, (16, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf24, (16, 12, 512, 64), (1179648, 64, 2304, 1), 1536), buf25, False)
        buf27 = buf26[0]
        del buf26
        buf31 = reinterpret_tensor(buf23, (8192, 768), (768, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (8192, 768), (768, 1), 0), arg20_1, out=buf31)
        del arg20_1
        buf32 = reinterpret_tensor(buf31, (16, 512, 768), (393216, 768, 1), 0); del buf31  # reuse
        buf36 = reinterpret_tensor(buf27, (16, 512, 768), (393216, 768, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_7, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf32, arg19_1, buf12, buf19, arg13_1, arg21_1, arg22_1, buf36, 8192, 768, grid=grid(8192), stream=stream0)
        del arg13_1
        del arg19_1
        del arg21_1
        del arg22_1
        del buf12
        buf37 = reinterpret_tensor(buf18, (8192, 3072), (3072, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf36, (8192, 768), (768, 1), 0), arg24_1, out=buf37)
        del arg24_1
        buf38 = reinterpret_tensor(buf37, (16, 512, 3072), (1572864, 3072, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [mul_4, pow_2, mul_5, add_7, mul_6, tanh_1, add_8, hidden_states_9], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf38, arg23_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg23_1
        buf39 = reinterpret_tensor(buf36, (8192, 768), (768, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (8192, 3072), (3072, 1), 0), arg26_1, out=buf39)
        del arg26_1
        buf43 = reinterpret_tensor(buf19, (16, 512, 768), (393216, 768, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_11, hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf32, buf39, arg25_1, arg27_1, arg28_1, buf43, 8192, 768, grid=grid(8192), stream=stream0)
        del arg27_1
        del arg28_1
        buf44 = empty_strided_cuda((8192, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg29_1, reinterpret_tensor(buf43, (8192, 768), (768, 1), 0), arg30_1, alpha=1, beta=1, out=buf44)
        del arg29_1
        del arg30_1
        buf45 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [attn_output_8], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf45, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_8], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf46 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf44, (16, 12, 512, 64), (1179648, 64, 2304, 1), 0), reinterpret_tensor(buf44, (16, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf44, (16, 12, 512, 64), (1179648, 64, 2304, 1), 1536), buf45, False)
        buf47 = buf46[0]
        del buf46
        buf51 = reinterpret_tensor(buf43, (8192, 768), (768, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (8192, 768), (768, 1), 0), arg32_1, out=buf51)
        del arg32_1
        buf52 = reinterpret_tensor(buf51, (16, 512, 768), (393216, 768, 1), 0); del buf51  # reuse
        buf56 = reinterpret_tensor(buf47, (16, 512, 768), (393216, 768, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_11, hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf52, arg31_1, buf32, buf39, arg25_1, arg33_1, arg34_1, buf56, 8192, 768, grid=grid(8192), stream=stream0)
        del arg25_1
        del arg31_1
        del arg33_1
        del arg34_1
        del buf32
        buf57 = reinterpret_tensor(buf38, (8192, 3072), (3072, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (8192, 768), (768, 1), 0), arg36_1, out=buf57)
        del arg36_1
        buf58 = reinterpret_tensor(buf57, (16, 512, 3072), (1572864, 3072, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [mul_8, pow_3, mul_9, add_11, mul_10, tanh_2, add_12, hidden_states_15], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf58, arg35_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg35_1
        buf59 = reinterpret_tensor(buf56, (8192, 768), (768, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (8192, 3072), (3072, 1), 0), arg38_1, out=buf59)
        del arg38_1
        buf63 = reinterpret_tensor(buf39, (16, 512, 768), (393216, 768, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf52, buf59, arg37_1, arg39_1, arg40_1, buf63, 8192, 768, grid=grid(8192), stream=stream0)
        del arg39_1
        del arg40_1
        buf64 = empty_strided_cuda((8192, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg41_1, reinterpret_tensor(buf63, (8192, 768), (768, 1), 0), arg42_1, alpha=1, beta=1, out=buf64)
        del arg41_1
        del arg42_1
        buf65 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [attn_output_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf65, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf66 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf64, (16, 12, 512, 64), (1179648, 64, 2304, 1), 0), reinterpret_tensor(buf64, (16, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf64, (16, 12, 512, 64), (1179648, 64, 2304, 1), 1536), buf65, False)
        buf67 = buf66[0]
        del buf66
        buf71 = reinterpret_tensor(buf63, (8192, 768), (768, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (8192, 768), (768, 1), 0), arg44_1, out=buf71)
        del arg44_1
        buf72 = reinterpret_tensor(buf71, (16, 512, 768), (393216, 768, 1), 0); del buf71  # reuse
        buf76 = reinterpret_tensor(buf67, (16, 512, 768), (393216, 768, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_19, hidden_states_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf72, arg43_1, buf52, buf59, arg37_1, arg45_1, arg46_1, buf76, 8192, 768, grid=grid(8192), stream=stream0)
        del arg37_1
        del arg43_1
        del arg45_1
        del arg46_1
        del buf52
        buf77 = reinterpret_tensor(buf58, (8192, 3072), (3072, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (8192, 768), (768, 1), 0), arg48_1, out=buf77)
        del arg48_1
        buf78 = reinterpret_tensor(buf77, (16, 512, 3072), (1572864, 3072, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [mul_12, pow_4, mul_13, add_15, mul_14, tanh_3, add_16, hidden_states_21], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf78, arg47_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg47_1
        buf79 = reinterpret_tensor(buf76, (8192, 768), (768, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (8192, 3072), (3072, 1), 0), arg50_1, out=buf79)
        del arg50_1
        buf83 = reinterpret_tensor(buf59, (16, 512, 768), (393216, 768, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23, hidden_states_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf72, buf79, arg49_1, arg51_1, arg52_1, buf83, 8192, 768, grid=grid(8192), stream=stream0)
        del arg51_1
        del arg52_1
        buf84 = empty_strided_cuda((8192, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg53_1, reinterpret_tensor(buf83, (8192, 768), (768, 1), 0), arg54_1, alpha=1, beta=1, out=buf84)
        del arg53_1
        del arg54_1
        buf85 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [attn_output_16], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf85, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_16], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf86 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf84, (16, 12, 512, 64), (1179648, 64, 2304, 1), 0), reinterpret_tensor(buf84, (16, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf84, (16, 12, 512, 64), (1179648, 64, 2304, 1), 1536), buf85, False)
        buf87 = buf86[0]
        del buf86
        buf91 = reinterpret_tensor(buf83, (8192, 768), (768, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (8192, 768), (768, 1), 0), arg56_1, out=buf91)
        del arg56_1
        buf92 = reinterpret_tensor(buf91, (16, 512, 768), (393216, 768, 1), 0); del buf91  # reuse
        buf96 = reinterpret_tensor(buf87, (16, 512, 768), (393216, 768, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23, hidden_states_25, hidden_states_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf92, arg55_1, buf72, buf79, arg49_1, arg57_1, arg58_1, buf96, 8192, 768, grid=grid(8192), stream=stream0)
        del arg49_1
        del arg55_1
        del arg57_1
        del arg58_1
        del buf72
        buf97 = reinterpret_tensor(buf78, (8192, 3072), (3072, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (8192, 768), (768, 1), 0), arg60_1, out=buf97)
        del arg60_1
        buf98 = reinterpret_tensor(buf97, (16, 512, 3072), (1572864, 3072, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [mul_16, pow_5, mul_17, add_19, mul_18, tanh_4, add_20, hidden_states_27], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf98, arg59_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg59_1
        buf99 = reinterpret_tensor(buf96, (8192, 768), (768, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (8192, 3072), (3072, 1), 0), arg62_1, out=buf99)
        del arg62_1
        buf103 = reinterpret_tensor(buf79, (16, 512, 768), (393216, 768, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_29, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf92, buf99, arg61_1, arg63_1, arg64_1, buf103, 8192, 768, grid=grid(8192), stream=stream0)
        del arg63_1
        del arg64_1
        buf104 = empty_strided_cuda((8192, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg65_1, reinterpret_tensor(buf103, (8192, 768), (768, 1), 0), arg66_1, alpha=1, beta=1, out=buf104)
        del arg65_1
        del arg66_1
        buf105 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [attn_output_20], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf105, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_20], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf106 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf104, (16, 12, 512, 64), (1179648, 64, 2304, 1), 0), reinterpret_tensor(buf104, (16, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf104, (16, 12, 512, 64), (1179648, 64, 2304, 1), 1536), buf105, False)
        del buf105
        buf107 = buf106[0]
        del buf106
        buf111 = reinterpret_tensor(buf103, (8192, 768), (768, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (8192, 768), (768, 1), 0), arg68_1, out=buf111)
        del arg68_1
        buf112 = reinterpret_tensor(buf111, (16, 512, 768), (393216, 768, 1), 0); del buf111  # reuse
        buf116 = reinterpret_tensor(buf107, (16, 512, 768), (393216, 768, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_29, hidden_states_31, hidden_states_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf112, arg67_1, buf92, buf99, arg61_1, arg69_1, arg70_1, buf116, 8192, 768, grid=grid(8192), stream=stream0)
        del arg61_1
        del arg67_1
        del arg69_1
        del arg70_1
        del buf92
        buf117 = reinterpret_tensor(buf98, (8192, 3072), (3072, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (8192, 768), (768, 1), 0), arg72_1, out=buf117)
        del arg72_1
        buf118 = reinterpret_tensor(buf117, (16, 512, 3072), (1572864, 3072, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [mul_20, pow_6, mul_21, add_23, mul_22, tanh_5, add_24, hidden_states_33], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf118, arg71_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg71_1
        buf119 = reinterpret_tensor(buf116, (8192, 768), (768, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (8192, 3072), (3072, 1), 0), arg74_1, out=buf119)
        del arg74_1
        del buf118
        buf123 = reinterpret_tensor(buf99, (16, 512, 768), (393216, 768, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, layer_norm_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf112, buf119, arg73_1, arg75_1, arg76_1, buf123, 8192, 768, grid=grid(8192), stream=stream0)
        del arg73_1
        del arg75_1
        del arg76_1
        del buf112
        del buf119
        buf124 = empty_strided_cuda((768, 50260), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(arg1_1, buf124, 38599680, grid=grid(38599680), stream=stream0)
        del arg1_1
        buf125 = empty_strided_cuda((8192, 50260), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (8192, 768), (768, 1), 0), buf124, out=buf125)
        del buf123
        del buf124
        buf126 = empty_strided_cuda((8176, 1), (1, 8192), torch.float32)
        buf127 = empty_strided_cuda((8176, 1), (1, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_7.run(buf125, buf126, buf127, 8176, 50257, grid=grid(8176), stream=stream0)
        buf128 = empty_strided_cuda((), (), torch.float32)
        buf130 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_8.run(buf130, arg77_1, buf125, buf126, buf127, 1, 8176, grid=grid(1), stream=stream0)
        del arg77_1
        del buf126
        del buf127
    return (buf130, reinterpret_tensor(buf125, (16, 512, 50257), (25739264, 50272, 1), 0), reinterpret_tensor(buf4, (16, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf4, (16, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf24, (16, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf24, (16, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf44, (16, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf44, (16, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf64, (16, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf64, (16, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf84, (16, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf84, (16, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf104, (16, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf104, (16, 12, 512, 64), (1179648, 64, 2304, 1), 1536), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((16, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistillGPT2', benchmark_compiled_module)
