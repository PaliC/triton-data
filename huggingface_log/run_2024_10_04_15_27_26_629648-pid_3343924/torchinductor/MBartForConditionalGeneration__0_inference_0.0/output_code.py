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


# kernel path: /tmp/torchinductor_sahanp/gf/cgfnwphxvpe3wwkv3ftwyhzklyes4ubhjsqhsvru4ouuokzoht3k.py
# Topologically Sorted Source Nodes: [embedding, inputs_embeds, add, embed_pos, hidden_states, hidden_states_1, hidden_states_3], Original ATen: [aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add
#   embed_pos => embedding_1
#   embedding => embedding
#   hidden_states => add_1
#   hidden_states_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
#   hidden_states_3 => add_4, add_5, mul_3, mul_4, rsqrt_1, sub_2, var_mean_1
#   inputs_embeds => mul
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %view, 1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, 1.0), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand, 2), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %add), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg4_1), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg5_1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_3), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg6_1), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg7_1), kwargs = {})
triton_red_fused_add_embedding_mul_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x0 = xindex % 1024
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr2 + (2048 + r2 + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert((0 <= tmp4) & (tmp4 < 50265), "index out of bounds: 0 <= tmp4 < 50265")
        tmp6 = tl.load(in_ptr1 + (r2 + (1024*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
        tmp10 = tmp8 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(rmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask, tmp12_weight_next, tmp12_weight)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp37_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp37_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp37_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp23 = tl.load(in_ptr2 + (2048 + r2 + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp16 = tmp0 + tmp15
        tmp17 = tmp0 < 0
        tmp18 = tl.where(tmp17, tmp16, tmp0)
        tl.device_assert((0 <= tmp18) & (tmp18 < 50265), "index out of bounds: 0 <= tmp18 < 50265")
        tmp20 = tl.load(in_ptr1 + (r2 + (1024*tmp18)), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = 1.0
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tmp25 = tmp24 - tmp12
        tmp26 = 1024.0
        tmp27 = tmp13 / tmp26
        tmp28 = 1e-05
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp33 = tmp31 * tmp32
        tmp35 = tmp33 + tmp34
        tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
        tmp37_mean_next, tmp37_m2_next, tmp37_weight_next = triton_helpers.welford_reduce(
            tmp36, tmp37_mean, tmp37_m2, tmp37_weight, roffset == 0
        )
        tmp37_mean = tl.where(rmask, tmp37_mean_next, tmp37_mean)
        tmp37_m2 = tl.where(rmask, tmp37_m2_next, tmp37_m2)
        tmp37_weight = tl.where(rmask, tmp37_weight_next, tmp37_weight)
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp35, rmask)
    tmp37_tmp, tmp38_tmp, tmp39_tmp = triton_helpers.welford(
        tmp37_mean, tmp37_m2, tmp37_weight, 1
    )
    tmp37 = tmp37_tmp[:, None]
    tmp38 = tmp38_tmp[:, None]
    tmp39 = tmp39_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp40 = tl.load(out_ptr2 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp48 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp50 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tmp40 - tmp37
        tmp42 = 1024.0
        tmp43 = tmp38 / tmp42
        tmp44 = 1e-05
        tmp45 = tmp43 + tmp44
        tmp46 = libdevice.rsqrt(tmp45)
        tmp47 = tmp41 * tmp46
        tmp49 = tmp47 * tmp48
        tmp51 = tmp49 + tmp50
        tl.store(out_ptr5 + (r2 + (1024*x3)), tmp51, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rk/crkut6dhi6os5npdt2se5ly2ubt6me7zyrkfjnedb7dxivqwdemh.py
# Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_efficient_attention
#   key_states => clone_3
#   query_states_1 => clone_5
#   value_states => clone_4
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_5, %clone_3, %clone_4, None, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536) % 16
    x3 = (xindex // 1048576)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1) + (1048576*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gt/cgt6jmjfly7rl2valfkyot323uwrsbtfkgdgokt7g3gonlhjk62q.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_5 => add_6
#   hidden_states_6 => add_7, add_8, mul_5, mul_6, rsqrt_2, sub_3, var_mean_2
# Graph fragment:
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_12), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_6, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %getitem_9), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %arg16_1), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %arg17_1), kwargs = {})
triton_per_fused_add_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_2', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 1024, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp4 - tmp12
    tmp19 = 1024.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ha/chaa7dapg2vp3i42mv3wawia4onornvcfeoqfczy3sxjafczjs2v.py
# Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_7 => add_9, erf, mul_7, mul_8, mul_9
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.5), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_8,), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_9), kwargs = {})
triton_poi_fused_gelu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/j6/cj63h3feuod2yb4y5ymg5js5pex55j7e2ufflxbufgxrnnsijbcm.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_11, hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_11 => add_10
#   hidden_states_12 => add_11, add_12, mul_10, mul_11, rsqrt_3, sub_4, var_mean_3
#   hidden_states_5 => add_6
# Graph fragment:
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_12), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_16), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_10, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_10, %getitem_11), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %arg22_1), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %arg23_1), kwargs = {})
triton_per_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), None)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tl.full([1], 1024, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp9 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tmp8 - tmp16
    tmp23 = 1024.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, None)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fj/cfjp7i57nck2cscwwvoeqlmh6xov6km7lrd62yffpkmiu2ioqy6s.py
# Topologically Sorted Source Nodes: [eq, masked_fill_, ne, sum_1], Original ATen: [aten.eq, aten.masked_fill, aten.ne, aten.sum]
# Source node to ATen node mapping:
#   eq => eq
#   masked_fill_ => full_default, where
#   ne => ne
#   sum_1 => sum_1
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg0_1, -100), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=5] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %arg0_1), kwargs = {})
#   %ne : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%where, 1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%ne, [1]), kwargs = {})
triton_per_fused_eq_masked_fill_ne_sum_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_eq_masked_fill_ne_sum_5', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 2
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), None)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tl.where(tmp2, tmp3, tmp0)
    tmp5 = tmp4 != tmp3
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tl.store(out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wl/cwlqssca2htuhnigpkusfw6q52bltyckbxfltxghfpdardx22ukw.py
# Topologically Sorted Source Nodes: [eq, masked_fill_, clone_1, setitem, setitem_1, embedding_2, inputs_embeds_1, add_27, positions_2, hidden_states_112, hidden_states_113, hidden_states_115], Original ATen: [aten.eq, aten.masked_fill, aten.clone, aten.copy, aten.view, aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_27 => add_91
#   clone_1 => clone_1
#   embedding_2 => embedding_2, view_194
#   eq => eq
#   hidden_states_112 => add_92
#   hidden_states_113 => add_93, add_94, mul_90, mul_91, rsqrt_26, sub_27, var_mean_26
#   hidden_states_115 => add_95, add_96, mul_92, mul_93, rsqrt_27, sub_28, var_mean_27
#   inputs_embeds_1 => mul_89
#   masked_fill_ => full_default, where
#   positions_2 => embedding_3
#   setitem => copy
#   setitem_1 => copy_1
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg0_1, -100), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=5] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %arg0_1), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_8, %clone_1), kwargs = {})
#   %slice_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%where, %copy, 1, 1, 9223372036854775807), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_1, %squeeze), kwargs = {})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%slice_scatter_default, %copy_1, 1, 0), kwargs = {})
#   %view_194 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_scatter_default, [-1, 1024]), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %view_194, 1), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding_2, 1.0), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_2, 2), kwargs = {})
#   %embedding_3 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg200_1, %add_91), kwargs = {})
#   %add_92 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %embedding_3), kwargs = {})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_92, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_92, %getitem_101), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_100, 1e-05), kwargs = {})
#   %rsqrt_26 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_93,), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_26), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %arg201_1), kwargs = {})
#   %add_94 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %arg202_1), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_94, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_94, %getitem_103), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_102, 1e-05), kwargs = {})
#   %rsqrt_27 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_95,), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_27), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_92, %arg203_1), kwargs = {})
#   %add_96 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_93, %arg204_1), kwargs = {})
triton_red_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_view_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_view_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp3 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    x3 = xindex
    tmp21 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp37_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp37_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp37_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp34 = tl.load(in_ptr3 + (2048 + r2 + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp3 - tmp4
        tmp6 = tl.full([XBLOCK, RBLOCK], 1024, tl.int32)
        tmp7 = tmp5 + tmp6
        tmp8 = tmp5 < 0
        tmp9 = tl.where(tmp8, tmp7, tmp5)
        tl.device_assert((0 <= tmp9) & (tmp9 < 1024), "index out of bounds: 0 <= tmp9 < 1024")
        tmp11 = tl.load(in_ptr1 + (tmp9 + (1024*x1)), None, eviction_policy='evict_last')
        tmp12 = tl.full([1, 1], -100, tl.int64)
        tmp13 = tmp11 == tmp12
        tmp14 = tl.where(tmp13, tmp4, tmp11)
        tmp15 = tmp0 >= tmp4
        tmp16 = tl.load(in_ptr1 + (tl.broadcast_to((-1) + x3, [XBLOCK, RBLOCK])), rmask & tmp15, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp16 == tmp12
        tmp18 = tl.where(tmp17, tmp4, tmp16)
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp15, tmp18, tmp19)
        tmp22 = tmp21 == tmp12
        tmp23 = tl.where(tmp22, tmp4, tmp21)
        tmp24 = tl.where(tmp15, tmp20, tmp23)
        tmp25 = tl.where(tmp2, tmp14, tmp24)
        tmp26 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp27 = tmp25 + tmp26
        tmp28 = tmp25 < 0
        tmp29 = tl.where(tmp28, tmp27, tmp25)
        tl.device_assert(((0 <= tmp29) & (tmp29 < 50265)) | ~(rmask), "index out of bounds: 0 <= tmp29 < 50265")
        tmp31 = tl.load(in_ptr2 + (r2 + (1024*tmp29)), rmask, eviction_policy='evict_first', other=0.0)
        tmp32 = 1.0
        tmp33 = tmp31 * tmp32
        tmp35 = tmp33 + tmp34
        tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
        tmp37_mean_next, tmp37_m2_next, tmp37_weight_next = triton_helpers.welford_reduce(
            tmp36, tmp37_mean, tmp37_m2, tmp37_weight, roffset == 0
        )
        tmp37_mean = tl.where(rmask, tmp37_mean_next, tmp37_mean)
        tmp37_m2 = tl.where(rmask, tmp37_m2_next, tmp37_m2)
        tmp37_weight = tl.where(rmask, tmp37_weight_next, tmp37_weight)
        tl.store(out_ptr0 + (r2 + (1024*x3)), tmp35, rmask)
    tmp37_tmp, tmp38_tmp, tmp39_tmp = triton_helpers.welford(
        tmp37_mean, tmp37_m2, tmp37_weight, 1
    )
    tmp37 = tmp37_tmp[:, None]
    tmp38 = tmp38_tmp[:, None]
    tmp39 = tmp39_tmp[:, None]
    tmp53_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp53_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp53_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp40 = tl.load(out_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp48 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp50 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tmp40 - tmp37
        tmp42 = 1024.0
        tmp43 = tmp38 / tmp42
        tmp44 = 1e-05
        tmp45 = tmp43 + tmp44
        tmp46 = libdevice.rsqrt(tmp45)
        tmp47 = tmp41 * tmp46
        tmp49 = tmp47 * tmp48
        tmp51 = tmp49 + tmp50
        tmp52 = tl.broadcast_to(tmp51, [XBLOCK, RBLOCK])
        tmp53_mean_next, tmp53_m2_next, tmp53_weight_next = triton_helpers.welford_reduce(
            tmp52, tmp53_mean, tmp53_m2, tmp53_weight, roffset == 0
        )
        tmp53_mean = tl.where(rmask, tmp53_mean_next, tmp53_mean)
        tmp53_m2 = tl.where(rmask, tmp53_m2_next, tmp53_m2)
        tmp53_weight = tl.where(rmask, tmp53_weight_next, tmp53_weight)
        tl.store(out_ptr3 + (r2 + (1024*x3)), tmp51, rmask)
    tmp53_tmp, tmp54_tmp, tmp55_tmp = triton_helpers.welford(
        tmp53_mean, tmp53_m2, tmp53_weight, 1
    )
    tmp53 = tmp53_tmp[:, None]
    tmp54 = tmp54_tmp[:, None]
    tmp55 = tmp55_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp56 = tl.load(out_ptr3 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp64 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp66 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp57 = tmp56 - tmp53
        tmp58 = 1024.0
        tmp59 = tmp54 / tmp58
        tmp60 = 1e-05
        tmp61 = tmp59 + tmp60
        tmp62 = libdevice.rsqrt(tmp61)
        tmp63 = tmp57 * tmp62
        tmp65 = tmp63 * tmp64
        tmp67 = tmp65 + tmp66
        tl.store(out_ptr6 + (r2 + (1024*x3)), tmp67, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nh/cnhquny7n6b3zazibmjpsm22qr3ci6y4xvtlsor5fnnfmyqd72c2.py
# Topologically Sorted Source Nodes: [query_states_25, key_states_12, value_states_12, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output_48 => _scaled_dot_product_efficient_attention_12
#   key_states_12 => clone_76
#   query_states_25 => clone_78
#   value_states_12 => clone_77
# Graph fragment:
#   %clone_78 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_125,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_76 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_122,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_77 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_124,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_dot_product_efficient_attention_12 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_78, %clone_76, %clone_77, %expand_5, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 1024
    x3 = xindex
    tmp0 = x0
    tmp1 = 1 + x1
    tmp2 = tmp0 < tmp1
    tmp3 = 0.0
    tmp4 = -3.4028234663852886e+38
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5t/c5tjtozhwohp5gqhhoytvcgv3tpiy4dp4rey6uxpufr76m4hlwmm.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_336, %full_default_5], 1), kwargs = {})
triton_poi_fused_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51474432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 50268
    x1 = (xindex // 50268)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 50265, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + (1024*x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 50268, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x0 + (50272*x1)), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ja/cjayq2lnm3t4rwsg6xblzzksaua2a67t2zmvfuncnf7xlfhw4ysy.py
# Topologically Sorted Source Nodes: [lm_logits, masked_lm_loss], Original ATen: [aten.add, aten._log_softmax]
# Source node to ATen node mapping:
#   lm_logits => add_217
#   masked_lm_loss => amax, exp, sub_65, sum_2
# Graph fragment:
#   %add_217 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_533, %arg517_1), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_534, [1], True), kwargs = {})
#   %sub_65 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_534, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_65,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_red_fused__log_softmax_add_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_add_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 50265
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50272*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tl.store(out_ptr0 + (r1 + (50265*x0)), tmp2, rmask)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp4, None)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(out_ptr0 + (r1 + (50265*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp6 - tmp4
        tmp8 = tl_math.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/r2/cr2xcjd5w3ghlxlhypnhq3ggew7tz2h32srph27h6xmmzt2ff52h.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   masked_lm_loss => convert_element_type, div, full_default_4, ne_2, ne_3, neg, sum_3, sum_4, where_3
# Graph fragment:
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_535, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_1,), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_2, %neg, %full_default_4), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_3,), kwargs = {})
#   %ne_3 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_535, -100), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_3,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_3, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_4, %convert_element_type), kwargs = {})
triton_red_fused_nll_loss_forward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {5: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2048
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
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 50265)) | ~(rmask), "index out of bounds: 0 <= tmp8 < 50265")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (50265*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 1024), (1024, 1))
    assert_size_stride(arg1_1, (2, 1024), (1024, 1))
    assert_size_stride(arg2_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg3_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg4_1, (1024, ), (1, ))
    assert_size_stride(arg5_1, (1024, ), (1, ))
    assert_size_stride(arg6_1, (1024, ), (1, ))
    assert_size_stride(arg7_1, (1024, ), (1, ))
    assert_size_stride(arg8_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg9_1, (1024, ), (1, ))
    assert_size_stride(arg10_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg11_1, (1024, ), (1, ))
    assert_size_stride(arg12_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg13_1, (1024, ), (1, ))
    assert_size_stride(arg14_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg15_1, (1024, ), (1, ))
    assert_size_stride(arg16_1, (1024, ), (1, ))
    assert_size_stride(arg17_1, (1024, ), (1, ))
    assert_size_stride(arg18_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg19_1, (4096, ), (1, ))
    assert_size_stride(arg20_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg21_1, (1024, ), (1, ))
    assert_size_stride(arg22_1, (1024, ), (1, ))
    assert_size_stride(arg23_1, (1024, ), (1, ))
    assert_size_stride(arg24_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg25_1, (1024, ), (1, ))
    assert_size_stride(arg26_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg27_1, (1024, ), (1, ))
    assert_size_stride(arg28_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg29_1, (1024, ), (1, ))
    assert_size_stride(arg30_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg31_1, (1024, ), (1, ))
    assert_size_stride(arg32_1, (1024, ), (1, ))
    assert_size_stride(arg33_1, (1024, ), (1, ))
    assert_size_stride(arg34_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg35_1, (4096, ), (1, ))
    assert_size_stride(arg36_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (1024, ), (1, ))
    assert_size_stride(arg40_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg41_1, (1024, ), (1, ))
    assert_size_stride(arg42_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg43_1, (1024, ), (1, ))
    assert_size_stride(arg44_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg45_1, (1024, ), (1, ))
    assert_size_stride(arg46_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg47_1, (1024, ), (1, ))
    assert_size_stride(arg48_1, (1024, ), (1, ))
    assert_size_stride(arg49_1, (1024, ), (1, ))
    assert_size_stride(arg50_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg51_1, (4096, ), (1, ))
    assert_size_stride(arg52_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg53_1, (1024, ), (1, ))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (1024, ), (1, ))
    assert_size_stride(arg56_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg57_1, (1024, ), (1, ))
    assert_size_stride(arg58_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg61_1, (1024, ), (1, ))
    assert_size_stride(arg62_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg63_1, (1024, ), (1, ))
    assert_size_stride(arg64_1, (1024, ), (1, ))
    assert_size_stride(arg65_1, (1024, ), (1, ))
    assert_size_stride(arg66_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg67_1, (4096, ), (1, ))
    assert_size_stride(arg68_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg69_1, (1024, ), (1, ))
    assert_size_stride(arg70_1, (1024, ), (1, ))
    assert_size_stride(arg71_1, (1024, ), (1, ))
    assert_size_stride(arg72_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg73_1, (1024, ), (1, ))
    assert_size_stride(arg74_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg75_1, (1024, ), (1, ))
    assert_size_stride(arg76_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (1024, ), (1, ))
    assert_size_stride(arg82_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg83_1, (4096, ), (1, ))
    assert_size_stride(arg84_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg85_1, (1024, ), (1, ))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg87_1, (1024, ), (1, ))
    assert_size_stride(arg88_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg89_1, (1024, ), (1, ))
    assert_size_stride(arg90_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg93_1, (1024, ), (1, ))
    assert_size_stride(arg94_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg95_1, (1024, ), (1, ))
    assert_size_stride(arg96_1, (1024, ), (1, ))
    assert_size_stride(arg97_1, (1024, ), (1, ))
    assert_size_stride(arg98_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg99_1, (4096, ), (1, ))
    assert_size_stride(arg100_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (1024, ), (1, ))
    assert_size_stride(arg103_1, (1024, ), (1, ))
    assert_size_stride(arg104_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg105_1, (1024, ), (1, ))
    assert_size_stride(arg106_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg107_1, (1024, ), (1, ))
    assert_size_stride(arg108_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg111_1, (1024, ), (1, ))
    assert_size_stride(arg112_1, (1024, ), (1, ))
    assert_size_stride(arg113_1, (1024, ), (1, ))
    assert_size_stride(arg114_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg115_1, (4096, ), (1, ))
    assert_size_stride(arg116_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, ), (1, ))
    assert_size_stride(arg120_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg121_1, (1024, ), (1, ))
    assert_size_stride(arg122_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg123_1, (1024, ), (1, ))
    assert_size_stride(arg124_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (1024, ), (1, ))
    assert_size_stride(arg130_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg131_1, (4096, ), (1, ))
    assert_size_stride(arg132_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg141_1, (1024, ), (1, ))
    assert_size_stride(arg142_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg143_1, (1024, ), (1, ))
    assert_size_stride(arg144_1, (1024, ), (1, ))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg147_1, (4096, ), (1, ))
    assert_size_stride(arg148_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, ), (1, ))
    assert_size_stride(arg152_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg153_1, (1024, ), (1, ))
    assert_size_stride(arg154_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg157_1, (1024, ), (1, ))
    assert_size_stride(arg158_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg159_1, (1024, ), (1, ))
    assert_size_stride(arg160_1, (1024, ), (1, ))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg163_1, (4096, ), (1, ))
    assert_size_stride(arg164_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg165_1, (1024, ), (1, ))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg171_1, (1024, ), (1, ))
    assert_size_stride(arg172_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg173_1, (1024, ), (1, ))
    assert_size_stride(arg174_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (1024, ), (1, ))
    assert_size_stride(arg178_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg179_1, (4096, ), (1, ))
    assert_size_stride(arg180_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, ), (1, ))
    assert_size_stride(arg184_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg185_1, (1024, ), (1, ))
    assert_size_stride(arg186_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg187_1, (1024, ), (1, ))
    assert_size_stride(arg188_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg189_1, (1024, ), (1, ))
    assert_size_stride(arg190_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (1024, ), (1, ))
    assert_size_stride(arg193_1, (1024, ), (1, ))
    assert_size_stride(arg194_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg195_1, (4096, ), (1, ))
    assert_size_stride(arg196_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg197_1, (1024, ), (1, ))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (1024, ), (1, ))
    assert_size_stride(arg200_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg201_1, (1024, ), (1, ))
    assert_size_stride(arg202_1, (1024, ), (1, ))
    assert_size_stride(arg203_1, (1024, ), (1, ))
    assert_size_stride(arg204_1, (1024, ), (1, ))
    assert_size_stride(arg205_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg208_1, (1024, ), (1, ))
    assert_size_stride(arg209_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg210_1, (1024, ), (1, ))
    assert_size_stride(arg211_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (1024, ), (1, ))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg216_1, (1024, ), (1, ))
    assert_size_stride(arg217_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg218_1, (1024, ), (1, ))
    assert_size_stride(arg219_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg220_1, (1024, ), (1, ))
    assert_size_stride(arg221_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg222_1, (1024, ), (1, ))
    assert_size_stride(arg223_1, (1024, ), (1, ))
    assert_size_stride(arg224_1, (1024, ), (1, ))
    assert_size_stride(arg225_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg226_1, (4096, ), (1, ))
    assert_size_stride(arg227_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg228_1, (1024, ), (1, ))
    assert_size_stride(arg229_1, (1024, ), (1, ))
    assert_size_stride(arg230_1, (1024, ), (1, ))
    assert_size_stride(arg231_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg232_1, (1024, ), (1, ))
    assert_size_stride(arg233_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg234_1, (1024, ), (1, ))
    assert_size_stride(arg235_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg236_1, (1024, ), (1, ))
    assert_size_stride(arg237_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg238_1, (1024, ), (1, ))
    assert_size_stride(arg239_1, (1024, ), (1, ))
    assert_size_stride(arg240_1, (1024, ), (1, ))
    assert_size_stride(arg241_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg246_1, (1024, ), (1, ))
    assert_size_stride(arg247_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg248_1, (1024, ), (1, ))
    assert_size_stride(arg249_1, (1024, ), (1, ))
    assert_size_stride(arg250_1, (1024, ), (1, ))
    assert_size_stride(arg251_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg252_1, (4096, ), (1, ))
    assert_size_stride(arg253_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg254_1, (1024, ), (1, ))
    assert_size_stride(arg255_1, (1024, ), (1, ))
    assert_size_stride(arg256_1, (1024, ), (1, ))
    assert_size_stride(arg257_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg258_1, (1024, ), (1, ))
    assert_size_stride(arg259_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg260_1, (1024, ), (1, ))
    assert_size_stride(arg261_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg262_1, (1024, ), (1, ))
    assert_size_stride(arg263_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg264_1, (1024, ), (1, ))
    assert_size_stride(arg265_1, (1024, ), (1, ))
    assert_size_stride(arg266_1, (1024, ), (1, ))
    assert_size_stride(arg267_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg268_1, (1024, ), (1, ))
    assert_size_stride(arg269_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg270_1, (1024, ), (1, ))
    assert_size_stride(arg271_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg272_1, (1024, ), (1, ))
    assert_size_stride(arg273_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg274_1, (1024, ), (1, ))
    assert_size_stride(arg275_1, (1024, ), (1, ))
    assert_size_stride(arg276_1, (1024, ), (1, ))
    assert_size_stride(arg277_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg278_1, (4096, ), (1, ))
    assert_size_stride(arg279_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (1024, ), (1, ))
    assert_size_stride(arg283_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg284_1, (1024, ), (1, ))
    assert_size_stride(arg285_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg286_1, (1024, ), (1, ))
    assert_size_stride(arg287_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg288_1, (1024, ), (1, ))
    assert_size_stride(arg289_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg290_1, (1024, ), (1, ))
    assert_size_stride(arg291_1, (1024, ), (1, ))
    assert_size_stride(arg292_1, (1024, ), (1, ))
    assert_size_stride(arg293_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg294_1, (1024, ), (1, ))
    assert_size_stride(arg295_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg296_1, (1024, ), (1, ))
    assert_size_stride(arg297_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg298_1, (1024, ), (1, ))
    assert_size_stride(arg299_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg300_1, (1024, ), (1, ))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg304_1, (4096, ), (1, ))
    assert_size_stride(arg305_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg306_1, (1024, ), (1, ))
    assert_size_stride(arg307_1, (1024, ), (1, ))
    assert_size_stride(arg308_1, (1024, ), (1, ))
    assert_size_stride(arg309_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg310_1, (1024, ), (1, ))
    assert_size_stride(arg311_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg312_1, (1024, ), (1, ))
    assert_size_stride(arg313_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg314_1, (1024, ), (1, ))
    assert_size_stride(arg315_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (1024, ), (1, ))
    assert_size_stride(arg319_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg320_1, (1024, ), (1, ))
    assert_size_stride(arg321_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg322_1, (1024, ), (1, ))
    assert_size_stride(arg323_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg324_1, (1024, ), (1, ))
    assert_size_stride(arg325_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1024, ), (1, ))
    assert_size_stride(arg328_1, (1024, ), (1, ))
    assert_size_stride(arg329_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg330_1, (4096, ), (1, ))
    assert_size_stride(arg331_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg332_1, (1024, ), (1, ))
    assert_size_stride(arg333_1, (1024, ), (1, ))
    assert_size_stride(arg334_1, (1024, ), (1, ))
    assert_size_stride(arg335_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg336_1, (1024, ), (1, ))
    assert_size_stride(arg337_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg338_1, (1024, ), (1, ))
    assert_size_stride(arg339_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg340_1, (1024, ), (1, ))
    assert_size_stride(arg341_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1024, ), (1, ))
    assert_size_stride(arg344_1, (1024, ), (1, ))
    assert_size_stride(arg345_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg348_1, (1024, ), (1, ))
    assert_size_stride(arg349_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg350_1, (1024, ), (1, ))
    assert_size_stride(arg351_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg352_1, (1024, ), (1, ))
    assert_size_stride(arg353_1, (1024, ), (1, ))
    assert_size_stride(arg354_1, (1024, ), (1, ))
    assert_size_stride(arg355_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg356_1, (4096, ), (1, ))
    assert_size_stride(arg357_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg358_1, (1024, ), (1, ))
    assert_size_stride(arg359_1, (1024, ), (1, ))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg364_1, (1024, ), (1, ))
    assert_size_stride(arg365_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg366_1, (1024, ), (1, ))
    assert_size_stride(arg367_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg368_1, (1024, ), (1, ))
    assert_size_stride(arg369_1, (1024, ), (1, ))
    assert_size_stride(arg370_1, (1024, ), (1, ))
    assert_size_stride(arg371_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg372_1, (1024, ), (1, ))
    assert_size_stride(arg373_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg374_1, (1024, ), (1, ))
    assert_size_stride(arg375_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg376_1, (1024, ), (1, ))
    assert_size_stride(arg377_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (1024, ), (1, ))
    assert_size_stride(arg380_1, (1024, ), (1, ))
    assert_size_stride(arg381_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg382_1, (4096, ), (1, ))
    assert_size_stride(arg383_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg384_1, (1024, ), (1, ))
    assert_size_stride(arg385_1, (1024, ), (1, ))
    assert_size_stride(arg386_1, (1024, ), (1, ))
    assert_size_stride(arg387_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg388_1, (1024, ), (1, ))
    assert_size_stride(arg389_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg390_1, (1024, ), (1, ))
    assert_size_stride(arg391_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg392_1, (1024, ), (1, ))
    assert_size_stride(arg393_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg394_1, (1024, ), (1, ))
    assert_size_stride(arg395_1, (1024, ), (1, ))
    assert_size_stride(arg396_1, (1024, ), (1, ))
    assert_size_stride(arg397_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg398_1, (1024, ), (1, ))
    assert_size_stride(arg399_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg400_1, (1024, ), (1, ))
    assert_size_stride(arg401_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg402_1, (1024, ), (1, ))
    assert_size_stride(arg403_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg404_1, (1024, ), (1, ))
    assert_size_stride(arg405_1, (1024, ), (1, ))
    assert_size_stride(arg406_1, (1024, ), (1, ))
    assert_size_stride(arg407_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg408_1, (4096, ), (1, ))
    assert_size_stride(arg409_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg410_1, (1024, ), (1, ))
    assert_size_stride(arg411_1, (1024, ), (1, ))
    assert_size_stride(arg412_1, (1024, ), (1, ))
    assert_size_stride(arg413_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg414_1, (1024, ), (1, ))
    assert_size_stride(arg415_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg416_1, (1024, ), (1, ))
    assert_size_stride(arg417_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg418_1, (1024, ), (1, ))
    assert_size_stride(arg419_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg420_1, (1024, ), (1, ))
    assert_size_stride(arg421_1, (1024, ), (1, ))
    assert_size_stride(arg422_1, (1024, ), (1, ))
    assert_size_stride(arg423_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg424_1, (1024, ), (1, ))
    assert_size_stride(arg425_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg426_1, (1024, ), (1, ))
    assert_size_stride(arg427_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg428_1, (1024, ), (1, ))
    assert_size_stride(arg429_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg430_1, (1024, ), (1, ))
    assert_size_stride(arg431_1, (1024, ), (1, ))
    assert_size_stride(arg432_1, (1024, ), (1, ))
    assert_size_stride(arg433_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg434_1, (4096, ), (1, ))
    assert_size_stride(arg435_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg436_1, (1024, ), (1, ))
    assert_size_stride(arg437_1, (1024, ), (1, ))
    assert_size_stride(arg438_1, (1024, ), (1, ))
    assert_size_stride(arg439_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg440_1, (1024, ), (1, ))
    assert_size_stride(arg441_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg442_1, (1024, ), (1, ))
    assert_size_stride(arg443_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg444_1, (1024, ), (1, ))
    assert_size_stride(arg445_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg446_1, (1024, ), (1, ))
    assert_size_stride(arg447_1, (1024, ), (1, ))
    assert_size_stride(arg448_1, (1024, ), (1, ))
    assert_size_stride(arg449_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg450_1, (1024, ), (1, ))
    assert_size_stride(arg451_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg452_1, (1024, ), (1, ))
    assert_size_stride(arg453_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg454_1, (1024, ), (1, ))
    assert_size_stride(arg455_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg456_1, (1024, ), (1, ))
    assert_size_stride(arg457_1, (1024, ), (1, ))
    assert_size_stride(arg458_1, (1024, ), (1, ))
    assert_size_stride(arg459_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg460_1, (4096, ), (1, ))
    assert_size_stride(arg461_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg462_1, (1024, ), (1, ))
    assert_size_stride(arg463_1, (1024, ), (1, ))
    assert_size_stride(arg464_1, (1024, ), (1, ))
    assert_size_stride(arg465_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg466_1, (1024, ), (1, ))
    assert_size_stride(arg467_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg468_1, (1024, ), (1, ))
    assert_size_stride(arg469_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg470_1, (1024, ), (1, ))
    assert_size_stride(arg471_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg472_1, (1024, ), (1, ))
    assert_size_stride(arg473_1, (1024, ), (1, ))
    assert_size_stride(arg474_1, (1024, ), (1, ))
    assert_size_stride(arg475_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg476_1, (1024, ), (1, ))
    assert_size_stride(arg477_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg478_1, (1024, ), (1, ))
    assert_size_stride(arg479_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg480_1, (1024, ), (1, ))
    assert_size_stride(arg481_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg482_1, (1024, ), (1, ))
    assert_size_stride(arg483_1, (1024, ), (1, ))
    assert_size_stride(arg484_1, (1024, ), (1, ))
    assert_size_stride(arg485_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg486_1, (4096, ), (1, ))
    assert_size_stride(arg487_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg488_1, (1024, ), (1, ))
    assert_size_stride(arg489_1, (1024, ), (1, ))
    assert_size_stride(arg490_1, (1024, ), (1, ))
    assert_size_stride(arg491_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg492_1, (1024, ), (1, ))
    assert_size_stride(arg493_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg494_1, (1024, ), (1, ))
    assert_size_stride(arg495_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg496_1, (1024, ), (1, ))
    assert_size_stride(arg497_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg498_1, (1024, ), (1, ))
    assert_size_stride(arg499_1, (1024, ), (1, ))
    assert_size_stride(arg500_1, (1024, ), (1, ))
    assert_size_stride(arg501_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg502_1, (1024, ), (1, ))
    assert_size_stride(arg503_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg504_1, (1024, ), (1, ))
    assert_size_stride(arg505_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg506_1, (1024, ), (1, ))
    assert_size_stride(arg507_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg508_1, (1024, ), (1, ))
    assert_size_stride(arg509_1, (1024, ), (1, ))
    assert_size_stride(arg510_1, (1024, ), (1, ))
    assert_size_stride(arg511_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg512_1, (4096, ), (1, ))
    assert_size_stride(arg513_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg514_1, (1024, ), (1, ))
    assert_size_stride(arg515_1, (1024, ), (1, ))
    assert_size_stride(arg516_1, (1024, ), (1, ))
    assert_size_stride(arg517_1, (1, 50265), (50265, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((2, 1024, 1024), (1048576, 1024, 1), torch.float32)
        buf7 = empty_strided_cuda((2, 1024, 1024), (1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, add, embed_pos, hidden_states, hidden_states_1, hidden_states_3], Original ATen: [aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, buf3, buf7, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg1_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        buf8 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg8_1, (1024, 1024), (1, 1024), 0), out=buf8)
        del arg8_1
        buf9 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg10_1, (1024, 1024), (1, 1024), 0), out=buf9)
        del arg10_1
        buf10 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg12_1, (1024, 1024), (1, 1024), 0), out=buf10)
        del arg12_1
        buf11 = reinterpret_tensor(buf7, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf8, arg9_1, buf11, 2097152, grid=grid(2097152), stream=stream0)
        del arg9_1
        buf12 = reinterpret_tensor(buf8, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf9, arg11_1, buf12, 2097152, grid=grid(2097152), stream=stream0)
        del arg11_1
        buf13 = reinterpret_tensor(buf9, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf10, arg13_1, buf13, 2097152, grid=grid(2097152), stream=stream0)
        del arg13_1
        del buf10
        # Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf11, buf12, buf13, None, False)
        buf15 = buf14[0]
        del buf14
        buf19 = reinterpret_tensor(buf13, (2048, 1024), (1024, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg14_1, (1024, 1024), (1, 1024), 0), out=buf19)
        del arg14_1
        buf23 = reinterpret_tensor(buf15, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf3, buf19, arg15_1, arg16_1, arg17_1, buf23, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg16_1
        del arg17_1
        buf24 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg18_1, (1024, 4096), (1, 1024), 0), out=buf24)
        del arg18_1
        buf25 = reinterpret_tensor(buf24, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf25, arg19_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg19_1
        buf26 = reinterpret_tensor(buf23, (2048, 1024), (1024, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf25, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 1024), (1, 4096), 0), out=buf26)
        del arg20_1
        buf27 = reinterpret_tensor(buf26, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf26  # reuse
        buf31 = reinterpret_tensor(buf12, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_11, hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf27, buf3, buf19, arg15_1, arg21_1, arg22_1, arg23_1, buf31, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg15_1
        del arg21_1
        del arg22_1
        del arg23_1
        buf32 = reinterpret_tensor(buf3, (2048, 1024), (1024, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg24_1, (1024, 1024), (1, 1024), 0), out=buf32)
        del arg24_1
        buf33 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg26_1, (1024, 1024), (1, 1024), 0), out=buf33)
        del arg26_1
        buf34 = reinterpret_tensor(buf11, (2048, 1024), (1024, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg28_1, (1024, 1024), (1, 1024), 0), out=buf34)
        del arg28_1
        buf35 = reinterpret_tensor(buf31, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [query_states_3, key_states_1, value_states_1, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf32, arg25_1, buf35, 2097152, grid=grid(2097152), stream=stream0)
        del arg25_1
        buf36 = reinterpret_tensor(buf32, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [query_states_3, key_states_1, value_states_1, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf33, arg27_1, buf36, 2097152, grid=grid(2097152), stream=stream0)
        del arg27_1
        buf37 = reinterpret_tensor(buf33, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [query_states_3, key_states_1, value_states_1, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf34, arg29_1, buf37, 2097152, grid=grid(2097152), stream=stream0)
        del arg29_1
        del buf34
        # Topologically Sorted Source Nodes: [query_states_3, key_states_1, value_states_1, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf38 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf35, buf36, buf37, None, False)
        buf39 = buf38[0]
        del buf38
        buf43 = reinterpret_tensor(buf37, (2048, 1024), (1024, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg30_1, (1024, 1024), (1, 1024), 0), out=buf43)
        del arg30_1
        buf47 = reinterpret_tensor(buf39, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_14, hidden_states_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf27, buf43, arg31_1, arg32_1, arg33_1, buf47, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg32_1
        del arg33_1
        buf48 = reinterpret_tensor(buf25, (2048, 4096), (4096, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg34_1, (1024, 4096), (1, 1024), 0), out=buf48)
        del arg34_1
        buf49 = reinterpret_tensor(buf48, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf49, arg35_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg35_1
        buf50 = reinterpret_tensor(buf47, (2048, 1024), (1024, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg36_1, (4096, 1024), (1, 4096), 0), out=buf50)
        del arg36_1
        buf51 = reinterpret_tensor(buf50, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf50  # reuse
        buf55 = reinterpret_tensor(buf36, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_14, hidden_states_20, hidden_states_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf51, buf27, buf43, arg31_1, arg37_1, arg38_1, arg39_1, buf55, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg31_1
        del arg37_1
        del arg38_1
        del arg39_1
        buf56 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf55, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg40_1, (1024, 1024), (1, 1024), 0), out=buf56)
        del arg40_1
        buf57 = reinterpret_tensor(buf27, (2048, 1024), (1024, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf55, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg42_1, (1024, 1024), (1, 1024), 0), out=buf57)
        del arg42_1
        buf58 = reinterpret_tensor(buf35, (2048, 1024), (1024, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf55, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg44_1, (1024, 1024), (1, 1024), 0), out=buf58)
        del arg44_1
        buf59 = reinterpret_tensor(buf55, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [query_states_5, key_states_2, value_states_2, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf56, arg41_1, buf59, 2097152, grid=grid(2097152), stream=stream0)
        del arg41_1
        buf60 = reinterpret_tensor(buf56, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [query_states_5, key_states_2, value_states_2, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf57, arg43_1, buf60, 2097152, grid=grid(2097152), stream=stream0)
        del arg43_1
        buf61 = reinterpret_tensor(buf57, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [query_states_5, key_states_2, value_states_2, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf58, arg45_1, buf61, 2097152, grid=grid(2097152), stream=stream0)
        del arg45_1
        del buf58
        # Topologically Sorted Source Nodes: [query_states_5, key_states_2, value_states_2, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf62 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf59, buf60, buf61, None, False)
        buf63 = buf62[0]
        del buf62
        buf67 = reinterpret_tensor(buf61, (2048, 1024), (1024, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg46_1, (1024, 1024), (1, 1024), 0), out=buf67)
        del arg46_1
        buf71 = reinterpret_tensor(buf63, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23, hidden_states_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf51, buf67, arg47_1, arg48_1, arg49_1, buf71, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg48_1
        del arg49_1
        buf72 = reinterpret_tensor(buf49, (2048, 4096), (4096, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg50_1, (1024, 4096), (1, 1024), 0), out=buf72)
        del arg50_1
        buf73 = reinterpret_tensor(buf72, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_25], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf73, arg51_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg51_1
        buf74 = reinterpret_tensor(buf71, (2048, 1024), (1024, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg52_1, (4096, 1024), (1, 4096), 0), out=buf74)
        del arg52_1
        buf75 = reinterpret_tensor(buf74, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf74  # reuse
        buf79 = reinterpret_tensor(buf60, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23, hidden_states_29, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf75, buf51, buf67, arg47_1, arg53_1, arg54_1, arg55_1, buf79, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg47_1
        del arg53_1
        del arg54_1
        del arg55_1
        buf80 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg56_1, (1024, 1024), (1, 1024), 0), out=buf80)
        del arg56_1
        buf81 = reinterpret_tensor(buf51, (2048, 1024), (1024, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg58_1, (1024, 1024), (1, 1024), 0), out=buf81)
        del arg58_1
        buf82 = reinterpret_tensor(buf59, (2048, 1024), (1024, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg60_1, (1024, 1024), (1, 1024), 0), out=buf82)
        del arg60_1
        buf83 = reinterpret_tensor(buf79, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [query_states_7, key_states_3, value_states_3, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf80, arg57_1, buf83, 2097152, grid=grid(2097152), stream=stream0)
        del arg57_1
        buf84 = reinterpret_tensor(buf80, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [query_states_7, key_states_3, value_states_3, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf81, arg59_1, buf84, 2097152, grid=grid(2097152), stream=stream0)
        del arg59_1
        buf85 = reinterpret_tensor(buf81, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [query_states_7, key_states_3, value_states_3, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf82, arg61_1, buf85, 2097152, grid=grid(2097152), stream=stream0)
        del arg61_1
        del buf82
        # Topologically Sorted Source Nodes: [query_states_7, key_states_3, value_states_3, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf86 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf83, buf84, buf85, None, False)
        buf87 = buf86[0]
        del buf86
        buf91 = reinterpret_tensor(buf85, (2048, 1024), (1024, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg62_1, (1024, 1024), (1, 1024), 0), out=buf91)
        del arg62_1
        buf95 = reinterpret_tensor(buf87, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf75, buf91, arg63_1, arg64_1, arg65_1, buf95, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg64_1
        del arg65_1
        buf96 = reinterpret_tensor(buf73, (2048, 4096), (4096, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg66_1, (1024, 4096), (1, 1024), 0), out=buf96)
        del arg66_1
        buf97 = reinterpret_tensor(buf96, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_34], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf97, arg67_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg67_1
        buf98 = reinterpret_tensor(buf95, (2048, 1024), (1024, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg68_1, (4096, 1024), (1, 4096), 0), out=buf98)
        del arg68_1
        buf99 = reinterpret_tensor(buf98, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf98  # reuse
        buf103 = reinterpret_tensor(buf84, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_38, hidden_states_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf99, buf75, buf91, arg63_1, arg69_1, arg70_1, arg71_1, buf103, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg63_1
        del arg69_1
        del arg70_1
        del arg71_1
        buf104 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg72_1, (1024, 1024), (1, 1024), 0), out=buf104)
        del arg72_1
        buf105 = reinterpret_tensor(buf75, (2048, 1024), (1024, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg74_1, (1024, 1024), (1, 1024), 0), out=buf105)
        del arg74_1
        buf106 = reinterpret_tensor(buf83, (2048, 1024), (1024, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg76_1, (1024, 1024), (1, 1024), 0), out=buf106)
        del arg76_1
        buf107 = reinterpret_tensor(buf103, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [query_states_9, key_states_4, value_states_4, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf104, arg73_1, buf107, 2097152, grid=grid(2097152), stream=stream0)
        del arg73_1
        buf108 = reinterpret_tensor(buf104, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [query_states_9, key_states_4, value_states_4, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf105, arg75_1, buf108, 2097152, grid=grid(2097152), stream=stream0)
        del arg75_1
        buf109 = reinterpret_tensor(buf105, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [query_states_9, key_states_4, value_states_4, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf106, arg77_1, buf109, 2097152, grid=grid(2097152), stream=stream0)
        del arg77_1
        del buf106
        # Topologically Sorted Source Nodes: [query_states_9, key_states_4, value_states_4, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf110 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf107, buf108, buf109, None, False)
        buf111 = buf110[0]
        del buf110
        buf115 = reinterpret_tensor(buf109, (2048, 1024), (1024, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg78_1, (1024, 1024), (1, 1024), 0), out=buf115)
        del arg78_1
        buf119 = reinterpret_tensor(buf111, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_41, hidden_states_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf99, buf115, arg79_1, arg80_1, arg81_1, buf119, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg80_1
        del arg81_1
        buf120 = reinterpret_tensor(buf97, (2048, 4096), (4096, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg82_1, (1024, 4096), (1, 1024), 0), out=buf120)
        del arg82_1
        buf121 = reinterpret_tensor(buf120, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_43], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf121, arg83_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg83_1
        buf122 = reinterpret_tensor(buf119, (2048, 1024), (1024, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg84_1, (4096, 1024), (1, 4096), 0), out=buf122)
        del arg84_1
        buf123 = reinterpret_tensor(buf122, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf122  # reuse
        buf127 = reinterpret_tensor(buf108, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_41, hidden_states_47, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf123, buf99, buf115, arg79_1, arg85_1, arg86_1, arg87_1, buf127, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg79_1
        del arg85_1
        del arg86_1
        del arg87_1
        buf128 = reinterpret_tensor(buf99, (2048, 1024), (1024, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg88_1, (1024, 1024), (1, 1024), 0), out=buf128)
        del arg88_1
        buf129 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg90_1, (1024, 1024), (1, 1024), 0), out=buf129)
        del arg90_1
        buf130 = reinterpret_tensor(buf107, (2048, 1024), (1024, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg92_1, (1024, 1024), (1, 1024), 0), out=buf130)
        del arg92_1
        buf131 = reinterpret_tensor(buf127, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [query_states_11, key_states_5, value_states_5, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf128, arg89_1, buf131, 2097152, grid=grid(2097152), stream=stream0)
        del arg89_1
        buf132 = reinterpret_tensor(buf128, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [query_states_11, key_states_5, value_states_5, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf129, arg91_1, buf132, 2097152, grid=grid(2097152), stream=stream0)
        del arg91_1
        buf133 = reinterpret_tensor(buf129, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [query_states_11, key_states_5, value_states_5, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf130, arg93_1, buf133, 2097152, grid=grid(2097152), stream=stream0)
        del arg93_1
        del buf130
        # Topologically Sorted Source Nodes: [query_states_11, key_states_5, value_states_5, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf134 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf131, buf132, buf133, None, False)
        buf135 = buf134[0]
        del buf134
        buf139 = reinterpret_tensor(buf133, (2048, 1024), (1024, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg94_1, (1024, 1024), (1, 1024), 0), out=buf139)
        del arg94_1
        buf143 = reinterpret_tensor(buf135, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_51], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf123, buf139, arg95_1, arg96_1, arg97_1, buf143, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg96_1
        del arg97_1
        buf144 = reinterpret_tensor(buf121, (2048, 4096), (4096, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg98_1, (1024, 4096), (1, 1024), 0), out=buf144)
        del arg98_1
        buf145 = reinterpret_tensor(buf144, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_52], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf145, arg99_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg99_1
        buf146 = reinterpret_tensor(buf143, (2048, 1024), (1024, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg100_1, (4096, 1024), (1, 4096), 0), out=buf146)
        del arg100_1
        buf147 = reinterpret_tensor(buf146, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf146  # reuse
        buf151 = reinterpret_tensor(buf132, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_56, hidden_states_57], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf147, buf123, buf139, arg95_1, arg101_1, arg102_1, arg103_1, buf151, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg101_1
        del arg102_1
        del arg103_1
        del arg95_1
        buf152 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg104_1, (1024, 1024), (1, 1024), 0), out=buf152)
        del arg104_1
        buf153 = reinterpret_tensor(buf123, (2048, 1024), (1024, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg106_1, (1024, 1024), (1, 1024), 0), out=buf153)
        del arg106_1
        buf154 = reinterpret_tensor(buf131, (2048, 1024), (1024, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg108_1, (1024, 1024), (1, 1024), 0), out=buf154)
        del arg108_1
        buf155 = reinterpret_tensor(buf151, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [query_states_13, key_states_6, value_states_6, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf152, arg105_1, buf155, 2097152, grid=grid(2097152), stream=stream0)
        del arg105_1
        buf156 = reinterpret_tensor(buf152, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [query_states_13, key_states_6, value_states_6, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf153, arg107_1, buf156, 2097152, grid=grid(2097152), stream=stream0)
        del arg107_1
        buf157 = reinterpret_tensor(buf153, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [query_states_13, key_states_6, value_states_6, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf154, arg109_1, buf157, 2097152, grid=grid(2097152), stream=stream0)
        del arg109_1
        del buf154
        # Topologically Sorted Source Nodes: [query_states_13, key_states_6, value_states_6, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf158 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf155, buf156, buf157, None, False)
        buf159 = buf158[0]
        del buf158
        buf163 = reinterpret_tensor(buf157, (2048, 1024), (1024, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg110_1, (1024, 1024), (1, 1024), 0), out=buf163)
        del arg110_1
        buf167 = reinterpret_tensor(buf159, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf147, buf163, arg111_1, arg112_1, arg113_1, buf167, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg112_1
        del arg113_1
        buf168 = reinterpret_tensor(buf145, (2048, 4096), (4096, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg114_1, (1024, 4096), (1, 1024), 0), out=buf168)
        del arg114_1
        buf169 = reinterpret_tensor(buf168, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf169, arg115_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg115_1
        buf170 = reinterpret_tensor(buf167, (2048, 1024), (1024, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg116_1, (4096, 1024), (1, 4096), 0), out=buf170)
        del arg116_1
        buf171 = reinterpret_tensor(buf170, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf170  # reuse
        buf175 = reinterpret_tensor(buf156, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_65, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf171, buf147, buf163, arg111_1, arg117_1, arg118_1, arg119_1, buf175, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg111_1
        del arg117_1
        del arg118_1
        del arg119_1
        buf176 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf175, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg120_1, (1024, 1024), (1, 1024), 0), out=buf176)
        del arg120_1
        buf177 = reinterpret_tensor(buf147, (2048, 1024), (1024, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf175, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg122_1, (1024, 1024), (1, 1024), 0), out=buf177)
        del arg122_1
        buf178 = reinterpret_tensor(buf155, (2048, 1024), (1024, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf175, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg124_1, (1024, 1024), (1, 1024), 0), out=buf178)
        del arg124_1
        buf179 = reinterpret_tensor(buf175, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [query_states_15, key_states_7, value_states_7, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf176, arg121_1, buf179, 2097152, grid=grid(2097152), stream=stream0)
        del arg121_1
        buf180 = reinterpret_tensor(buf176, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [query_states_15, key_states_7, value_states_7, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf177, arg123_1, buf180, 2097152, grid=grid(2097152), stream=stream0)
        del arg123_1
        buf181 = reinterpret_tensor(buf177, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [query_states_15, key_states_7, value_states_7, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf178, arg125_1, buf181, 2097152, grid=grid(2097152), stream=stream0)
        del arg125_1
        del buf178
        # Topologically Sorted Source Nodes: [query_states_15, key_states_7, value_states_7, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf182 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf179, buf180, buf181, None, False)
        buf183 = buf182[0]
        del buf182
        buf187 = reinterpret_tensor(buf181, (2048, 1024), (1024, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf183, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg126_1, (1024, 1024), (1, 1024), 0), out=buf187)
        del arg126_1
        buf191 = reinterpret_tensor(buf183, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_68, hidden_states_69], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf171, buf187, arg127_1, arg128_1, arg129_1, buf191, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg128_1
        del arg129_1
        buf192 = reinterpret_tensor(buf169, (2048, 4096), (4096, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg130_1, (1024, 4096), (1, 1024), 0), out=buf192)
        del arg130_1
        buf193 = reinterpret_tensor(buf192, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf193, arg131_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg131_1
        buf194 = reinterpret_tensor(buf191, (2048, 1024), (1024, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf193, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg132_1, (4096, 1024), (1, 4096), 0), out=buf194)
        del arg132_1
        buf195 = reinterpret_tensor(buf194, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf194  # reuse
        buf199 = reinterpret_tensor(buf180, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_68, hidden_states_74, hidden_states_75], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf195, buf171, buf187, arg127_1, arg133_1, arg134_1, arg135_1, buf199, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg127_1
        del arg133_1
        del arg134_1
        del arg135_1
        buf200 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg136_1, (1024, 1024), (1, 1024), 0), out=buf200)
        del arg136_1
        buf201 = reinterpret_tensor(buf171, (2048, 1024), (1024, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg138_1, (1024, 1024), (1, 1024), 0), out=buf201)
        del arg138_1
        buf202 = reinterpret_tensor(buf179, (2048, 1024), (1024, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg140_1, (1024, 1024), (1, 1024), 0), out=buf202)
        del arg140_1
        buf203 = reinterpret_tensor(buf199, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [query_states_17, key_states_8, value_states_8, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf200, arg137_1, buf203, 2097152, grid=grid(2097152), stream=stream0)
        del arg137_1
        buf204 = reinterpret_tensor(buf200, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [query_states_17, key_states_8, value_states_8, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf201, arg139_1, buf204, 2097152, grid=grid(2097152), stream=stream0)
        del arg139_1
        buf205 = reinterpret_tensor(buf201, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [query_states_17, key_states_8, value_states_8, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf202, arg141_1, buf205, 2097152, grid=grid(2097152), stream=stream0)
        del arg141_1
        del buf202
        # Topologically Sorted Source Nodes: [query_states_17, key_states_8, value_states_8, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf206 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf203, buf204, buf205, None, False)
        buf207 = buf206[0]
        del buf206
        buf211 = reinterpret_tensor(buf205, (2048, 1024), (1024, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg142_1, (1024, 1024), (1, 1024), 0), out=buf211)
        del arg142_1
        buf215 = reinterpret_tensor(buf207, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77, hidden_states_78], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf195, buf211, arg143_1, arg144_1, arg145_1, buf215, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg144_1
        del arg145_1
        buf216 = reinterpret_tensor(buf193, (2048, 4096), (4096, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg146_1, (1024, 4096), (1, 1024), 0), out=buf216)
        del arg146_1
        buf217 = reinterpret_tensor(buf216, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_79], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf217, arg147_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg147_1
        buf218 = reinterpret_tensor(buf215, (2048, 1024), (1024, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg148_1, (4096, 1024), (1, 4096), 0), out=buf218)
        del arg148_1
        buf219 = reinterpret_tensor(buf218, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf218  # reuse
        buf223 = reinterpret_tensor(buf204, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77, hidden_states_83, hidden_states_84], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf219, buf195, buf211, arg143_1, arg149_1, arg150_1, arg151_1, buf223, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg143_1
        del arg149_1
        del arg150_1
        del arg151_1
        buf224 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg152_1, (1024, 1024), (1, 1024), 0), out=buf224)
        del arg152_1
        buf225 = reinterpret_tensor(buf195, (2048, 1024), (1024, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg154_1, (1024, 1024), (1, 1024), 0), out=buf225)
        del arg154_1
        buf226 = reinterpret_tensor(buf203, (2048, 1024), (1024, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg156_1, (1024, 1024), (1, 1024), 0), out=buf226)
        del arg156_1
        buf227 = reinterpret_tensor(buf223, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [query_states_19, key_states_9, value_states_9, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf224, arg153_1, buf227, 2097152, grid=grid(2097152), stream=stream0)
        del arg153_1
        buf228 = reinterpret_tensor(buf224, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [query_states_19, key_states_9, value_states_9, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf225, arg155_1, buf228, 2097152, grid=grid(2097152), stream=stream0)
        del arg155_1
        buf229 = reinterpret_tensor(buf225, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [query_states_19, key_states_9, value_states_9, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf226, arg157_1, buf229, 2097152, grid=grid(2097152), stream=stream0)
        del arg157_1
        del buf226
        # Topologically Sorted Source Nodes: [query_states_19, key_states_9, value_states_9, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf230 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf227, buf228, buf229, None, False)
        buf231 = buf230[0]
        del buf230
        buf235 = reinterpret_tensor(buf229, (2048, 1024), (1024, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg158_1, (1024, 1024), (1, 1024), 0), out=buf235)
        del arg158_1
        buf239 = reinterpret_tensor(buf231, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_86, hidden_states_87], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf219, buf235, arg159_1, arg160_1, arg161_1, buf239, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg160_1
        del arg161_1
        buf240 = reinterpret_tensor(buf217, (2048, 4096), (4096, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg162_1, (1024, 4096), (1, 1024), 0), out=buf240)
        del arg162_1
        buf241 = reinterpret_tensor(buf240, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf241, arg163_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg163_1
        buf242 = reinterpret_tensor(buf239, (2048, 1024), (1024, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf241, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg164_1, (4096, 1024), (1, 4096), 0), out=buf242)
        del arg164_1
        buf243 = reinterpret_tensor(buf242, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf242  # reuse
        buf247 = reinterpret_tensor(buf228, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_86, hidden_states_92, hidden_states_93], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf243, buf219, buf235, arg159_1, arg165_1, arg166_1, arg167_1, buf247, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg159_1
        del arg165_1
        del arg166_1
        del arg167_1
        buf248 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg168_1, (1024, 1024), (1, 1024), 0), out=buf248)
        del arg168_1
        buf249 = reinterpret_tensor(buf219, (2048, 1024), (1024, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg170_1, (1024, 1024), (1, 1024), 0), out=buf249)
        del arg170_1
        buf250 = reinterpret_tensor(buf227, (2048, 1024), (1024, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg172_1, (1024, 1024), (1, 1024), 0), out=buf250)
        del arg172_1
        buf251 = reinterpret_tensor(buf247, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [query_states_21, key_states_10, value_states_10, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf248, arg169_1, buf251, 2097152, grid=grid(2097152), stream=stream0)
        del arg169_1
        buf252 = reinterpret_tensor(buf248, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [query_states_21, key_states_10, value_states_10, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf249, arg171_1, buf252, 2097152, grid=grid(2097152), stream=stream0)
        del arg171_1
        buf253 = reinterpret_tensor(buf249, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [query_states_21, key_states_10, value_states_10, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf250, arg173_1, buf253, 2097152, grid=grid(2097152), stream=stream0)
        del arg173_1
        del buf250
        # Topologically Sorted Source Nodes: [query_states_21, key_states_10, value_states_10, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf254 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf251, buf252, buf253, None, False)
        buf255 = buf254[0]
        del buf254
        buf259 = reinterpret_tensor(buf253, (2048, 1024), (1024, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf255, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg174_1, (1024, 1024), (1, 1024), 0), out=buf259)
        del arg174_1
        buf263 = reinterpret_tensor(buf255, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_95, hidden_states_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf243, buf259, arg175_1, arg176_1, arg177_1, buf263, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg176_1
        del arg177_1
        buf264 = reinterpret_tensor(buf241, (2048, 4096), (4096, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf263, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg178_1, (1024, 4096), (1, 1024), 0), out=buf264)
        del arg178_1
        buf265 = reinterpret_tensor(buf264, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_97], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf265, arg179_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg179_1
        buf266 = reinterpret_tensor(buf263, (2048, 1024), (1024, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf265, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg180_1, (4096, 1024), (1, 4096), 0), out=buf266)
        del arg180_1
        buf267 = reinterpret_tensor(buf266, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf266  # reuse
        buf271 = reinterpret_tensor(buf252, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_95, hidden_states_101, hidden_states_102], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf267, buf243, buf259, arg175_1, arg181_1, arg182_1, arg183_1, buf271, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg175_1
        del arg181_1
        del arg182_1
        del arg183_1
        buf272 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg184_1, (1024, 1024), (1, 1024), 0), out=buf272)
        del arg184_1
        buf273 = reinterpret_tensor(buf243, (2048, 1024), (1024, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg186_1, (1024, 1024), (1, 1024), 0), out=buf273)
        del arg186_1
        buf274 = reinterpret_tensor(buf251, (2048, 1024), (1024, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg188_1, (1024, 1024), (1, 1024), 0), out=buf274)
        del arg188_1
        buf275 = reinterpret_tensor(buf271, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [query_states_23, key_states_11, value_states_11, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf272, arg185_1, buf275, 2097152, grid=grid(2097152), stream=stream0)
        del arg185_1
        buf276 = reinterpret_tensor(buf272, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [query_states_23, key_states_11, value_states_11, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf273, arg187_1, buf276, 2097152, grid=grid(2097152), stream=stream0)
        del arg187_1
        buf277 = reinterpret_tensor(buf273, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [query_states_23, key_states_11, value_states_11, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf274, arg189_1, buf277, 2097152, grid=grid(2097152), stream=stream0)
        del arg189_1
        # Topologically Sorted Source Nodes: [query_states_23, key_states_11, value_states_11, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf278 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf275, buf276, buf277, None, False)
        buf279 = buf278[0]
        del buf278
        buf283 = reinterpret_tensor(buf277, (2048, 1024), (1024, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf279, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg190_1, (1024, 1024), (1, 1024), 0), out=buf283)
        del arg190_1
        buf287 = reinterpret_tensor(buf279, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_104, hidden_states_105], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf267, buf283, arg191_1, arg192_1, arg193_1, buf287, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg192_1
        del arg193_1
        buf288 = reinterpret_tensor(buf265, (2048, 4096), (4096, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf287, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg194_1, (1024, 4096), (1, 1024), 0), out=buf288)
        del arg194_1
        buf289 = reinterpret_tensor(buf288, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_106], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf289, arg195_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg195_1
        buf290 = reinterpret_tensor(buf287, (2048, 1024), (1024, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf289, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg196_1, (4096, 1024), (1, 4096), 0), out=buf290)
        del arg196_1
        buf291 = reinterpret_tensor(buf290, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf290  # reuse
        buf323 = reinterpret_tensor(buf276, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_104, hidden_states_110, hidden_states_111], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf291, buf267, buf283, arg191_1, arg197_1, arg198_1, arg199_1, buf323, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg191_1
        del arg197_1
        del arg198_1
        del arg199_1
        buf295 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [eq, masked_fill_, ne, sum_1], Original ATen: [aten.eq, aten.masked_fill, aten.ne, aten.sum]
        triton_per_fused_eq_masked_fill_ne_sum_5.run(arg0_1, buf295, 2, 1024, grid=grid(2), stream=stream0)
        buf296 = buf291; del buf291  # reuse
        buf300 = reinterpret_tensor(buf283, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf283  # reuse
        buf304 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [eq, masked_fill_, clone_1, setitem, setitem_1, embedding_2, inputs_embeds_1, add_27, positions_2, hidden_states_112, hidden_states_113, hidden_states_115], Original ATen: [aten.eq, aten.masked_fill, aten.clone, aten.copy, aten.view, aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_view_6.run(buf295, arg0_1, arg2_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, buf296, buf300, buf304, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg200_1
        del arg201_1
        del arg202_1
        del arg203_1
        del arg204_1
        del buf295
        buf305 = reinterpret_tensor(buf296, (2048, 1024), (1024, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg205_1, (1024, 1024), (1, 1024), 0), out=buf305)
        del arg205_1
        buf306 = reinterpret_tensor(buf275, (2048, 1024), (1024, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg207_1, (1024, 1024), (1, 1024), 0), out=buf306)
        del arg207_1
        buf307 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg209_1, (1024, 1024), (1, 1024), 0), out=buf307)
        del arg209_1
        buf308 = reinterpret_tensor(buf304, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [query_states_25, key_states_12, value_states_12, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf305, arg206_1, buf308, 2097152, grid=grid(2097152), stream=stream0)
        del arg206_1
        buf309 = reinterpret_tensor(buf305, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [query_states_25, key_states_12, value_states_12, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf306, arg208_1, buf309, 2097152, grid=grid(2097152), stream=stream0)
        del arg208_1
        buf310 = reinterpret_tensor(buf306, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [query_states_25, key_states_12, value_states_12, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf307, arg210_1, buf310, 2097152, grid=grid(2097152), stream=stream0)
        del arg210_1
        buf311 = empty_strided_cuda((2, 16, 1024, 1024), (16777216, 1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [query_states_25, key_states_12, value_states_12, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_7.run(buf311, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_25, key_states_12, value_states_12, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf312 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf308, buf309, buf310, buf311, False)
        buf313 = buf312[0]
        del buf312
        buf317 = reinterpret_tensor(buf310, (2048, 1024), (1024, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg211_1, (1024, 1024), (1, 1024), 0), out=buf317)
        del arg211_1
        buf321 = reinterpret_tensor(buf313, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_117, hidden_states_118], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf300, buf317, arg212_1, arg213_1, arg214_1, buf321, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg213_1
        del arg214_1
        buf322 = reinterpret_tensor(buf309, (2048, 1024), (1024, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf321, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg215_1, (1024, 1024), (1, 1024), 0), out=buf322)
        del arg215_1
        buf324 = reinterpret_tensor(buf321, (2048, 1024), (1024, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg217_1, (1024, 1024), (1, 1024), 0), out=buf324)
        del arg217_1
        buf325 = reinterpret_tensor(buf308, (2048, 1024), (1024, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg219_1, (1024, 1024), (1, 1024), 0), out=buf325)
        del arg219_1
        buf326 = reinterpret_tensor(buf307, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [query_states_27, key_states_13, value_states_13, attn_output_52], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf322, arg216_1, buf326, 2097152, grid=grid(2097152), stream=stream0)
        del arg216_1
        buf327 = reinterpret_tensor(buf322, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [query_states_27, key_states_13, value_states_13, attn_output_52], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf324, arg218_1, buf327, 2097152, grid=grid(2097152), stream=stream0)
        del arg218_1
        buf328 = reinterpret_tensor(buf324, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [query_states_27, key_states_13, value_states_13, attn_output_52], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf325, arg220_1, buf328, 2097152, grid=grid(2097152), stream=stream0)
        del arg220_1
        del buf325
        # Topologically Sorted Source Nodes: [query_states_27, key_states_13, value_states_13, attn_output_52], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf329 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf326, buf327, buf328, None, False)
        buf330 = buf329[0]
        del buf329
        buf334 = reinterpret_tensor(buf328, (2048, 1024), (1024, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf330, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg221_1, (1024, 1024), (1, 1024), 0), out=buf334)
        del arg221_1
        buf335 = reinterpret_tensor(buf334, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf334  # reuse
        buf339 = reinterpret_tensor(buf330, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_117, hidden_states_120, hidden_states_121], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf335, buf300, buf317, arg212_1, arg222_1, arg223_1, arg224_1, buf339, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg212_1
        del arg222_1
        del arg223_1
        del arg224_1
        buf340 = reinterpret_tensor(buf289, (2048, 4096), (4096, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf339, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg225_1, (1024, 4096), (1, 1024), 0), out=buf340)
        del arg225_1
        buf341 = reinterpret_tensor(buf340, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_122], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf341, arg226_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg226_1
        buf342 = reinterpret_tensor(buf339, (2048, 1024), (1024, 1), 0); del buf339  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf341, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg227_1, (4096, 1024), (1, 4096), 0), out=buf342)
        del arg227_1
        buf346 = reinterpret_tensor(buf317, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_126, hidden_states_127], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf335, buf342, arg228_1, arg229_1, arg230_1, buf346, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg229_1
        del arg230_1
        buf347 = reinterpret_tensor(buf300, (2048, 1024), (1024, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg231_1, (1024, 1024), (1, 1024), 0), out=buf347)
        del arg231_1
        buf348 = reinterpret_tensor(buf327, (2048, 1024), (1024, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg233_1, (1024, 1024), (1, 1024), 0), out=buf348)
        del arg233_1
        buf349 = reinterpret_tensor(buf326, (2048, 1024), (1024, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg235_1, (1024, 1024), (1, 1024), 0), out=buf349)
        del arg235_1
        buf350 = reinterpret_tensor(buf346, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [query_states_29, key_states_14, value_states_14, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf347, arg232_1, buf350, 2097152, grid=grid(2097152), stream=stream0)
        del arg232_1
        buf351 = reinterpret_tensor(buf347, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [query_states_29, key_states_14, value_states_14, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf348, arg234_1, buf351, 2097152, grid=grid(2097152), stream=stream0)
        del arg234_1
        buf352 = reinterpret_tensor(buf348, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf348  # reuse
        # Topologically Sorted Source Nodes: [query_states_29, key_states_14, value_states_14, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf349, arg236_1, buf352, 2097152, grid=grid(2097152), stream=stream0)
        del arg236_1
        del buf349
        buf353 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [query_states_29, key_states_14, value_states_14, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_7.run(buf353, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_29, key_states_14, value_states_14, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf354 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf350, buf351, buf352, buf353, False)
        del buf350
        buf355 = buf354[0]
        del buf354
        buf359 = reinterpret_tensor(buf352, (2048, 1024), (1024, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf355, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg237_1, (1024, 1024), (1, 1024), 0), out=buf359)
        del arg237_1
        buf360 = reinterpret_tensor(buf359, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf359  # reuse
        buf364 = reinterpret_tensor(buf355, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf355  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_126, hidden_states_129, hidden_states_130], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf360, buf335, buf342, arg228_1, arg238_1, arg239_1, arg240_1, buf364, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg228_1
        del arg238_1
        del arg239_1
        del arg240_1
        buf365 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg241_1, (1024, 1024), (1, 1024), 0), out=buf365)
        del arg241_1
        buf366 = reinterpret_tensor(buf364, (2048, 1024), (1024, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg243_1, (1024, 1024), (1, 1024), 0), out=buf366)
        del arg243_1
        buf367 = reinterpret_tensor(buf335, (2048, 1024), (1024, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg245_1, (1024, 1024), (1, 1024), 0), out=buf367)
        del arg245_1
        buf368 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [query_states_31, key_states_15, value_states_15, attn_output_60], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf365, arg242_1, buf368, 2097152, grid=grid(2097152), stream=stream0)
        del arg242_1
        buf369 = reinterpret_tensor(buf365, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [query_states_31, key_states_15, value_states_15, attn_output_60], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf366, arg244_1, buf369, 2097152, grid=grid(2097152), stream=stream0)
        del arg244_1
        buf370 = reinterpret_tensor(buf366, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [query_states_31, key_states_15, value_states_15, attn_output_60], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf367, arg246_1, buf370, 2097152, grid=grid(2097152), stream=stream0)
        del arg246_1
        del buf367
        # Topologically Sorted Source Nodes: [query_states_31, key_states_15, value_states_15, attn_output_60], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf371 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf368, buf369, buf370, None, False)
        buf372 = buf371[0]
        del buf371
        buf376 = reinterpret_tensor(buf370, (2048, 1024), (1024, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf372, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg247_1, (1024, 1024), (1, 1024), 0), out=buf376)
        del arg247_1
        buf380 = reinterpret_tensor(buf372, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_132, hidden_states_133], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf360, buf376, arg248_1, arg249_1, arg250_1, buf380, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg249_1
        del arg250_1
        buf381 = reinterpret_tensor(buf341, (2048, 4096), (4096, 1), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf380, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg251_1, (1024, 4096), (1, 1024), 0), out=buf381)
        del arg251_1
        buf382 = reinterpret_tensor(buf381, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_134], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf382, arg252_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg252_1
        buf383 = reinterpret_tensor(buf380, (2048, 1024), (1024, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf382, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg253_1, (4096, 1024), (1, 4096), 0), out=buf383)
        del arg253_1
        buf384 = reinterpret_tensor(buf383, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf383  # reuse
        buf388 = reinterpret_tensor(buf369, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf369  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_132, hidden_states_138, hidden_states_139], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf384, buf360, buf376, arg248_1, arg254_1, arg255_1, arg256_1, buf388, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg248_1
        del arg254_1
        del arg255_1
        del arg256_1
        buf389 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf388, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg257_1, (1024, 1024), (1, 1024), 0), out=buf389)
        del arg257_1
        buf390 = reinterpret_tensor(buf360, (2048, 1024), (1024, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf388, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg259_1, (1024, 1024), (1, 1024), 0), out=buf390)
        del arg259_1
        buf391 = reinterpret_tensor(buf368, (2048, 1024), (1024, 1), 0); del buf368  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf388, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg261_1, (1024, 1024), (1, 1024), 0), out=buf391)
        del arg261_1
        buf392 = reinterpret_tensor(buf388, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [query_states_33, key_states_16, value_states_16, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf389, arg258_1, buf392, 2097152, grid=grid(2097152), stream=stream0)
        del arg258_1
        buf393 = reinterpret_tensor(buf389, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [query_states_33, key_states_16, value_states_16, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf390, arg260_1, buf393, 2097152, grid=grid(2097152), stream=stream0)
        del arg260_1
        buf394 = reinterpret_tensor(buf390, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf390  # reuse
        # Topologically Sorted Source Nodes: [query_states_33, key_states_16, value_states_16, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf391, arg262_1, buf394, 2097152, grid=grid(2097152), stream=stream0)
        del arg262_1
        buf395 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [query_states_33, key_states_16, value_states_16, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_7.run(buf395, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_33, key_states_16, value_states_16, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf396 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf392, buf393, buf394, buf395, False)
        buf397 = buf396[0]
        del buf396
        buf401 = reinterpret_tensor(buf394, (2048, 1024), (1024, 1), 0); del buf394  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf397, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg263_1, (1024, 1024), (1, 1024), 0), out=buf401)
        del arg263_1
        buf405 = reinterpret_tensor(buf397, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_141, hidden_states_142], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf384, buf401, arg264_1, arg265_1, arg266_1, buf405, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg265_1
        del arg266_1
        buf406 = reinterpret_tensor(buf393, (2048, 1024), (1024, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf405, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg267_1, (1024, 1024), (1, 1024), 0), out=buf406)
        del arg267_1
        buf407 = reinterpret_tensor(buf405, (2048, 1024), (1024, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg269_1, (1024, 1024), (1, 1024), 0), out=buf407)
        del arg269_1
        buf408 = reinterpret_tensor(buf392, (2048, 1024), (1024, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg271_1, (1024, 1024), (1, 1024), 0), out=buf408)
        del arg271_1
        buf409 = reinterpret_tensor(buf391, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [query_states_35, key_states_17, value_states_17, attn_output_68], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf406, arg268_1, buf409, 2097152, grid=grid(2097152), stream=stream0)
        del arg268_1
        buf410 = reinterpret_tensor(buf406, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [query_states_35, key_states_17, value_states_17, attn_output_68], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf407, arg270_1, buf410, 2097152, grid=grid(2097152), stream=stream0)
        del arg270_1
        buf411 = reinterpret_tensor(buf407, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [query_states_35, key_states_17, value_states_17, attn_output_68], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf408, arg272_1, buf411, 2097152, grid=grid(2097152), stream=stream0)
        del arg272_1
        del buf408
        # Topologically Sorted Source Nodes: [query_states_35, key_states_17, value_states_17, attn_output_68], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf412 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf409, buf410, buf411, None, False)
        buf413 = buf412[0]
        del buf412
        buf417 = reinterpret_tensor(buf411, (2048, 1024), (1024, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf413, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg273_1, (1024, 1024), (1, 1024), 0), out=buf417)
        del arg273_1
        buf418 = reinterpret_tensor(buf417, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf417  # reuse
        buf422 = reinterpret_tensor(buf413, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_141, hidden_states_144, hidden_states_145], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf418, buf384, buf401, arg264_1, arg274_1, arg275_1, arg276_1, buf422, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg264_1
        del arg274_1
        del arg275_1
        del arg276_1
        buf423 = reinterpret_tensor(buf382, (2048, 4096), (4096, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf422, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg277_1, (1024, 4096), (1, 1024), 0), out=buf423)
        del arg277_1
        buf424 = reinterpret_tensor(buf423, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_146], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf424, arg278_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg278_1
        buf425 = reinterpret_tensor(buf422, (2048, 1024), (1024, 1), 0); del buf422  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf424, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg279_1, (4096, 1024), (1, 4096), 0), out=buf425)
        del arg279_1
        buf429 = reinterpret_tensor(buf401, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf418, buf425, arg280_1, arg281_1, arg282_1, buf429, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg281_1
        del arg282_1
        buf430 = reinterpret_tensor(buf384, (2048, 1024), (1024, 1), 0); del buf384  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf429, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg283_1, (1024, 1024), (1, 1024), 0), out=buf430)
        del arg283_1
        buf431 = reinterpret_tensor(buf410, (2048, 1024), (1024, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf429, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg285_1, (1024, 1024), (1, 1024), 0), out=buf431)
        del arg285_1
        buf432 = reinterpret_tensor(buf409, (2048, 1024), (1024, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf429, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg287_1, (1024, 1024), (1, 1024), 0), out=buf432)
        del arg287_1
        buf433 = reinterpret_tensor(buf429, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [query_states_37, key_states_18, value_states_18, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf430, arg284_1, buf433, 2097152, grid=grid(2097152), stream=stream0)
        del arg284_1
        buf434 = reinterpret_tensor(buf430, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf430  # reuse
        # Topologically Sorted Source Nodes: [query_states_37, key_states_18, value_states_18, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf431, arg286_1, buf434, 2097152, grid=grid(2097152), stream=stream0)
        del arg286_1
        buf435 = reinterpret_tensor(buf431, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf431  # reuse
        # Topologically Sorted Source Nodes: [query_states_37, key_states_18, value_states_18, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf432, arg288_1, buf435, 2097152, grid=grid(2097152), stream=stream0)
        del arg288_1
        del buf432
        buf436 = buf395; del buf395  # reuse
        # Topologically Sorted Source Nodes: [query_states_37, key_states_18, value_states_18, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_7.run(buf436, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_37, key_states_18, value_states_18, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf437 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf433, buf434, buf435, buf436, False)
        del buf433
        buf438 = buf437[0]
        del buf437
        buf442 = reinterpret_tensor(buf435, (2048, 1024), (1024, 1), 0); del buf435  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf438, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg289_1, (1024, 1024), (1, 1024), 0), out=buf442)
        del arg289_1
        buf443 = reinterpret_tensor(buf442, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf442  # reuse
        buf447 = reinterpret_tensor(buf438, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf438  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_153, hidden_states_154], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf443, buf418, buf425, arg280_1, arg290_1, arg291_1, arg292_1, buf447, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg280_1
        del arg290_1
        del arg291_1
        del arg292_1
        buf448 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf447, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg293_1, (1024, 1024), (1, 1024), 0), out=buf448)
        del arg293_1
        buf449 = reinterpret_tensor(buf447, (2048, 1024), (1024, 1), 0); del buf447  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg295_1, (1024, 1024), (1, 1024), 0), out=buf449)
        del arg295_1
        buf450 = reinterpret_tensor(buf418, (2048, 1024), (1024, 1), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg297_1, (1024, 1024), (1, 1024), 0), out=buf450)
        del arg297_1
        buf451 = buf434; del buf434  # reuse
        # Topologically Sorted Source Nodes: [query_states_39, key_states_19, value_states_19, attn_output_76], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf448, arg294_1, buf451, 2097152, grid=grid(2097152), stream=stream0)
        del arg294_1
        buf452 = reinterpret_tensor(buf448, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [query_states_39, key_states_19, value_states_19, attn_output_76], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf449, arg296_1, buf452, 2097152, grid=grid(2097152), stream=stream0)
        del arg296_1
        buf453 = reinterpret_tensor(buf449, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [query_states_39, key_states_19, value_states_19, attn_output_76], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf450, arg298_1, buf453, 2097152, grid=grid(2097152), stream=stream0)
        del arg298_1
        del buf450
        # Topologically Sorted Source Nodes: [query_states_39, key_states_19, value_states_19, attn_output_76], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf454 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf451, buf452, buf453, None, False)
        buf455 = buf454[0]
        del buf454
        buf459 = reinterpret_tensor(buf453, (2048, 1024), (1024, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf455, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg299_1, (1024, 1024), (1, 1024), 0), out=buf459)
        del arg299_1
        buf463 = reinterpret_tensor(buf455, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf455  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_156, hidden_states_157], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf443, buf459, arg300_1, arg301_1, arg302_1, buf463, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg301_1
        del arg302_1
        buf464 = reinterpret_tensor(buf424, (2048, 4096), (4096, 1), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf463, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg303_1, (1024, 4096), (1, 1024), 0), out=buf464)
        del arg303_1
        buf465 = reinterpret_tensor(buf464, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf464  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_158], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf465, arg304_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg304_1
        buf466 = reinterpret_tensor(buf463, (2048, 1024), (1024, 1), 0); del buf463  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf465, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg305_1, (4096, 1024), (1, 4096), 0), out=buf466)
        del arg305_1
        buf467 = reinterpret_tensor(buf466, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf466  # reuse
        buf471 = reinterpret_tensor(buf452, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf452  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_156, hidden_states_162, hidden_states_163], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf467, buf443, buf459, arg300_1, arg306_1, arg307_1, arg308_1, buf471, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg300_1
        del arg306_1
        del arg307_1
        del arg308_1
        buf472 = buf459; del buf459  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf471, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg309_1, (1024, 1024), (1, 1024), 0), out=buf472)
        del arg309_1
        buf473 = reinterpret_tensor(buf443, (2048, 1024), (1024, 1), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf471, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg311_1, (1024, 1024), (1, 1024), 0), out=buf473)
        del arg311_1
        buf474 = reinterpret_tensor(buf451, (2048, 1024), (1024, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf471, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg313_1, (1024, 1024), (1, 1024), 0), out=buf474)
        del arg313_1
        buf475 = reinterpret_tensor(buf471, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [query_states_41, key_states_20, value_states_20, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf472, arg310_1, buf475, 2097152, grid=grid(2097152), stream=stream0)
        del arg310_1
        buf476 = reinterpret_tensor(buf472, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf472  # reuse
        # Topologically Sorted Source Nodes: [query_states_41, key_states_20, value_states_20, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf473, arg312_1, buf476, 2097152, grid=grid(2097152), stream=stream0)
        del arg312_1
        buf477 = reinterpret_tensor(buf473, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf473  # reuse
        # Topologically Sorted Source Nodes: [query_states_41, key_states_20, value_states_20, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf474, arg314_1, buf477, 2097152, grid=grid(2097152), stream=stream0)
        del arg314_1
        buf478 = buf436; del buf436  # reuse
        # Topologically Sorted Source Nodes: [query_states_41, key_states_20, value_states_20, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_7.run(buf478, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_41, key_states_20, value_states_20, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf479 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf475, buf476, buf477, buf478, False)
        buf480 = buf479[0]
        del buf479
        buf484 = reinterpret_tensor(buf477, (2048, 1024), (1024, 1), 0); del buf477  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf480, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg315_1, (1024, 1024), (1, 1024), 0), out=buf484)
        del arg315_1
        buf488 = reinterpret_tensor(buf480, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_165, hidden_states_166], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf467, buf484, arg316_1, arg317_1, arg318_1, buf488, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg317_1
        del arg318_1
        buf489 = reinterpret_tensor(buf476, (2048, 1024), (1024, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf488, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg319_1, (1024, 1024), (1, 1024), 0), out=buf489)
        del arg319_1
        buf490 = reinterpret_tensor(buf488, (2048, 1024), (1024, 1), 0); del buf488  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg321_1, (1024, 1024), (1, 1024), 0), out=buf490)
        del arg321_1
        buf491 = reinterpret_tensor(buf475, (2048, 1024), (1024, 1), 0); del buf475  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg323_1, (1024, 1024), (1, 1024), 0), out=buf491)
        del arg323_1
        buf492 = reinterpret_tensor(buf474, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf474  # reuse
        # Topologically Sorted Source Nodes: [query_states_43, key_states_21, value_states_21, attn_output_84], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf489, arg320_1, buf492, 2097152, grid=grid(2097152), stream=stream0)
        del arg320_1
        buf493 = reinterpret_tensor(buf489, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [query_states_43, key_states_21, value_states_21, attn_output_84], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf490, arg322_1, buf493, 2097152, grid=grid(2097152), stream=stream0)
        del arg322_1
        buf494 = reinterpret_tensor(buf490, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [query_states_43, key_states_21, value_states_21, attn_output_84], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf491, arg324_1, buf494, 2097152, grid=grid(2097152), stream=stream0)
        del arg324_1
        del buf491
        # Topologically Sorted Source Nodes: [query_states_43, key_states_21, value_states_21, attn_output_84], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf495 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf492, buf493, buf494, None, False)
        buf496 = buf495[0]
        del buf495
        buf500 = reinterpret_tensor(buf494, (2048, 1024), (1024, 1), 0); del buf494  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf496, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg325_1, (1024, 1024), (1, 1024), 0), out=buf500)
        del arg325_1
        buf501 = reinterpret_tensor(buf500, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf500  # reuse
        buf505 = reinterpret_tensor(buf496, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf496  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_165, hidden_states_168, hidden_states_169], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf501, buf467, buf484, arg316_1, arg326_1, arg327_1, arg328_1, buf505, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg316_1
        del arg326_1
        del arg327_1
        del arg328_1
        buf506 = reinterpret_tensor(buf465, (2048, 4096), (4096, 1), 0); del buf465  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf505, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg329_1, (1024, 4096), (1, 1024), 0), out=buf506)
        del arg329_1
        buf507 = reinterpret_tensor(buf506, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf506  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_170], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf507, arg330_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg330_1
        buf508 = reinterpret_tensor(buf505, (2048, 1024), (1024, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf507, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg331_1, (4096, 1024), (1, 4096), 0), out=buf508)
        del arg331_1
        buf512 = reinterpret_tensor(buf484, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf484  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_174, hidden_states_175], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf501, buf508, arg332_1, arg333_1, arg334_1, buf512, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg333_1
        del arg334_1
        buf513 = reinterpret_tensor(buf467, (2048, 1024), (1024, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf512, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg335_1, (1024, 1024), (1, 1024), 0), out=buf513)
        del arg335_1
        buf514 = reinterpret_tensor(buf493, (2048, 1024), (1024, 1), 0); del buf493  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf512, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg337_1, (1024, 1024), (1, 1024), 0), out=buf514)
        del arg337_1
        buf515 = reinterpret_tensor(buf492, (2048, 1024), (1024, 1), 0); del buf492  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf512, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg339_1, (1024, 1024), (1, 1024), 0), out=buf515)
        del arg339_1
        buf516 = reinterpret_tensor(buf512, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf512  # reuse
        # Topologically Sorted Source Nodes: [query_states_45, key_states_22, value_states_22, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf513, arg336_1, buf516, 2097152, grid=grid(2097152), stream=stream0)
        del arg336_1
        buf517 = reinterpret_tensor(buf513, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf513  # reuse
        # Topologically Sorted Source Nodes: [query_states_45, key_states_22, value_states_22, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf514, arg338_1, buf517, 2097152, grid=grid(2097152), stream=stream0)
        del arg338_1
        buf518 = reinterpret_tensor(buf514, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [query_states_45, key_states_22, value_states_22, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf515, arg340_1, buf518, 2097152, grid=grid(2097152), stream=stream0)
        del arg340_1
        del buf515
        buf519 = buf478; del buf478  # reuse
        # Topologically Sorted Source Nodes: [query_states_45, key_states_22, value_states_22, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_7.run(buf519, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_45, key_states_22, value_states_22, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf520 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf516, buf517, buf518, buf519, False)
        del buf516
        buf521 = buf520[0]
        del buf520
        buf525 = reinterpret_tensor(buf518, (2048, 1024), (1024, 1), 0); del buf518  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf521, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg341_1, (1024, 1024), (1, 1024), 0), out=buf525)
        del arg341_1
        buf526 = reinterpret_tensor(buf525, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf525  # reuse
        buf530 = reinterpret_tensor(buf521, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf521  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_174, hidden_states_177, hidden_states_178], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf526, buf501, buf508, arg332_1, arg342_1, arg343_1, arg344_1, buf530, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg332_1
        del arg342_1
        del arg343_1
        del arg344_1
        buf531 = buf508; del buf508  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf530, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg345_1, (1024, 1024), (1, 1024), 0), out=buf531)
        del arg345_1
        buf532 = reinterpret_tensor(buf530, (2048, 1024), (1024, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg347_1, (1024, 1024), (1, 1024), 0), out=buf532)
        del arg347_1
        buf533 = reinterpret_tensor(buf501, (2048, 1024), (1024, 1), 0); del buf501  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg349_1, (1024, 1024), (1, 1024), 0), out=buf533)
        del arg349_1
        buf534 = buf517; del buf517  # reuse
        # Topologically Sorted Source Nodes: [query_states_47, key_states_23, value_states_23, attn_output_92], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf531, arg346_1, buf534, 2097152, grid=grid(2097152), stream=stream0)
        del arg346_1
        buf535 = reinterpret_tensor(buf531, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf531  # reuse
        # Topologically Sorted Source Nodes: [query_states_47, key_states_23, value_states_23, attn_output_92], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf532, arg348_1, buf535, 2097152, grid=grid(2097152), stream=stream0)
        del arg348_1
        buf536 = reinterpret_tensor(buf532, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [query_states_47, key_states_23, value_states_23, attn_output_92], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf533, arg350_1, buf536, 2097152, grid=grid(2097152), stream=stream0)
        del arg350_1
        del buf533
        # Topologically Sorted Source Nodes: [query_states_47, key_states_23, value_states_23, attn_output_92], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf537 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf534, buf535, buf536, None, False)
        buf538 = buf537[0]
        del buf537
        buf542 = reinterpret_tensor(buf536, (2048, 1024), (1024, 1), 0); del buf536  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf538, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg351_1, (1024, 1024), (1, 1024), 0), out=buf542)
        del arg351_1
        buf546 = reinterpret_tensor(buf538, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf538  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_180, hidden_states_181], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf526, buf542, arg352_1, arg353_1, arg354_1, buf546, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg353_1
        del arg354_1
        buf547 = reinterpret_tensor(buf507, (2048, 4096), (4096, 1), 0); del buf507  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf546, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg355_1, (1024, 4096), (1, 1024), 0), out=buf547)
        del arg355_1
        buf548 = reinterpret_tensor(buf547, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_182], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf548, arg356_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg356_1
        buf549 = reinterpret_tensor(buf546, (2048, 1024), (1024, 1), 0); del buf546  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf548, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg357_1, (4096, 1024), (1, 4096), 0), out=buf549)
        del arg357_1
        buf550 = reinterpret_tensor(buf549, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf549  # reuse
        buf554 = reinterpret_tensor(buf535, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf535  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_180, hidden_states_186, hidden_states_187], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf550, buf526, buf542, arg352_1, arg358_1, arg359_1, arg360_1, buf554, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg352_1
        del arg358_1
        del arg359_1
        del arg360_1
        buf555 = buf542; del buf542  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf554, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg361_1, (1024, 1024), (1, 1024), 0), out=buf555)
        del arg361_1
        buf556 = reinterpret_tensor(buf526, (2048, 1024), (1024, 1), 0); del buf526  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf554, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg363_1, (1024, 1024), (1, 1024), 0), out=buf556)
        del arg363_1
        buf557 = reinterpret_tensor(buf534, (2048, 1024), (1024, 1), 0); del buf534  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf554, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg365_1, (1024, 1024), (1, 1024), 0), out=buf557)
        del arg365_1
        buf558 = reinterpret_tensor(buf554, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf554  # reuse
        # Topologically Sorted Source Nodes: [query_states_49, key_states_24, value_states_24, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf555, arg362_1, buf558, 2097152, grid=grid(2097152), stream=stream0)
        del arg362_1
        buf559 = reinterpret_tensor(buf555, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf555  # reuse
        # Topologically Sorted Source Nodes: [query_states_49, key_states_24, value_states_24, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf556, arg364_1, buf559, 2097152, grid=grid(2097152), stream=stream0)
        del arg364_1
        buf560 = reinterpret_tensor(buf556, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf556  # reuse
        # Topologically Sorted Source Nodes: [query_states_49, key_states_24, value_states_24, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf557, arg366_1, buf560, 2097152, grid=grid(2097152), stream=stream0)
        del arg366_1
        buf561 = buf519; del buf519  # reuse
        # Topologically Sorted Source Nodes: [query_states_49, key_states_24, value_states_24, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_7.run(buf561, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_49, key_states_24, value_states_24, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf562 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf558, buf559, buf560, buf561, False)
        buf563 = buf562[0]
        del buf562
        buf567 = reinterpret_tensor(buf560, (2048, 1024), (1024, 1), 0); del buf560  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf563, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg367_1, (1024, 1024), (1, 1024), 0), out=buf567)
        del arg367_1
        buf571 = reinterpret_tensor(buf563, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf563  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_189, hidden_states_190], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf550, buf567, arg368_1, arg369_1, arg370_1, buf571, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg369_1
        del arg370_1
        buf572 = reinterpret_tensor(buf559, (2048, 1024), (1024, 1), 0); del buf559  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf571, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg371_1, (1024, 1024), (1, 1024), 0), out=buf572)
        del arg371_1
        buf573 = reinterpret_tensor(buf571, (2048, 1024), (1024, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg373_1, (1024, 1024), (1, 1024), 0), out=buf573)
        del arg373_1
        buf574 = reinterpret_tensor(buf558, (2048, 1024), (1024, 1), 0); del buf558  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg375_1, (1024, 1024), (1, 1024), 0), out=buf574)
        del arg375_1
        buf575 = reinterpret_tensor(buf557, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [query_states_51, key_states_25, value_states_25, attn_output_100], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf572, arg372_1, buf575, 2097152, grid=grid(2097152), stream=stream0)
        del arg372_1
        buf576 = reinterpret_tensor(buf572, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf572  # reuse
        # Topologically Sorted Source Nodes: [query_states_51, key_states_25, value_states_25, attn_output_100], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf573, arg374_1, buf576, 2097152, grid=grid(2097152), stream=stream0)
        del arg374_1
        buf577 = reinterpret_tensor(buf573, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf573  # reuse
        # Topologically Sorted Source Nodes: [query_states_51, key_states_25, value_states_25, attn_output_100], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf574, arg376_1, buf577, 2097152, grid=grid(2097152), stream=stream0)
        del arg376_1
        del buf574
        # Topologically Sorted Source Nodes: [query_states_51, key_states_25, value_states_25, attn_output_100], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf578 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf575, buf576, buf577, None, False)
        buf579 = buf578[0]
        del buf578
        buf583 = reinterpret_tensor(buf577, (2048, 1024), (1024, 1), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf579, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg377_1, (1024, 1024), (1, 1024), 0), out=buf583)
        del arg377_1
        buf584 = reinterpret_tensor(buf583, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf583  # reuse
        buf588 = reinterpret_tensor(buf579, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf579  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_189, hidden_states_192, hidden_states_193], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf584, buf550, buf567, arg368_1, arg378_1, arg379_1, arg380_1, buf588, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg368_1
        del arg378_1
        del arg379_1
        del arg380_1
        buf589 = reinterpret_tensor(buf548, (2048, 4096), (4096, 1), 0); del buf548  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf588, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg381_1, (1024, 4096), (1, 1024), 0), out=buf589)
        del arg381_1
        buf590 = reinterpret_tensor(buf589, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf589  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_194], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf590, arg382_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg382_1
        buf591 = reinterpret_tensor(buf588, (2048, 1024), (1024, 1), 0); del buf588  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf590, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg383_1, (4096, 1024), (1, 4096), 0), out=buf591)
        del arg383_1
        buf595 = reinterpret_tensor(buf567, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf567  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_198, hidden_states_199], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf584, buf591, arg384_1, arg385_1, arg386_1, buf595, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg385_1
        del arg386_1
        buf596 = reinterpret_tensor(buf550, (2048, 1024), (1024, 1), 0); del buf550  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf595, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg387_1, (1024, 1024), (1, 1024), 0), out=buf596)
        del arg387_1
        buf597 = reinterpret_tensor(buf576, (2048, 1024), (1024, 1), 0); del buf576  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf595, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg389_1, (1024, 1024), (1, 1024), 0), out=buf597)
        del arg389_1
        buf598 = reinterpret_tensor(buf575, (2048, 1024), (1024, 1), 0); del buf575  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf595, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg391_1, (1024, 1024), (1, 1024), 0), out=buf598)
        del arg391_1
        buf599 = reinterpret_tensor(buf595, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf595  # reuse
        # Topologically Sorted Source Nodes: [query_states_53, key_states_26, value_states_26, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf596, arg388_1, buf599, 2097152, grid=grid(2097152), stream=stream0)
        del arg388_1
        buf600 = reinterpret_tensor(buf596, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf596  # reuse
        # Topologically Sorted Source Nodes: [query_states_53, key_states_26, value_states_26, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf597, arg390_1, buf600, 2097152, grid=grid(2097152), stream=stream0)
        del arg390_1
        buf601 = reinterpret_tensor(buf597, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf597  # reuse
        # Topologically Sorted Source Nodes: [query_states_53, key_states_26, value_states_26, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf598, arg392_1, buf601, 2097152, grid=grid(2097152), stream=stream0)
        del arg392_1
        del buf598
        buf602 = buf561; del buf561  # reuse
        # Topologically Sorted Source Nodes: [query_states_53, key_states_26, value_states_26, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_7.run(buf602, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_53, key_states_26, value_states_26, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf603 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf599, buf600, buf601, buf602, False)
        del buf599
        buf604 = buf603[0]
        del buf603
        buf608 = reinterpret_tensor(buf601, (2048, 1024), (1024, 1), 0); del buf601  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf604, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg393_1, (1024, 1024), (1, 1024), 0), out=buf608)
        del arg393_1
        buf609 = reinterpret_tensor(buf608, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf608  # reuse
        buf613 = reinterpret_tensor(buf604, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf604  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_198, hidden_states_201, hidden_states_202], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf609, buf584, buf591, arg384_1, arg394_1, arg395_1, arg396_1, buf613, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg384_1
        del arg394_1
        del arg395_1
        del arg396_1
        buf614 = buf591; del buf591  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf613, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg397_1, (1024, 1024), (1, 1024), 0), out=buf614)
        del arg397_1
        buf615 = reinterpret_tensor(buf613, (2048, 1024), (1024, 1), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg399_1, (1024, 1024), (1, 1024), 0), out=buf615)
        del arg399_1
        buf616 = reinterpret_tensor(buf584, (2048, 1024), (1024, 1), 0); del buf584  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg401_1, (1024, 1024), (1, 1024), 0), out=buf616)
        del arg401_1
        buf617 = buf600; del buf600  # reuse
        # Topologically Sorted Source Nodes: [query_states_55, key_states_27, value_states_27, attn_output_108], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf614, arg398_1, buf617, 2097152, grid=grid(2097152), stream=stream0)
        del arg398_1
        buf618 = reinterpret_tensor(buf614, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf614  # reuse
        # Topologically Sorted Source Nodes: [query_states_55, key_states_27, value_states_27, attn_output_108], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf615, arg400_1, buf618, 2097152, grid=grid(2097152), stream=stream0)
        del arg400_1
        buf619 = reinterpret_tensor(buf615, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf615  # reuse
        # Topologically Sorted Source Nodes: [query_states_55, key_states_27, value_states_27, attn_output_108], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf616, arg402_1, buf619, 2097152, grid=grid(2097152), stream=stream0)
        del arg402_1
        del buf616
        # Topologically Sorted Source Nodes: [query_states_55, key_states_27, value_states_27, attn_output_108], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf620 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf617, buf618, buf619, None, False)
        buf621 = buf620[0]
        del buf620
        buf625 = reinterpret_tensor(buf619, (2048, 1024), (1024, 1), 0); del buf619  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf621, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg403_1, (1024, 1024), (1, 1024), 0), out=buf625)
        del arg403_1
        buf629 = reinterpret_tensor(buf621, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf621  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_204, hidden_states_205], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf609, buf625, arg404_1, arg405_1, arg406_1, buf629, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg405_1
        del arg406_1
        buf630 = reinterpret_tensor(buf590, (2048, 4096), (4096, 1), 0); del buf590  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf629, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg407_1, (1024, 4096), (1, 1024), 0), out=buf630)
        del arg407_1
        buf631 = reinterpret_tensor(buf630, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf630  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_206], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf631, arg408_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg408_1
        buf632 = reinterpret_tensor(buf629, (2048, 1024), (1024, 1), 0); del buf629  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf631, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg409_1, (4096, 1024), (1, 4096), 0), out=buf632)
        del arg409_1
        buf633 = reinterpret_tensor(buf632, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf632  # reuse
        buf637 = reinterpret_tensor(buf618, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf618  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_204, hidden_states_210, hidden_states_211], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf633, buf609, buf625, arg404_1, arg410_1, arg411_1, arg412_1, buf637, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg404_1
        del arg410_1
        del arg411_1
        del arg412_1
        buf638 = buf625; del buf625  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf637, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg413_1, (1024, 1024), (1, 1024), 0), out=buf638)
        del arg413_1
        buf639 = reinterpret_tensor(buf609, (2048, 1024), (1024, 1), 0); del buf609  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf637, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg415_1, (1024, 1024), (1, 1024), 0), out=buf639)
        del arg415_1
        buf640 = reinterpret_tensor(buf617, (2048, 1024), (1024, 1), 0); del buf617  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf637, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg417_1, (1024, 1024), (1, 1024), 0), out=buf640)
        del arg417_1
        buf641 = reinterpret_tensor(buf637, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf637  # reuse
        # Topologically Sorted Source Nodes: [query_states_57, key_states_28, value_states_28, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf638, arg414_1, buf641, 2097152, grid=grid(2097152), stream=stream0)
        del arg414_1
        buf642 = reinterpret_tensor(buf638, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf638  # reuse
        # Topologically Sorted Source Nodes: [query_states_57, key_states_28, value_states_28, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf639, arg416_1, buf642, 2097152, grid=grid(2097152), stream=stream0)
        del arg416_1
        buf643 = reinterpret_tensor(buf639, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf639  # reuse
        # Topologically Sorted Source Nodes: [query_states_57, key_states_28, value_states_28, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf640, arg418_1, buf643, 2097152, grid=grid(2097152), stream=stream0)
        del arg418_1
        buf644 = buf602; del buf602  # reuse
        # Topologically Sorted Source Nodes: [query_states_57, key_states_28, value_states_28, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_7.run(buf644, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_57, key_states_28, value_states_28, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf645 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf641, buf642, buf643, buf644, False)
        buf646 = buf645[0]
        del buf645
        buf650 = reinterpret_tensor(buf643, (2048, 1024), (1024, 1), 0); del buf643  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf646, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg419_1, (1024, 1024), (1, 1024), 0), out=buf650)
        del arg419_1
        buf654 = reinterpret_tensor(buf646, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf646  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_213, hidden_states_214], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf633, buf650, arg420_1, arg421_1, arg422_1, buf654, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg421_1
        del arg422_1
        buf655 = reinterpret_tensor(buf642, (2048, 1024), (1024, 1), 0); del buf642  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf654, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg423_1, (1024, 1024), (1, 1024), 0), out=buf655)
        del arg423_1
        buf656 = reinterpret_tensor(buf654, (2048, 1024), (1024, 1), 0); del buf654  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg425_1, (1024, 1024), (1, 1024), 0), out=buf656)
        del arg425_1
        buf657 = reinterpret_tensor(buf641, (2048, 1024), (1024, 1), 0); del buf641  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg427_1, (1024, 1024), (1, 1024), 0), out=buf657)
        del arg427_1
        buf658 = reinterpret_tensor(buf640, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf640  # reuse
        # Topologically Sorted Source Nodes: [query_states_59, key_states_29, value_states_29, attn_output_116], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf655, arg424_1, buf658, 2097152, grid=grid(2097152), stream=stream0)
        del arg424_1
        buf659 = reinterpret_tensor(buf655, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf655  # reuse
        # Topologically Sorted Source Nodes: [query_states_59, key_states_29, value_states_29, attn_output_116], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf656, arg426_1, buf659, 2097152, grid=grid(2097152), stream=stream0)
        del arg426_1
        buf660 = reinterpret_tensor(buf656, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf656  # reuse
        # Topologically Sorted Source Nodes: [query_states_59, key_states_29, value_states_29, attn_output_116], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf657, arg428_1, buf660, 2097152, grid=grid(2097152), stream=stream0)
        del arg428_1
        del buf657
        # Topologically Sorted Source Nodes: [query_states_59, key_states_29, value_states_29, attn_output_116], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf661 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf658, buf659, buf660, None, False)
        buf662 = buf661[0]
        del buf661
        buf666 = reinterpret_tensor(buf660, (2048, 1024), (1024, 1), 0); del buf660  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf662, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg429_1, (1024, 1024), (1, 1024), 0), out=buf666)
        del arg429_1
        buf667 = reinterpret_tensor(buf666, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf666  # reuse
        buf671 = reinterpret_tensor(buf662, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf662  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_213, hidden_states_216, hidden_states_217], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf667, buf633, buf650, arg420_1, arg430_1, arg431_1, arg432_1, buf671, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg420_1
        del arg430_1
        del arg431_1
        del arg432_1
        buf672 = reinterpret_tensor(buf631, (2048, 4096), (4096, 1), 0); del buf631  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf671, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg433_1, (1024, 4096), (1, 1024), 0), out=buf672)
        del arg433_1
        buf673 = reinterpret_tensor(buf672, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf672  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_218], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf673, arg434_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg434_1
        buf674 = reinterpret_tensor(buf671, (2048, 1024), (1024, 1), 0); del buf671  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf673, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg435_1, (4096, 1024), (1, 4096), 0), out=buf674)
        del arg435_1
        buf678 = reinterpret_tensor(buf650, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf650  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_222, hidden_states_223], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf667, buf674, arg436_1, arg437_1, arg438_1, buf678, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg437_1
        del arg438_1
        buf679 = reinterpret_tensor(buf633, (2048, 1024), (1024, 1), 0); del buf633  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf678, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg439_1, (1024, 1024), (1, 1024), 0), out=buf679)
        del arg439_1
        buf680 = reinterpret_tensor(buf659, (2048, 1024), (1024, 1), 0); del buf659  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf678, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg441_1, (1024, 1024), (1, 1024), 0), out=buf680)
        del arg441_1
        buf681 = reinterpret_tensor(buf658, (2048, 1024), (1024, 1), 0); del buf658  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf678, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg443_1, (1024, 1024), (1, 1024), 0), out=buf681)
        del arg443_1
        buf682 = reinterpret_tensor(buf678, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf678  # reuse
        # Topologically Sorted Source Nodes: [query_states_61, key_states_30, value_states_30, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf679, arg440_1, buf682, 2097152, grid=grid(2097152), stream=stream0)
        del arg440_1
        buf683 = reinterpret_tensor(buf679, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf679  # reuse
        # Topologically Sorted Source Nodes: [query_states_61, key_states_30, value_states_30, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf680, arg442_1, buf683, 2097152, grid=grid(2097152), stream=stream0)
        del arg442_1
        buf684 = reinterpret_tensor(buf680, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf680  # reuse
        # Topologically Sorted Source Nodes: [query_states_61, key_states_30, value_states_30, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf681, arg444_1, buf684, 2097152, grid=grid(2097152), stream=stream0)
        del arg444_1
        del buf681
        buf685 = buf644; del buf644  # reuse
        # Topologically Sorted Source Nodes: [query_states_61, key_states_30, value_states_30, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_7.run(buf685, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_61, key_states_30, value_states_30, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf686 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf682, buf683, buf684, buf685, False)
        del buf682
        buf687 = buf686[0]
        del buf686
        buf691 = reinterpret_tensor(buf684, (2048, 1024), (1024, 1), 0); del buf684  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf687, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg445_1, (1024, 1024), (1, 1024), 0), out=buf691)
        del arg445_1
        buf692 = reinterpret_tensor(buf691, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf691  # reuse
        buf696 = reinterpret_tensor(buf687, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf687  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_222, hidden_states_225, hidden_states_226], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf692, buf667, buf674, arg436_1, arg446_1, arg447_1, arg448_1, buf696, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg436_1
        del arg446_1
        del arg447_1
        del arg448_1
        buf697 = buf674; del buf674  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf696, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg449_1, (1024, 1024), (1, 1024), 0), out=buf697)
        del arg449_1
        buf698 = reinterpret_tensor(buf696, (2048, 1024), (1024, 1), 0); del buf696  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg451_1, (1024, 1024), (1, 1024), 0), out=buf698)
        del arg451_1
        buf699 = reinterpret_tensor(buf667, (2048, 1024), (1024, 1), 0); del buf667  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg453_1, (1024, 1024), (1, 1024), 0), out=buf699)
        del arg453_1
        buf700 = buf683; del buf683  # reuse
        # Topologically Sorted Source Nodes: [query_states_63, key_states_31, value_states_31, attn_output_124], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf697, arg450_1, buf700, 2097152, grid=grid(2097152), stream=stream0)
        del arg450_1
        buf701 = reinterpret_tensor(buf697, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf697  # reuse
        # Topologically Sorted Source Nodes: [query_states_63, key_states_31, value_states_31, attn_output_124], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf698, arg452_1, buf701, 2097152, grid=grid(2097152), stream=stream0)
        del arg452_1
        buf702 = reinterpret_tensor(buf698, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf698  # reuse
        # Topologically Sorted Source Nodes: [query_states_63, key_states_31, value_states_31, attn_output_124], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf699, arg454_1, buf702, 2097152, grid=grid(2097152), stream=stream0)
        del arg454_1
        del buf699
        # Topologically Sorted Source Nodes: [query_states_63, key_states_31, value_states_31, attn_output_124], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf703 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf700, buf701, buf702, None, False)
        buf704 = buf703[0]
        del buf703
        buf708 = reinterpret_tensor(buf702, (2048, 1024), (1024, 1), 0); del buf702  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf704, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg455_1, (1024, 1024), (1, 1024), 0), out=buf708)
        del arg455_1
        buf712 = reinterpret_tensor(buf704, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf704  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_228, hidden_states_229], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf692, buf708, arg456_1, arg457_1, arg458_1, buf712, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg457_1
        del arg458_1
        buf713 = reinterpret_tensor(buf673, (2048, 4096), (4096, 1), 0); del buf673  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf712, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg459_1, (1024, 4096), (1, 1024), 0), out=buf713)
        del arg459_1
        buf714 = reinterpret_tensor(buf713, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf713  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_230], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf714, arg460_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg460_1
        buf715 = reinterpret_tensor(buf712, (2048, 1024), (1024, 1), 0); del buf712  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf714, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg461_1, (4096, 1024), (1, 4096), 0), out=buf715)
        del arg461_1
        buf716 = reinterpret_tensor(buf715, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf715  # reuse
        buf720 = reinterpret_tensor(buf701, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf701  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_228, hidden_states_234, hidden_states_235], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf716, buf692, buf708, arg456_1, arg462_1, arg463_1, arg464_1, buf720, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg456_1
        del arg462_1
        del arg463_1
        del arg464_1
        buf721 = buf708; del buf708  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf720, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg465_1, (1024, 1024), (1, 1024), 0), out=buf721)
        del arg465_1
        buf722 = reinterpret_tensor(buf692, (2048, 1024), (1024, 1), 0); del buf692  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf720, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg467_1, (1024, 1024), (1, 1024), 0), out=buf722)
        del arg467_1
        buf723 = reinterpret_tensor(buf700, (2048, 1024), (1024, 1), 0); del buf700  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf720, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg469_1, (1024, 1024), (1, 1024), 0), out=buf723)
        del arg469_1
        buf724 = reinterpret_tensor(buf720, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf720  # reuse
        # Topologically Sorted Source Nodes: [query_states_65, key_states_32, value_states_32, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf721, arg466_1, buf724, 2097152, grid=grid(2097152), stream=stream0)
        del arg466_1
        buf725 = reinterpret_tensor(buf721, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf721  # reuse
        # Topologically Sorted Source Nodes: [query_states_65, key_states_32, value_states_32, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf722, arg468_1, buf725, 2097152, grid=grid(2097152), stream=stream0)
        del arg468_1
        buf726 = reinterpret_tensor(buf722, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf722  # reuse
        # Topologically Sorted Source Nodes: [query_states_65, key_states_32, value_states_32, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf723, arg470_1, buf726, 2097152, grid=grid(2097152), stream=stream0)
        del arg470_1
        buf727 = buf685; del buf685  # reuse
        # Topologically Sorted Source Nodes: [query_states_65, key_states_32, value_states_32, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_7.run(buf727, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_65, key_states_32, value_states_32, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf728 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf724, buf725, buf726, buf727, False)
        buf729 = buf728[0]
        del buf728
        buf733 = reinterpret_tensor(buf726, (2048, 1024), (1024, 1), 0); del buf726  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf729, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg471_1, (1024, 1024), (1, 1024), 0), out=buf733)
        del arg471_1
        buf737 = reinterpret_tensor(buf729, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf729  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_237, hidden_states_238], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf716, buf733, arg472_1, arg473_1, arg474_1, buf737, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg473_1
        del arg474_1
        buf738 = reinterpret_tensor(buf725, (2048, 1024), (1024, 1), 0); del buf725  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf737, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg475_1, (1024, 1024), (1, 1024), 0), out=buf738)
        del arg475_1
        buf739 = reinterpret_tensor(buf737, (2048, 1024), (1024, 1), 0); del buf737  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg477_1, (1024, 1024), (1, 1024), 0), out=buf739)
        del arg477_1
        buf740 = reinterpret_tensor(buf724, (2048, 1024), (1024, 1), 0); del buf724  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg479_1, (1024, 1024), (1, 1024), 0), out=buf740)
        del arg479_1
        buf741 = reinterpret_tensor(buf723, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf723  # reuse
        # Topologically Sorted Source Nodes: [query_states_67, key_states_33, value_states_33, attn_output_132], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf738, arg476_1, buf741, 2097152, grid=grid(2097152), stream=stream0)
        del arg476_1
        buf742 = reinterpret_tensor(buf738, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf738  # reuse
        # Topologically Sorted Source Nodes: [query_states_67, key_states_33, value_states_33, attn_output_132], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf739, arg478_1, buf742, 2097152, grid=grid(2097152), stream=stream0)
        del arg478_1
        buf743 = reinterpret_tensor(buf739, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf739  # reuse
        # Topologically Sorted Source Nodes: [query_states_67, key_states_33, value_states_33, attn_output_132], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf740, arg480_1, buf743, 2097152, grid=grid(2097152), stream=stream0)
        del arg480_1
        del buf740
        # Topologically Sorted Source Nodes: [query_states_67, key_states_33, value_states_33, attn_output_132], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf744 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf741, buf742, buf743, None, False)
        buf745 = buf744[0]
        del buf744
        buf749 = reinterpret_tensor(buf743, (2048, 1024), (1024, 1), 0); del buf743  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf745, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg481_1, (1024, 1024), (1, 1024), 0), out=buf749)
        del arg481_1
        buf750 = reinterpret_tensor(buf749, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf749  # reuse
        buf754 = reinterpret_tensor(buf745, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf745  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_237, hidden_states_240, hidden_states_241], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf750, buf716, buf733, arg472_1, arg482_1, arg483_1, arg484_1, buf754, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg472_1
        del arg482_1
        del arg483_1
        del arg484_1
        buf755 = reinterpret_tensor(buf714, (2048, 4096), (4096, 1), 0); del buf714  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf754, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg485_1, (1024, 4096), (1, 1024), 0), out=buf755)
        del arg485_1
        buf756 = reinterpret_tensor(buf755, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf755  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_242], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf756, arg486_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg486_1
        buf757 = reinterpret_tensor(buf754, (2048, 1024), (1024, 1), 0); del buf754  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf756, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg487_1, (4096, 1024), (1, 4096), 0), out=buf757)
        del arg487_1
        buf761 = reinterpret_tensor(buf733, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf733  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_246, hidden_states_247], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf750, buf757, arg488_1, arg489_1, arg490_1, buf761, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg489_1
        del arg490_1
        buf762 = reinterpret_tensor(buf716, (2048, 1024), (1024, 1), 0); del buf716  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf761, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg491_1, (1024, 1024), (1, 1024), 0), out=buf762)
        del arg491_1
        buf763 = reinterpret_tensor(buf742, (2048, 1024), (1024, 1), 0); del buf742  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf761, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg493_1, (1024, 1024), (1, 1024), 0), out=buf763)
        del arg493_1
        buf764 = reinterpret_tensor(buf741, (2048, 1024), (1024, 1), 0); del buf741  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf761, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg495_1, (1024, 1024), (1, 1024), 0), out=buf764)
        del arg495_1
        buf765 = reinterpret_tensor(buf761, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf761  # reuse
        # Topologically Sorted Source Nodes: [query_states_69, key_states_34, value_states_34, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf762, arg492_1, buf765, 2097152, grid=grid(2097152), stream=stream0)
        del arg492_1
        buf766 = reinterpret_tensor(buf762, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf762  # reuse
        # Topologically Sorted Source Nodes: [query_states_69, key_states_34, value_states_34, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf763, arg494_1, buf766, 2097152, grid=grid(2097152), stream=stream0)
        del arg494_1
        buf767 = reinterpret_tensor(buf763, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf763  # reuse
        # Topologically Sorted Source Nodes: [query_states_69, key_states_34, value_states_34, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf764, arg496_1, buf767, 2097152, grid=grid(2097152), stream=stream0)
        del arg496_1
        del buf764
        buf768 = buf727; del buf727  # reuse
        # Topologically Sorted Source Nodes: [query_states_69, key_states_34, value_states_34, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_7.run(buf768, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_69, key_states_34, value_states_34, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf769 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf765, buf766, buf767, buf768, False)
        del buf765
        del buf768
        buf770 = buf769[0]
        del buf769
        buf774 = reinterpret_tensor(buf767, (2048, 1024), (1024, 1), 0); del buf767  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf770, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg497_1, (1024, 1024), (1, 1024), 0), out=buf774)
        del arg497_1
        buf775 = reinterpret_tensor(buf774, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf774  # reuse
        buf779 = reinterpret_tensor(buf770, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf770  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_246, hidden_states_249, hidden_states_250], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf775, buf750, buf757, arg488_1, arg498_1, arg499_1, arg500_1, buf779, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg488_1
        del arg498_1
        del arg499_1
        del arg500_1
        buf780 = buf757; del buf757  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf779, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg501_1, (1024, 1024), (1, 1024), 0), out=buf780)
        del arg501_1
        buf781 = reinterpret_tensor(buf779, (2048, 1024), (1024, 1), 0); del buf779  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg503_1, (1024, 1024), (1, 1024), 0), out=buf781)
        del arg503_1
        buf782 = reinterpret_tensor(buf750, (2048, 1024), (1024, 1), 0); del buf750  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg505_1, (1024, 1024), (1, 1024), 0), out=buf782)
        del arg505_1
        buf783 = buf766; del buf766  # reuse
        # Topologically Sorted Source Nodes: [query_states_71, key_states_35, value_states_35, attn_output_140], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf780, arg502_1, buf783, 2097152, grid=grid(2097152), stream=stream0)
        del arg502_1
        buf784 = reinterpret_tensor(buf780, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf780  # reuse
        # Topologically Sorted Source Nodes: [query_states_71, key_states_35, value_states_35, attn_output_140], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf781, arg504_1, buf784, 2097152, grid=grid(2097152), stream=stream0)
        del arg504_1
        buf785 = reinterpret_tensor(buf781, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf781  # reuse
        # Topologically Sorted Source Nodes: [query_states_71, key_states_35, value_states_35, attn_output_140], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf782, arg506_1, buf785, 2097152, grid=grid(2097152), stream=stream0)
        del arg506_1
        del buf782
        # Topologically Sorted Source Nodes: [query_states_71, key_states_35, value_states_35, attn_output_140], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf786 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf783, buf784, buf785, None, False)
        del buf783
        buf787 = buf786[0]
        del buf786
        buf791 = reinterpret_tensor(buf785, (2048, 1024), (1024, 1), 0); del buf785  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf787, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg507_1, (1024, 1024), (1, 1024), 0), out=buf791)
        del arg507_1
        buf795 = reinterpret_tensor(buf787, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf787  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_252, hidden_states_253], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf775, buf791, arg508_1, arg509_1, arg510_1, buf795, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg509_1
        del arg510_1
        buf796 = reinterpret_tensor(buf756, (2048, 4096), (4096, 1), 0); del buf756  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf795, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg511_1, (1024, 4096), (1, 1024), 0), out=buf796)
        del arg511_1
        buf797 = reinterpret_tensor(buf796, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf796  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_254], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf797, arg512_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg512_1
        buf798 = reinterpret_tensor(buf795, (2048, 1024), (1024, 1), 0); del buf795  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf797, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg513_1, (4096, 1024), (1, 4096), 0), out=buf798)
        del arg513_1
        del buf797
        buf799 = reinterpret_tensor(buf798, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf798  # reuse
        buf803 = reinterpret_tensor(buf784, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf784  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_252, hidden_states_258, hidden_states_259], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf799, buf775, buf791, arg508_1, arg514_1, arg515_1, arg516_1, buf803, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg508_1
        del arg514_1
        del arg515_1
        del arg516_1
        del buf775
        del buf791
        del buf799
        buf804 = empty_strided_cuda((1024, 50268), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_8.run(arg2_1, buf804, 51474432, grid=grid(51474432), stream=stream0)
        del arg2_1
        buf805 = empty_strided_cuda((2048, 50268), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf803, (2048, 1024), (1024, 1), 0), buf804, out=buf805)
        del buf803
        del buf804
        buf806 = empty_strided_cuda((2, 1024, 50265), (51471360, 50265, 1), torch.float32)
        buf807 = empty_strided_cuda((2048, 1), (1, 2048), torch.float32)
        buf808 = empty_strided_cuda((2048, 1), (1, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [lm_logits, masked_lm_loss], Original ATen: [aten.add, aten._log_softmax]
        triton_red_fused__log_softmax_add_9.run(buf805, arg517_1, buf806, buf807, buf808, 2048, 50265, grid=grid(2048), stream=stream0)
        del arg517_1
        del buf805
        buf809 = empty_strided_cuda((), (), torch.float32)
        buf811 = buf809; del buf809  # reuse
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_10.run(buf811, arg0_1, buf806, buf807, buf808, 1, 2048, grid=grid(1), stream=stream0)
        del arg0_1
        del buf807
        del buf808
    return (buf811, buf806, buf323, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((2, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((1, 50265), (50265, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MBartForConditionalGeneration', benchmark_compiled_module)
