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


# kernel path: /tmp/torchinductor_sahanp/6p/c6psvhvp7dzgeaw63p5ovhqu2l4bpb23wng5p7umjaifzad2xgx4.py
# Topologically Sorted Source Nodes: [embedding, inputs_embeds, add, embed_pos, hidden_states, hidden_states_1], Original ATen: [aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add
#   embed_pos => embedding_1
#   embedding => embedding
#   hidden_states => add_1
#   hidden_states_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub, var_mean
#   inputs_embeds => mul
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %view, 1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, 1.0), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand, 2), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %add), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg4_1), kwargs = {})
#   %add_3 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg5_1), kwargs = {})
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rk/crkut6dhi6os5npdt2se5ly2ubt6me7zyrkfjnedb7dxivqwdemh.py
# Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_efficient_attention
#   key_states => clone_2
#   query_states_1 => clone_4
#   value_states => clone_3
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_4, %clone_2, %clone_3, None, False), kwargs = {})
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
# Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_5], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_4 => add_4
#   hidden_states_5 => add_5, add_6, mul_3, mul_4, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_12), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_7), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg14_1), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg15_1), kwargs = {})
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
# Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_6 => add_7, erf, mul_5, mul_6, mul_7
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_7), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/ft/cftex47rlk3wimeanm46r6tkuhxnoz2yn2zmxri2t2aeh7e5q6m7.py
# Topologically Sorted Source Nodes: [shifted_input_ids, clone, setitem, setitem_1, eq, masked_fill_, embedding_2, inputs_embeds_1, add_27, positions_2, hidden_states_111, hidden_states_112], Original ATen: [aten.new_zeros, aten.clone, aten.copy, aten.lift_fresh, aten.fill, aten.eq, aten.masked_fill, aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_27 => add_89
#   clone => clone
#   embedding_2 => embedding_2
#   eq => eq
#   hidden_states_111 => add_90
#   hidden_states_112 => add_91, add_92, mul_88, mul_89, rsqrt_25, sub_25, var_mean_25
#   inputs_embeds_1 => mul_87
#   masked_fill_ => full_default_1, where
#   positions_2 => embedding_3
#   setitem => copy
#   setitem_1 => copy_1, full_default
#   shifted_input_ids => full
# Graph fragment:
#   %full : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([2, 1024], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_4, %clone), kwargs = {})
#   %slice_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full, %copy, 1, 1, 9223372036854775807), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 2), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_1, %full_default), kwargs = {})
#   %select_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%slice_scatter_default, %copy_1, 1, 0), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%select_scatter_default, -100), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_1, %select_scatter_default), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %where, 1), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding_2, 1.0), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_2, 2), kwargs = {})
#   %embedding_3 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg198_1, %add_89), kwargs = {})
#   %add_90 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %embedding_3), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_90, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_90, %getitem_99), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_98, 1e-05), kwargs = {})
#   %rsqrt_25 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_91,), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt_25), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %arg199_1), kwargs = {})
#   %add_92 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %arg200_1), kwargs = {})
triton_red_fused_add_clone_copy_embedding_eq_fill_lift_fresh_masked_fill_mul_native_layer_norm_new_zeros_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_copy_embedding_eq_fill_lift_fresh_masked_fill_mul_native_layer_norm_new_zeros_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x3 = xindex
    tmp24_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp21 = tl.load(in_ptr2 + (2048 + r2 + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 >= tmp3
        tmp5 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + x3, [XBLOCK, RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full([1, 1], 0, tl.int64)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tl.full([1, 1], 2, tl.int64)
        tmp9 = tl.where(tmp2, tmp8, tmp7)
        tmp10 = tl.full([1, 1], -100, tl.int64)
        tmp11 = tmp9 == tmp10
        tmp12 = tl.where(tmp11, tmp3, tmp9)
        tmp13 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp14 = tmp12 + tmp13
        tmp15 = tmp12 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp12)
        tl.device_assert(((0 <= tmp16) & (tmp16 < 50265)) | ~(rmask), "index out of bounds: 0 <= tmp16 < 50265")
        tmp18 = tl.load(in_ptr1 + (r2 + (1024*tmp16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = 1.0
        tmp20 = tmp18 * tmp19
        tmp22 = tmp20 + tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp24_mean_next, tmp24_m2_next, tmp24_weight_next = triton_helpers.welford_reduce(
            tmp23, tmp24_mean, tmp24_m2, tmp24_weight, roffset == 0
        )
        tmp24_mean = tl.where(rmask, tmp24_mean_next, tmp24_mean)
        tmp24_m2 = tl.where(rmask, tmp24_m2_next, tmp24_m2)
        tmp24_weight = tl.where(rmask, tmp24_weight_next, tmp24_weight)
    tmp24_tmp, tmp25_tmp, tmp26_tmp = triton_helpers.welford(
        tmp24_mean, tmp24_m2, tmp24_weight, 1
    )
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp48 = tl.load(in_ptr2 + (2048 + r2 + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp57 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp59 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = x0
        tmp28 = tl.full([1, 1], 0, tl.int32)
        tmp29 = tmp27 == tmp28
        tmp30 = tl.full([1, 1], 1, tl.int64)
        tmp31 = tmp27 >= tmp30
        tmp32 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + x3, [XBLOCK, RBLOCK])), rmask & tmp31, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.full([1, 1], 0, tl.int64)
        tmp34 = tl.where(tmp31, tmp32, tmp33)
        tmp35 = tl.full([1, 1], 2, tl.int64)
        tmp36 = tl.where(tmp29, tmp35, tmp34)
        tmp37 = tl.full([1, 1], -100, tl.int64)
        tmp38 = tmp36 == tmp37
        tmp39 = tl.where(tmp38, tmp30, tmp36)
        tmp40 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp41 = tmp39 + tmp40
        tmp42 = tmp39 < 0
        tmp43 = tl.where(tmp42, tmp41, tmp39)
        tl.device_assert(((0 <= tmp43) & (tmp43 < 50265)) | ~(rmask), "index out of bounds: 0 <= tmp43 < 50265")
        tmp45 = tl.load(in_ptr1 + (r2 + (1024*tmp43)), rmask, eviction_policy='evict_first', other=0.0)
        tmp46 = 1.0
        tmp47 = tmp45 * tmp46
        tmp49 = tmp47 + tmp48
        tmp50 = tmp49 - tmp24
        tmp51 = 1024.0
        tmp52 = tmp25 / tmp51
        tmp53 = 1e-05
        tmp54 = tmp52 + tmp53
        tmp55 = libdevice.rsqrt(tmp54)
        tmp56 = tmp50 * tmp55
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(in_out_ptr0 + (r2 + (1024*x3)), tmp60, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tn/ctnve3fyxzp346576zexj4e6mefo4o3pgzk5nkdfmxzqujft6mse.py
# Topologically Sorted Source Nodes: [query_states_25, key_states_12, value_states_12, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output_48 => _scaled_dot_product_efficient_attention_12
#   key_states_12 => clone_75
#   query_states_25 => clone_77
#   value_states_12 => clone_76
# Graph fragment:
#   %clone_77 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_125,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_75 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_122,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_76 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_124,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_dot_product_efficient_attention_12 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_77, %clone_75, %clone_76, %expand_5, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/qo/cqodmbnzkpgxdm23bwtimxmb2bncn376okepj7qfixwwfe3455s3.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_336, %full_default_6], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/ay/cay2sflwm5lncr5ohwkvgeesgfgpeoonn2r7tag6fgg7s34lgtaj.py
# Topologically Sorted Source Nodes: [lm_logits_1, masked_lm_loss], Original ATen: [aten.add, aten._log_softmax]
# Source node to ATen node mapping:
#   lm_logits_1 => add_213
#   masked_lm_loss => amax, exp, sub_62, sum_1
# Graph fragment:
#   %add_213 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_532, %arg513_1), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_533, [1], True), kwargs = {})
#   %sub_62 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_533, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_62,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_red_fused__log_softmax_add_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_add_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/ol/colaq5vjhmhe3amncw6sqg5rwe5zicznqmwq523dnuc2dx7tdepa.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   masked_lm_loss => convert_element_type, div, full_default_5, ne_1, ne_2, neg, sum_2, sum_3, where_3
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_534, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_5), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_3,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_534, -100), kwargs = {})
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
    size_hints=[1, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {5: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 1024), (1024, 1))
    assert_size_stride(arg1_1, (2, 1024), (1024, 1))
    assert_size_stride(arg2_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg3_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg4_1, (1024, ), (1, ))
    assert_size_stride(arg5_1, (1024, ), (1, ))
    assert_size_stride(arg6_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg7_1, (1024, ), (1, ))
    assert_size_stride(arg8_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg9_1, (1024, ), (1, ))
    assert_size_stride(arg10_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg11_1, (1024, ), (1, ))
    assert_size_stride(arg12_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg13_1, (1024, ), (1, ))
    assert_size_stride(arg14_1, (1024, ), (1, ))
    assert_size_stride(arg15_1, (1024, ), (1, ))
    assert_size_stride(arg16_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg17_1, (4096, ), (1, ))
    assert_size_stride(arg18_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg19_1, (1024, ), (1, ))
    assert_size_stride(arg20_1, (1024, ), (1, ))
    assert_size_stride(arg21_1, (1024, ), (1, ))
    assert_size_stride(arg22_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg23_1, (1024, ), (1, ))
    assert_size_stride(arg24_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg25_1, (1024, ), (1, ))
    assert_size_stride(arg26_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg27_1, (1024, ), (1, ))
    assert_size_stride(arg28_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg29_1, (1024, ), (1, ))
    assert_size_stride(arg30_1, (1024, ), (1, ))
    assert_size_stride(arg31_1, (1024, ), (1, ))
    assert_size_stride(arg32_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg33_1, (4096, ), (1, ))
    assert_size_stride(arg34_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg35_1, (1024, ), (1, ))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg39_1, (1024, ), (1, ))
    assert_size_stride(arg40_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg41_1, (1024, ), (1, ))
    assert_size_stride(arg42_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg43_1, (1024, ), (1, ))
    assert_size_stride(arg44_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg45_1, (1024, ), (1, ))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (1024, ), (1, ))
    assert_size_stride(arg48_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg49_1, (4096, ), (1, ))
    assert_size_stride(arg50_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg51_1, (1024, ), (1, ))
    assert_size_stride(arg52_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (1024, ), (1, ))
    assert_size_stride(arg54_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg55_1, (1024, ), (1, ))
    assert_size_stride(arg56_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg57_1, (1024, ), (1, ))
    assert_size_stride(arg58_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg61_1, (1024, ), (1, ))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg63_1, (1024, ), (1, ))
    assert_size_stride(arg64_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg65_1, (4096, ), (1, ))
    assert_size_stride(arg66_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (1024, ), (1, ))
    assert_size_stride(arg70_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg71_1, (1024, ), (1, ))
    assert_size_stride(arg72_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg73_1, (1024, ), (1, ))
    assert_size_stride(arg74_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg75_1, (1024, ), (1, ))
    assert_size_stride(arg76_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (1024, ), (1, ))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg81_1, (4096, ), (1, ))
    assert_size_stride(arg82_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, ), (1, ))
    assert_size_stride(arg86_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg87_1, (1024, ), (1, ))
    assert_size_stride(arg88_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg89_1, (1024, ), (1, ))
    assert_size_stride(arg90_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg93_1, (1024, ), (1, ))
    assert_size_stride(arg94_1, (1024, ), (1, ))
    assert_size_stride(arg95_1, (1024, ), (1, ))
    assert_size_stride(arg96_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg97_1, (4096, ), (1, ))
    assert_size_stride(arg98_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg103_1, (1024, ), (1, ))
    assert_size_stride(arg104_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg105_1, (1024, ), (1, ))
    assert_size_stride(arg106_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg107_1, (1024, ), (1, ))
    assert_size_stride(arg108_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (1024, ), (1, ))
    assert_size_stride(arg112_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg113_1, (4096, ), (1, ))
    assert_size_stride(arg114_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg119_1, (1024, ), (1, ))
    assert_size_stride(arg120_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg121_1, (1024, ), (1, ))
    assert_size_stride(arg122_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg123_1, (1024, ), (1, ))
    assert_size_stride(arg124_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (1024, ), (1, ))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg129_1, (4096, ), (1, ))
    assert_size_stride(arg130_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg141_1, (1024, ), (1, ))
    assert_size_stride(arg142_1, (1024, ), (1, ))
    assert_size_stride(arg143_1, (1024, ), (1, ))
    assert_size_stride(arg144_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg145_1, (4096, ), (1, ))
    assert_size_stride(arg146_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg151_1, (1024, ), (1, ))
    assert_size_stride(arg152_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg153_1, (1024, ), (1, ))
    assert_size_stride(arg154_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg157_1, (1024, ), (1, ))
    assert_size_stride(arg158_1, (1024, ), (1, ))
    assert_size_stride(arg159_1, (1024, ), (1, ))
    assert_size_stride(arg160_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg161_1, (4096, ), (1, ))
    assert_size_stride(arg162_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (1024, ), (1, ))
    assert_size_stride(arg166_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg171_1, (1024, ), (1, ))
    assert_size_stride(arg172_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg173_1, (1024, ), (1, ))
    assert_size_stride(arg174_1, (1024, ), (1, ))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg177_1, (4096, ), (1, ))
    assert_size_stride(arg178_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg179_1, (1024, ), (1, ))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg183_1, (1024, ), (1, ))
    assert_size_stride(arg184_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg185_1, (1024, ), (1, ))
    assert_size_stride(arg186_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg187_1, (1024, ), (1, ))
    assert_size_stride(arg188_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg189_1, (1024, ), (1, ))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg193_1, (4096, ), (1, ))
    assert_size_stride(arg194_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg195_1, (1024, ), (1, ))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (1024, ), (1, ))
    assert_size_stride(arg198_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg199_1, (1024, ), (1, ))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg202_1, (1024, ), (1, ))
    assert_size_stride(arg203_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg204_1, (1024, ), (1, ))
    assert_size_stride(arg205_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg208_1, (1024, ), (1, ))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (1024, ), (1, ))
    assert_size_stride(arg211_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg216_1, (1024, ), (1, ))
    assert_size_stride(arg217_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg218_1, (1024, ), (1, ))
    assert_size_stride(arg219_1, (1024, ), (1, ))
    assert_size_stride(arg220_1, (1024, ), (1, ))
    assert_size_stride(arg221_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg222_1, (4096, ), (1, ))
    assert_size_stride(arg223_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg224_1, (1024, ), (1, ))
    assert_size_stride(arg225_1, (1024, ), (1, ))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg228_1, (1024, ), (1, ))
    assert_size_stride(arg229_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg230_1, (1024, ), (1, ))
    assert_size_stride(arg231_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg232_1, (1024, ), (1, ))
    assert_size_stride(arg233_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg234_1, (1024, ), (1, ))
    assert_size_stride(arg235_1, (1024, ), (1, ))
    assert_size_stride(arg236_1, (1024, ), (1, ))
    assert_size_stride(arg237_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg238_1, (1024, ), (1, ))
    assert_size_stride(arg239_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg240_1, (1024, ), (1, ))
    assert_size_stride(arg241_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (1024, ), (1, ))
    assert_size_stride(arg246_1, (1024, ), (1, ))
    assert_size_stride(arg247_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg248_1, (4096, ), (1, ))
    assert_size_stride(arg249_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg250_1, (1024, ), (1, ))
    assert_size_stride(arg251_1, (1024, ), (1, ))
    assert_size_stride(arg252_1, (1024, ), (1, ))
    assert_size_stride(arg253_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg254_1, (1024, ), (1, ))
    assert_size_stride(arg255_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg256_1, (1024, ), (1, ))
    assert_size_stride(arg257_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg258_1, (1024, ), (1, ))
    assert_size_stride(arg259_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg260_1, (1024, ), (1, ))
    assert_size_stride(arg261_1, (1024, ), (1, ))
    assert_size_stride(arg262_1, (1024, ), (1, ))
    assert_size_stride(arg263_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg264_1, (1024, ), (1, ))
    assert_size_stride(arg265_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg266_1, (1024, ), (1, ))
    assert_size_stride(arg267_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg268_1, (1024, ), (1, ))
    assert_size_stride(arg269_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg270_1, (1024, ), (1, ))
    assert_size_stride(arg271_1, (1024, ), (1, ))
    assert_size_stride(arg272_1, (1024, ), (1, ))
    assert_size_stride(arg273_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg274_1, (4096, ), (1, ))
    assert_size_stride(arg275_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg276_1, (1024, ), (1, ))
    assert_size_stride(arg277_1, (1024, ), (1, ))
    assert_size_stride(arg278_1, (1024, ), (1, ))
    assert_size_stride(arg279_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg282_1, (1024, ), (1, ))
    assert_size_stride(arg283_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg284_1, (1024, ), (1, ))
    assert_size_stride(arg285_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg286_1, (1024, ), (1, ))
    assert_size_stride(arg287_1, (1024, ), (1, ))
    assert_size_stride(arg288_1, (1024, ), (1, ))
    assert_size_stride(arg289_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg290_1, (1024, ), (1, ))
    assert_size_stride(arg291_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg292_1, (1024, ), (1, ))
    assert_size_stride(arg293_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg294_1, (1024, ), (1, ))
    assert_size_stride(arg295_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg296_1, (1024, ), (1, ))
    assert_size_stride(arg297_1, (1024, ), (1, ))
    assert_size_stride(arg298_1, (1024, ), (1, ))
    assert_size_stride(arg299_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg300_1, (4096, ), (1, ))
    assert_size_stride(arg301_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (1024, ), (1, ))
    assert_size_stride(arg304_1, (1024, ), (1, ))
    assert_size_stride(arg305_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg306_1, (1024, ), (1, ))
    assert_size_stride(arg307_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg308_1, (1024, ), (1, ))
    assert_size_stride(arg309_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg310_1, (1024, ), (1, ))
    assert_size_stride(arg311_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg312_1, (1024, ), (1, ))
    assert_size_stride(arg313_1, (1024, ), (1, ))
    assert_size_stride(arg314_1, (1024, ), (1, ))
    assert_size_stride(arg315_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg318_1, (1024, ), (1, ))
    assert_size_stride(arg319_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg320_1, (1024, ), (1, ))
    assert_size_stride(arg321_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg322_1, (1024, ), (1, ))
    assert_size_stride(arg323_1, (1024, ), (1, ))
    assert_size_stride(arg324_1, (1024, ), (1, ))
    assert_size_stride(arg325_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg326_1, (4096, ), (1, ))
    assert_size_stride(arg327_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg328_1, (1024, ), (1, ))
    assert_size_stride(arg329_1, (1024, ), (1, ))
    assert_size_stride(arg330_1, (1024, ), (1, ))
    assert_size_stride(arg331_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg332_1, (1024, ), (1, ))
    assert_size_stride(arg333_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg334_1, (1024, ), (1, ))
    assert_size_stride(arg335_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg336_1, (1024, ), (1, ))
    assert_size_stride(arg337_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg338_1, (1024, ), (1, ))
    assert_size_stride(arg339_1, (1024, ), (1, ))
    assert_size_stride(arg340_1, (1024, ), (1, ))
    assert_size_stride(arg341_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg344_1, (1024, ), (1, ))
    assert_size_stride(arg345_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg348_1, (1024, ), (1, ))
    assert_size_stride(arg349_1, (1024, ), (1, ))
    assert_size_stride(arg350_1, (1024, ), (1, ))
    assert_size_stride(arg351_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg352_1, (4096, ), (1, ))
    assert_size_stride(arg353_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg354_1, (1024, ), (1, ))
    assert_size_stride(arg355_1, (1024, ), (1, ))
    assert_size_stride(arg356_1, (1024, ), (1, ))
    assert_size_stride(arg357_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg358_1, (1024, ), (1, ))
    assert_size_stride(arg359_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg364_1, (1024, ), (1, ))
    assert_size_stride(arg365_1, (1024, ), (1, ))
    assert_size_stride(arg366_1, (1024, ), (1, ))
    assert_size_stride(arg367_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg368_1, (1024, ), (1, ))
    assert_size_stride(arg369_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg370_1, (1024, ), (1, ))
    assert_size_stride(arg371_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg372_1, (1024, ), (1, ))
    assert_size_stride(arg373_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg374_1, (1024, ), (1, ))
    assert_size_stride(arg375_1, (1024, ), (1, ))
    assert_size_stride(arg376_1, (1024, ), (1, ))
    assert_size_stride(arg377_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg378_1, (4096, ), (1, ))
    assert_size_stride(arg379_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg380_1, (1024, ), (1, ))
    assert_size_stride(arg381_1, (1024, ), (1, ))
    assert_size_stride(arg382_1, (1024, ), (1, ))
    assert_size_stride(arg383_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg384_1, (1024, ), (1, ))
    assert_size_stride(arg385_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg386_1, (1024, ), (1, ))
    assert_size_stride(arg387_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg388_1, (1024, ), (1, ))
    assert_size_stride(arg389_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg390_1, (1024, ), (1, ))
    assert_size_stride(arg391_1, (1024, ), (1, ))
    assert_size_stride(arg392_1, (1024, ), (1, ))
    assert_size_stride(arg393_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg394_1, (1024, ), (1, ))
    assert_size_stride(arg395_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg396_1, (1024, ), (1, ))
    assert_size_stride(arg397_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg398_1, (1024, ), (1, ))
    assert_size_stride(arg399_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg400_1, (1024, ), (1, ))
    assert_size_stride(arg401_1, (1024, ), (1, ))
    assert_size_stride(arg402_1, (1024, ), (1, ))
    assert_size_stride(arg403_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg404_1, (4096, ), (1, ))
    assert_size_stride(arg405_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg406_1, (1024, ), (1, ))
    assert_size_stride(arg407_1, (1024, ), (1, ))
    assert_size_stride(arg408_1, (1024, ), (1, ))
    assert_size_stride(arg409_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg410_1, (1024, ), (1, ))
    assert_size_stride(arg411_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg412_1, (1024, ), (1, ))
    assert_size_stride(arg413_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg414_1, (1024, ), (1, ))
    assert_size_stride(arg415_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg416_1, (1024, ), (1, ))
    assert_size_stride(arg417_1, (1024, ), (1, ))
    assert_size_stride(arg418_1, (1024, ), (1, ))
    assert_size_stride(arg419_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg420_1, (1024, ), (1, ))
    assert_size_stride(arg421_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg422_1, (1024, ), (1, ))
    assert_size_stride(arg423_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg424_1, (1024, ), (1, ))
    assert_size_stride(arg425_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg426_1, (1024, ), (1, ))
    assert_size_stride(arg427_1, (1024, ), (1, ))
    assert_size_stride(arg428_1, (1024, ), (1, ))
    assert_size_stride(arg429_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg430_1, (4096, ), (1, ))
    assert_size_stride(arg431_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg432_1, (1024, ), (1, ))
    assert_size_stride(arg433_1, (1024, ), (1, ))
    assert_size_stride(arg434_1, (1024, ), (1, ))
    assert_size_stride(arg435_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg436_1, (1024, ), (1, ))
    assert_size_stride(arg437_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg438_1, (1024, ), (1, ))
    assert_size_stride(arg439_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg440_1, (1024, ), (1, ))
    assert_size_stride(arg441_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg442_1, (1024, ), (1, ))
    assert_size_stride(arg443_1, (1024, ), (1, ))
    assert_size_stride(arg444_1, (1024, ), (1, ))
    assert_size_stride(arg445_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg446_1, (1024, ), (1, ))
    assert_size_stride(arg447_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg448_1, (1024, ), (1, ))
    assert_size_stride(arg449_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg450_1, (1024, ), (1, ))
    assert_size_stride(arg451_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg452_1, (1024, ), (1, ))
    assert_size_stride(arg453_1, (1024, ), (1, ))
    assert_size_stride(arg454_1, (1024, ), (1, ))
    assert_size_stride(arg455_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg456_1, (4096, ), (1, ))
    assert_size_stride(arg457_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg458_1, (1024, ), (1, ))
    assert_size_stride(arg459_1, (1024, ), (1, ))
    assert_size_stride(arg460_1, (1024, ), (1, ))
    assert_size_stride(arg461_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg462_1, (1024, ), (1, ))
    assert_size_stride(arg463_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg464_1, (1024, ), (1, ))
    assert_size_stride(arg465_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg466_1, (1024, ), (1, ))
    assert_size_stride(arg467_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg468_1, (1024, ), (1, ))
    assert_size_stride(arg469_1, (1024, ), (1, ))
    assert_size_stride(arg470_1, (1024, ), (1, ))
    assert_size_stride(arg471_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg472_1, (1024, ), (1, ))
    assert_size_stride(arg473_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg474_1, (1024, ), (1, ))
    assert_size_stride(arg475_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg476_1, (1024, ), (1, ))
    assert_size_stride(arg477_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg478_1, (1024, ), (1, ))
    assert_size_stride(arg479_1, (1024, ), (1, ))
    assert_size_stride(arg480_1, (1024, ), (1, ))
    assert_size_stride(arg481_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg482_1, (4096, ), (1, ))
    assert_size_stride(arg483_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg484_1, (1024, ), (1, ))
    assert_size_stride(arg485_1, (1024, ), (1, ))
    assert_size_stride(arg486_1, (1024, ), (1, ))
    assert_size_stride(arg487_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg488_1, (1024, ), (1, ))
    assert_size_stride(arg489_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg490_1, (1024, ), (1, ))
    assert_size_stride(arg491_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg492_1, (1024, ), (1, ))
    assert_size_stride(arg493_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg494_1, (1024, ), (1, ))
    assert_size_stride(arg495_1, (1024, ), (1, ))
    assert_size_stride(arg496_1, (1024, ), (1, ))
    assert_size_stride(arg497_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg498_1, (1024, ), (1, ))
    assert_size_stride(arg499_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg500_1, (1024, ), (1, ))
    assert_size_stride(arg501_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg502_1, (1024, ), (1, ))
    assert_size_stride(arg503_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg504_1, (1024, ), (1, ))
    assert_size_stride(arg505_1, (1024, ), (1, ))
    assert_size_stride(arg506_1, (1024, ), (1, ))
    assert_size_stride(arg507_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg508_1, (4096, ), (1, ))
    assert_size_stride(arg509_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg510_1, (1024, ), (1, ))
    assert_size_stride(arg511_1, (1024, ), (1, ))
    assert_size_stride(arg512_1, (1024, ), (1, ))
    assert_size_stride(arg513_1, (1, 50265), (50265, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((2, 1024, 1024), (1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, add, embed_pos, hidden_states, hidden_states_1], Original ATen: [aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, buf3, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg1_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg6_1, (1024, 1024), (1, 1024), 0), out=buf4)
        del arg6_1
        buf5 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg8_1, (1024, 1024), (1, 1024), 0), out=buf5)
        del arg8_1
        buf6 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg10_1, (1024, 1024), (1, 1024), 0), out=buf6)
        del arg10_1
        buf7 = empty_strided_cuda((2, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf4, arg7_1, buf7, 2097152, grid=grid(2097152), stream=stream0)
        del arg7_1
        buf8 = reinterpret_tensor(buf4, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf5, arg9_1, buf8, 2097152, grid=grid(2097152), stream=stream0)
        del arg9_1
        buf9 = reinterpret_tensor(buf5, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf6, arg11_1, buf9, 2097152, grid=grid(2097152), stream=stream0)
        del arg11_1
        del buf6
        # Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf7, buf8, buf9, None, False)
        buf11 = buf10[0]
        del buf10
        buf15 = reinterpret_tensor(buf9, (2048, 1024), (1024, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg12_1, (1024, 1024), (1, 1024), 0), out=buf15)
        del arg12_1
        buf19 = reinterpret_tensor(buf11, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf3, buf15, arg13_1, arg14_1, arg15_1, buf19, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg13_1
        del arg14_1
        del arg15_1
        buf20 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg16_1, (1024, 4096), (1, 1024), 0), out=buf20)
        del arg16_1
        buf21 = reinterpret_tensor(buf20, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf21, arg17_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg17_1
        buf22 = reinterpret_tensor(buf3, (2048, 1024), (1024, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg18_1, (4096, 1024), (1, 4096), 0), out=buf22)
        del arg18_1
        buf26 = reinterpret_tensor(buf15, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf19, buf22, arg19_1, arg20_1, arg21_1, buf26, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg19_1
        del arg20_1
        del arg21_1
        buf27 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg22_1, (1024, 1024), (1, 1024), 0), out=buf27)
        del arg22_1
        buf28 = reinterpret_tensor(buf19, (2048, 1024), (1024, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg24_1, (1024, 1024), (1, 1024), 0), out=buf28)
        del arg24_1
        buf29 = reinterpret_tensor(buf8, (2048, 1024), (1024, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg26_1, (1024, 1024), (1, 1024), 0), out=buf29)
        del arg26_1
        buf30 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [query_states_3, key_states_1, value_states_1, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf27, arg23_1, buf30, 2097152, grid=grid(2097152), stream=stream0)
        del arg23_1
        buf31 = reinterpret_tensor(buf27, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [query_states_3, key_states_1, value_states_1, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf28, arg25_1, buf31, 2097152, grid=grid(2097152), stream=stream0)
        del arg25_1
        buf32 = reinterpret_tensor(buf28, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [query_states_3, key_states_1, value_states_1, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf29, arg27_1, buf32, 2097152, grid=grid(2097152), stream=stream0)
        del arg27_1
        del buf29
        # Topologically Sorted Source Nodes: [query_states_3, key_states_1, value_states_1, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf33 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf30, buf31, buf32, None, False)
        buf34 = buf33[0]
        del buf33
        buf38 = reinterpret_tensor(buf32, (2048, 1024), (1024, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg28_1, (1024, 1024), (1, 1024), 0), out=buf38)
        del arg28_1
        buf42 = reinterpret_tensor(buf34, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf26, buf38, arg29_1, arg30_1, arg31_1, buf42, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg29_1
        del arg30_1
        del arg31_1
        buf43 = reinterpret_tensor(buf21, (2048, 4096), (4096, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg32_1, (1024, 4096), (1, 1024), 0), out=buf43)
        del arg32_1
        buf44 = reinterpret_tensor(buf43, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf44, arg33_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg33_1
        buf45 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg34_1, (4096, 1024), (1, 4096), 0), out=buf45)
        del arg34_1
        buf49 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf42, buf45, arg35_1, arg36_1, arg37_1, buf49, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg35_1
        del arg36_1
        del arg37_1
        buf50 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg38_1, (1024, 1024), (1, 1024), 0), out=buf50)
        del arg38_1
        buf51 = reinterpret_tensor(buf42, (2048, 1024), (1024, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg40_1, (1024, 1024), (1, 1024), 0), out=buf51)
        del arg40_1
        buf52 = reinterpret_tensor(buf31, (2048, 1024), (1024, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg42_1, (1024, 1024), (1, 1024), 0), out=buf52)
        del arg42_1
        buf53 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [query_states_5, key_states_2, value_states_2, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf50, arg39_1, buf53, 2097152, grid=grid(2097152), stream=stream0)
        del arg39_1
        buf54 = reinterpret_tensor(buf50, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [query_states_5, key_states_2, value_states_2, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf51, arg41_1, buf54, 2097152, grid=grid(2097152), stream=stream0)
        del arg41_1
        buf55 = reinterpret_tensor(buf51, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [query_states_5, key_states_2, value_states_2, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf52, arg43_1, buf55, 2097152, grid=grid(2097152), stream=stream0)
        del arg43_1
        del buf52
        # Topologically Sorted Source Nodes: [query_states_5, key_states_2, value_states_2, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf56 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf53, buf54, buf55, None, False)
        buf57 = buf56[0]
        del buf56
        buf61 = reinterpret_tensor(buf55, (2048, 1024), (1024, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg44_1, (1024, 1024), (1, 1024), 0), out=buf61)
        del arg44_1
        buf65 = reinterpret_tensor(buf57, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_22, hidden_states_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf49, buf61, arg45_1, arg46_1, arg47_1, buf65, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg45_1
        del arg46_1
        del arg47_1
        buf66 = reinterpret_tensor(buf44, (2048, 4096), (4096, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg48_1, (1024, 4096), (1, 1024), 0), out=buf66)
        del arg48_1
        buf67 = reinterpret_tensor(buf66, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf67, arg49_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg49_1
        buf68 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg50_1, (4096, 1024), (1, 4096), 0), out=buf68)
        del arg50_1
        buf72 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf65, buf68, arg51_1, arg52_1, arg53_1, buf72, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg51_1
        del arg52_1
        del arg53_1
        buf73 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg54_1, (1024, 1024), (1, 1024), 0), out=buf73)
        del arg54_1
        buf74 = reinterpret_tensor(buf65, (2048, 1024), (1024, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg56_1, (1024, 1024), (1, 1024), 0), out=buf74)
        del arg56_1
        buf75 = reinterpret_tensor(buf54, (2048, 1024), (1024, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg58_1, (1024, 1024), (1, 1024), 0), out=buf75)
        del arg58_1
        buf76 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [query_states_7, key_states_3, value_states_3, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf73, arg55_1, buf76, 2097152, grid=grid(2097152), stream=stream0)
        del arg55_1
        buf77 = reinterpret_tensor(buf73, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [query_states_7, key_states_3, value_states_3, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf74, arg57_1, buf77, 2097152, grid=grid(2097152), stream=stream0)
        del arg57_1
        buf78 = reinterpret_tensor(buf74, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [query_states_7, key_states_3, value_states_3, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf75, arg59_1, buf78, 2097152, grid=grid(2097152), stream=stream0)
        del arg59_1
        del buf75
        # Topologically Sorted Source Nodes: [query_states_7, key_states_3, value_states_3, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf79 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf76, buf77, buf78, None, False)
        buf80 = buf79[0]
        del buf79
        buf84 = reinterpret_tensor(buf78, (2048, 1024), (1024, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg60_1, (1024, 1024), (1, 1024), 0), out=buf84)
        del arg60_1
        buf88 = reinterpret_tensor(buf80, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf72, buf84, arg61_1, arg62_1, arg63_1, buf88, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        buf89 = reinterpret_tensor(buf67, (2048, 4096), (4096, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf88, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg64_1, (1024, 4096), (1, 1024), 0), out=buf89)
        del arg64_1
        buf90 = reinterpret_tensor(buf89, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_33], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf90, arg65_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg65_1
        buf91 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg66_1, (4096, 1024), (1, 4096), 0), out=buf91)
        del arg66_1
        buf95 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf88, buf91, arg67_1, arg68_1, arg69_1, buf95, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        buf96 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg70_1, (1024, 1024), (1, 1024), 0), out=buf96)
        del arg70_1
        buf97 = reinterpret_tensor(buf88, (2048, 1024), (1024, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg72_1, (1024, 1024), (1, 1024), 0), out=buf97)
        del arg72_1
        buf98 = reinterpret_tensor(buf77, (2048, 1024), (1024, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg74_1, (1024, 1024), (1, 1024), 0), out=buf98)
        del arg74_1
        buf99 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [query_states_9, key_states_4, value_states_4, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf96, arg71_1, buf99, 2097152, grid=grid(2097152), stream=stream0)
        del arg71_1
        buf100 = reinterpret_tensor(buf96, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [query_states_9, key_states_4, value_states_4, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf97, arg73_1, buf100, 2097152, grid=grid(2097152), stream=stream0)
        del arg73_1
        buf101 = reinterpret_tensor(buf97, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [query_states_9, key_states_4, value_states_4, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf98, arg75_1, buf101, 2097152, grid=grid(2097152), stream=stream0)
        del arg75_1
        del buf98
        # Topologically Sorted Source Nodes: [query_states_9, key_states_4, value_states_4, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf102 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf99, buf100, buf101, None, False)
        buf103 = buf102[0]
        del buf102
        buf107 = reinterpret_tensor(buf99, (2048, 1024), (1024, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg76_1, (1024, 1024), (1, 1024), 0), out=buf107)
        del arg76_1
        buf111 = reinterpret_tensor(buf103, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf95, buf107, arg77_1, arg78_1, arg79_1, buf111, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        buf112 = reinterpret_tensor(buf90, (2048, 4096), (4096, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg80_1, (1024, 4096), (1, 1024), 0), out=buf112)
        del arg80_1
        buf113 = reinterpret_tensor(buf112, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf113, arg81_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg81_1
        buf114 = reinterpret_tensor(buf95, (2048, 1024), (1024, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf113, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg82_1, (4096, 1024), (1, 4096), 0), out=buf114)
        del arg82_1
        buf118 = reinterpret_tensor(buf107, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf111, buf114, arg83_1, arg84_1, arg85_1, buf118, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg83_1
        del arg84_1
        del arg85_1
        buf119 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg86_1, (1024, 1024), (1, 1024), 0), out=buf119)
        del arg86_1
        buf120 = reinterpret_tensor(buf111, (2048, 1024), (1024, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg88_1, (1024, 1024), (1, 1024), 0), out=buf120)
        del arg88_1
        buf121 = reinterpret_tensor(buf101, (2048, 1024), (1024, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg90_1, (1024, 1024), (1, 1024), 0), out=buf121)
        del arg90_1
        buf122 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [query_states_11, key_states_5, value_states_5, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf119, arg87_1, buf122, 2097152, grid=grid(2097152), stream=stream0)
        del arg87_1
        buf123 = reinterpret_tensor(buf119, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [query_states_11, key_states_5, value_states_5, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf120, arg89_1, buf123, 2097152, grid=grid(2097152), stream=stream0)
        del arg89_1
        buf124 = reinterpret_tensor(buf120, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [query_states_11, key_states_5, value_states_5, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf121, arg91_1, buf124, 2097152, grid=grid(2097152), stream=stream0)
        del arg91_1
        del buf121
        # Topologically Sorted Source Nodes: [query_states_11, key_states_5, value_states_5, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf125 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf122, buf123, buf124, None, False)
        buf126 = buf125[0]
        del buf125
        buf130 = reinterpret_tensor(buf124, (2048, 1024), (1024, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg92_1, (1024, 1024), (1, 1024), 0), out=buf130)
        del arg92_1
        buf134 = reinterpret_tensor(buf126, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf118, buf130, arg93_1, arg94_1, arg95_1, buf134, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg93_1
        del arg94_1
        del arg95_1
        buf135 = reinterpret_tensor(buf113, (2048, 4096), (4096, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf134, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg96_1, (1024, 4096), (1, 1024), 0), out=buf135)
        del arg96_1
        buf136 = reinterpret_tensor(buf135, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf136, arg97_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg97_1
        buf137 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg98_1, (4096, 1024), (1, 4096), 0), out=buf137)
        del arg98_1
        buf141 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf134, buf137, arg99_1, arg100_1, arg101_1, buf141, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg100_1
        del arg101_1
        del arg99_1
        buf142 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg102_1, (1024, 1024), (1, 1024), 0), out=buf142)
        del arg102_1
        buf143 = reinterpret_tensor(buf134, (2048, 1024), (1024, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg104_1, (1024, 1024), (1, 1024), 0), out=buf143)
        del arg104_1
        buf144 = reinterpret_tensor(buf123, (2048, 1024), (1024, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg106_1, (1024, 1024), (1, 1024), 0), out=buf144)
        del arg106_1
        buf145 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [query_states_13, key_states_6, value_states_6, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf142, arg103_1, buf145, 2097152, grid=grid(2097152), stream=stream0)
        del arg103_1
        buf146 = reinterpret_tensor(buf142, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [query_states_13, key_states_6, value_states_6, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf143, arg105_1, buf146, 2097152, grid=grid(2097152), stream=stream0)
        del arg105_1
        buf147 = reinterpret_tensor(buf143, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [query_states_13, key_states_6, value_states_6, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf144, arg107_1, buf147, 2097152, grid=grid(2097152), stream=stream0)
        del arg107_1
        del buf144
        # Topologically Sorted Source Nodes: [query_states_13, key_states_6, value_states_6, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf148 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf145, buf146, buf147, None, False)
        buf149 = buf148[0]
        del buf148
        buf153 = reinterpret_tensor(buf147, (2048, 1024), (1024, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg108_1, (1024, 1024), (1, 1024), 0), out=buf153)
        del arg108_1
        buf157 = reinterpret_tensor(buf149, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf141, buf153, arg109_1, arg110_1, arg111_1, buf157, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg109_1
        del arg110_1
        del arg111_1
        buf158 = reinterpret_tensor(buf136, (2048, 4096), (4096, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg112_1, (1024, 4096), (1, 1024), 0), out=buf158)
        del arg112_1
        buf159 = reinterpret_tensor(buf158, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_60], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf159, arg113_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg113_1
        buf160 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg114_1, (4096, 1024), (1, 4096), 0), out=buf160)
        del arg114_1
        buf164 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf157, buf160, arg115_1, arg116_1, arg117_1, buf164, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg115_1
        del arg116_1
        del arg117_1
        buf165 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg118_1, (1024, 1024), (1, 1024), 0), out=buf165)
        del arg118_1
        buf166 = reinterpret_tensor(buf157, (2048, 1024), (1024, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg120_1, (1024, 1024), (1, 1024), 0), out=buf166)
        del arg120_1
        buf167 = reinterpret_tensor(buf146, (2048, 1024), (1024, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg122_1, (1024, 1024), (1, 1024), 0), out=buf167)
        del arg122_1
        buf168 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [query_states_15, key_states_7, value_states_7, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf165, arg119_1, buf168, 2097152, grid=grid(2097152), stream=stream0)
        del arg119_1
        buf169 = reinterpret_tensor(buf165, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [query_states_15, key_states_7, value_states_7, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf166, arg121_1, buf169, 2097152, grid=grid(2097152), stream=stream0)
        del arg121_1
        buf170 = reinterpret_tensor(buf166, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [query_states_15, key_states_7, value_states_7, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf167, arg123_1, buf170, 2097152, grid=grid(2097152), stream=stream0)
        del arg123_1
        del buf167
        # Topologically Sorted Source Nodes: [query_states_15, key_states_7, value_states_7, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf171 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf168, buf169, buf170, None, False)
        buf172 = buf171[0]
        del buf171
        buf176 = reinterpret_tensor(buf170, (2048, 1024), (1024, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg124_1, (1024, 1024), (1, 1024), 0), out=buf176)
        del arg124_1
        buf180 = reinterpret_tensor(buf172, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf164, buf176, arg125_1, arg126_1, arg127_1, buf180, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg125_1
        del arg126_1
        del arg127_1
        buf181 = reinterpret_tensor(buf159, (2048, 4096), (4096, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg128_1, (1024, 4096), (1, 1024), 0), out=buf181)
        del arg128_1
        buf182 = reinterpret_tensor(buf181, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf182, arg129_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg129_1
        buf183 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg130_1, (4096, 1024), (1, 4096), 0), out=buf183)
        del arg130_1
        buf187 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf180, buf183, arg131_1, arg132_1, arg133_1, buf187, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg131_1
        del arg132_1
        del arg133_1
        buf188 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg134_1, (1024, 1024), (1, 1024), 0), out=buf188)
        del arg134_1
        buf189 = reinterpret_tensor(buf180, (2048, 1024), (1024, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg136_1, (1024, 1024), (1, 1024), 0), out=buf189)
        del arg136_1
        buf190 = reinterpret_tensor(buf169, (2048, 1024), (1024, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg138_1, (1024, 1024), (1, 1024), 0), out=buf190)
        del arg138_1
        buf191 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [query_states_17, key_states_8, value_states_8, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf188, arg135_1, buf191, 2097152, grid=grid(2097152), stream=stream0)
        del arg135_1
        buf192 = reinterpret_tensor(buf188, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [query_states_17, key_states_8, value_states_8, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf189, arg137_1, buf192, 2097152, grid=grid(2097152), stream=stream0)
        del arg137_1
        buf193 = reinterpret_tensor(buf189, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [query_states_17, key_states_8, value_states_8, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf190, arg139_1, buf193, 2097152, grid=grid(2097152), stream=stream0)
        del arg139_1
        del buf190
        # Topologically Sorted Source Nodes: [query_states_17, key_states_8, value_states_8, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf194 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf191, buf192, buf193, None, False)
        buf195 = buf194[0]
        del buf194
        buf199 = reinterpret_tensor(buf193, (2048, 1024), (1024, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg140_1, (1024, 1024), (1, 1024), 0), out=buf199)
        del arg140_1
        buf203 = reinterpret_tensor(buf195, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76, hidden_states_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf187, buf199, arg141_1, arg142_1, arg143_1, buf203, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg141_1
        del arg142_1
        del arg143_1
        buf204 = reinterpret_tensor(buf182, (2048, 4096), (4096, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg144_1, (1024, 4096), (1, 1024), 0), out=buf204)
        del arg144_1
        buf205 = reinterpret_tensor(buf204, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf205, arg145_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg145_1
        buf206 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf205, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg146_1, (4096, 1024), (1, 4096), 0), out=buf206)
        del arg146_1
        buf210 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82, hidden_states_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf203, buf206, arg147_1, arg148_1, arg149_1, buf210, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        buf211 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg150_1, (1024, 1024), (1, 1024), 0), out=buf211)
        del arg150_1
        buf212 = reinterpret_tensor(buf203, (2048, 1024), (1024, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg152_1, (1024, 1024), (1, 1024), 0), out=buf212)
        del arg152_1
        buf213 = reinterpret_tensor(buf192, (2048, 1024), (1024, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg154_1, (1024, 1024), (1, 1024), 0), out=buf213)
        del arg154_1
        buf214 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [query_states_19, key_states_9, value_states_9, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf211, arg151_1, buf214, 2097152, grid=grid(2097152), stream=stream0)
        del arg151_1
        buf215 = reinterpret_tensor(buf211, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [query_states_19, key_states_9, value_states_9, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf212, arg153_1, buf215, 2097152, grid=grid(2097152), stream=stream0)
        del arg153_1
        buf216 = reinterpret_tensor(buf212, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [query_states_19, key_states_9, value_states_9, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf213, arg155_1, buf216, 2097152, grid=grid(2097152), stream=stream0)
        del arg155_1
        del buf213
        # Topologically Sorted Source Nodes: [query_states_19, key_states_9, value_states_9, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf217 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf214, buf215, buf216, None, False)
        buf218 = buf217[0]
        del buf217
        buf222 = reinterpret_tensor(buf216, (2048, 1024), (1024, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg156_1, (1024, 1024), (1, 1024), 0), out=buf222)
        del arg156_1
        buf226 = reinterpret_tensor(buf218, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf210, buf222, arg157_1, arg158_1, arg159_1, buf226, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        buf227 = reinterpret_tensor(buf205, (2048, 4096), (4096, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg160_1, (1024, 4096), (1, 1024), 0), out=buf227)
        del arg160_1
        buf228 = reinterpret_tensor(buf227, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf228, arg161_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg161_1
        buf229 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg162_1, (4096, 1024), (1, 4096), 0), out=buf229)
        del arg162_1
        buf233 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_92], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf226, buf229, arg163_1, arg164_1, arg165_1, buf233, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg163_1
        del arg164_1
        del arg165_1
        buf234 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg166_1, (1024, 1024), (1, 1024), 0), out=buf234)
        del arg166_1
        buf235 = reinterpret_tensor(buf226, (2048, 1024), (1024, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg168_1, (1024, 1024), (1, 1024), 0), out=buf235)
        del arg168_1
        buf236 = reinterpret_tensor(buf215, (2048, 1024), (1024, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg170_1, (1024, 1024), (1, 1024), 0), out=buf236)
        del arg170_1
        buf237 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [query_states_21, key_states_10, value_states_10, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf234, arg167_1, buf237, 2097152, grid=grid(2097152), stream=stream0)
        del arg167_1
        buf238 = reinterpret_tensor(buf234, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [query_states_21, key_states_10, value_states_10, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf235, arg169_1, buf238, 2097152, grid=grid(2097152), stream=stream0)
        del arg169_1
        buf239 = reinterpret_tensor(buf235, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [query_states_21, key_states_10, value_states_10, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf236, arg171_1, buf239, 2097152, grid=grid(2097152), stream=stream0)
        del arg171_1
        del buf236
        # Topologically Sorted Source Nodes: [query_states_21, key_states_10, value_states_10, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf240 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf237, buf238, buf239, None, False)
        buf241 = buf240[0]
        del buf240
        buf245 = reinterpret_tensor(buf239, (2048, 1024), (1024, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf241, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg172_1, (1024, 1024), (1, 1024), 0), out=buf245)
        del arg172_1
        buf249 = reinterpret_tensor(buf241, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf233, buf245, arg173_1, arg174_1, arg175_1, buf249, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg173_1
        del arg174_1
        del arg175_1
        buf250 = reinterpret_tensor(buf228, (2048, 4096), (4096, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg176_1, (1024, 4096), (1, 1024), 0), out=buf250)
        del arg176_1
        buf251 = reinterpret_tensor(buf250, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf251, arg177_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg177_1
        buf252 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf251, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg178_1, (4096, 1024), (1, 4096), 0), out=buf252)
        del arg178_1
        buf256 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf249, buf252, arg179_1, arg180_1, arg181_1, buf256, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg179_1
        del arg180_1
        del arg181_1
        buf257 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg182_1, (1024, 1024), (1, 1024), 0), out=buf257)
        del arg182_1
        buf258 = reinterpret_tensor(buf249, (2048, 1024), (1024, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg184_1, (1024, 1024), (1, 1024), 0), out=buf258)
        del arg184_1
        buf259 = reinterpret_tensor(buf238, (2048, 1024), (1024, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg186_1, (1024, 1024), (1, 1024), 0), out=buf259)
        del arg186_1
        buf260 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [query_states_23, key_states_11, value_states_11, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf257, arg183_1, buf260, 2097152, grid=grid(2097152), stream=stream0)
        del arg183_1
        buf261 = reinterpret_tensor(buf257, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [query_states_23, key_states_11, value_states_11, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf258, arg185_1, buf261, 2097152, grid=grid(2097152), stream=stream0)
        del arg185_1
        buf262 = reinterpret_tensor(buf258, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [query_states_23, key_states_11, value_states_11, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf259, arg187_1, buf262, 2097152, grid=grid(2097152), stream=stream0)
        del arg187_1
        # Topologically Sorted Source Nodes: [query_states_23, key_states_11, value_states_11, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf263 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf260, buf261, buf262, None, False)
        buf264 = buf263[0]
        del buf263
        buf268 = reinterpret_tensor(buf262, (2048, 1024), (1024, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf264, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg188_1, (1024, 1024), (1, 1024), 0), out=buf268)
        del arg188_1
        buf272 = reinterpret_tensor(buf264, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_103, hidden_states_104], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf256, buf268, arg189_1, arg190_1, arg191_1, buf272, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg189_1
        del arg190_1
        del arg191_1
        buf273 = reinterpret_tensor(buf251, (2048, 4096), (4096, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg192_1, (1024, 4096), (1, 1024), 0), out=buf273)
        del arg192_1
        buf274 = reinterpret_tensor(buf273, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_105], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf274, arg193_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg193_1
        buf275 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg194_1, (4096, 1024), (1, 4096), 0), out=buf275)
        del arg194_1
        buf302 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_109, hidden_states_110], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf272, buf275, arg195_1, arg196_1, arg197_1, buf302, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg195_1
        del arg196_1
        del arg197_1
        buf282 = reinterpret_tensor(buf275, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf275  # reuse
        buf283 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [shifted_input_ids, clone, setitem, setitem_1, eq, masked_fill_, embedding_2, inputs_embeds_1, add_27, positions_2, hidden_states_111, hidden_states_112], Original ATen: [aten.new_zeros, aten.clone, aten.copy, aten.lift_fresh, aten.fill, aten.eq, aten.masked_fill, aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_copy_embedding_eq_fill_lift_fresh_masked_fill_mul_native_layer_norm_new_zeros_4.run(buf283, arg0_1, arg2_1, arg198_1, arg199_1, arg200_1, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg198_1
        del arg199_1
        del arg200_1
        buf284 = reinterpret_tensor(buf272, (2048, 1024), (1024, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf283, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg201_1, (1024, 1024), (1, 1024), 0), out=buf284)
        del arg201_1
        buf285 = reinterpret_tensor(buf261, (2048, 1024), (1024, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf283, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg203_1, (1024, 1024), (1, 1024), 0), out=buf285)
        del arg203_1
        buf286 = reinterpret_tensor(buf260, (2048, 1024), (1024, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf283, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg205_1, (1024, 1024), (1, 1024), 0), out=buf286)
        del arg205_1
        buf287 = reinterpret_tensor(buf259, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [query_states_25, key_states_12, value_states_12, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf284, arg202_1, buf287, 2097152, grid=grid(2097152), stream=stream0)
        del arg202_1
        buf288 = reinterpret_tensor(buf284, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [query_states_25, key_states_12, value_states_12, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf285, arg204_1, buf288, 2097152, grid=grid(2097152), stream=stream0)
        del arg204_1
        buf289 = reinterpret_tensor(buf285, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [query_states_25, key_states_12, value_states_12, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf286, arg206_1, buf289, 2097152, grid=grid(2097152), stream=stream0)
        del arg206_1
        del buf286
        buf290 = empty_strided_cuda((2, 16, 1024, 1024), (16777216, 1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [query_states_25, key_states_12, value_states_12, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_5.run(buf290, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_25, key_states_12, value_states_12, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf291 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf287, buf288, buf289, buf290, False)
        buf292 = buf291[0]
        del buf291
        buf296 = reinterpret_tensor(buf289, (2048, 1024), (1024, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg207_1, (1024, 1024), (1, 1024), 0), out=buf296)
        del arg207_1
        buf300 = reinterpret_tensor(buf292, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_115, hidden_states_116], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf283, buf296, arg208_1, arg209_1, arg210_1, buf300, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg208_1
        del arg209_1
        del arg210_1
        buf301 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf300, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg211_1, (1024, 1024), (1, 1024), 0), out=buf301)
        del arg211_1
        buf303 = reinterpret_tensor(buf283, (2048, 1024), (1024, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg213_1, (1024, 1024), (1, 1024), 0), out=buf303)
        del arg213_1
        buf304 = reinterpret_tensor(buf288, (2048, 1024), (1024, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg215_1, (1024, 1024), (1, 1024), 0), out=buf304)
        del arg215_1
        buf305 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [query_states_27, key_states_13, value_states_13, attn_output_52], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf301, arg212_1, buf305, 2097152, grid=grid(2097152), stream=stream0)
        del arg212_1
        buf306 = reinterpret_tensor(buf301, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [query_states_27, key_states_13, value_states_13, attn_output_52], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf303, arg214_1, buf306, 2097152, grid=grid(2097152), stream=stream0)
        del arg214_1
        buf307 = reinterpret_tensor(buf303, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [query_states_27, key_states_13, value_states_13, attn_output_52], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf304, arg216_1, buf307, 2097152, grid=grid(2097152), stream=stream0)
        del arg216_1
        del buf304
        # Topologically Sorted Source Nodes: [query_states_27, key_states_13, value_states_13, attn_output_52], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf308 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf305, buf306, buf307, None, False)
        buf309 = buf308[0]
        del buf308
        buf313 = reinterpret_tensor(buf307, (2048, 1024), (1024, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf309, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg217_1, (1024, 1024), (1, 1024), 0), out=buf313)
        del arg217_1
        buf317 = reinterpret_tensor(buf309, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_118, hidden_states_119], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf300, buf313, arg218_1, arg219_1, arg220_1, buf317, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg218_1
        del arg219_1
        del arg220_1
        buf318 = reinterpret_tensor(buf274, (2048, 4096), (4096, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg221_1, (1024, 4096), (1, 1024), 0), out=buf318)
        del arg221_1
        buf319 = reinterpret_tensor(buf318, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_120], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf319, arg222_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg222_1
        buf320 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf319, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg223_1, (4096, 1024), (1, 4096), 0), out=buf320)
        del arg223_1
        buf324 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_124, hidden_states_125], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf317, buf320, arg224_1, arg225_1, arg226_1, buf324, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg224_1
        del arg225_1
        del arg226_1
        buf325 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg227_1, (1024, 1024), (1, 1024), 0), out=buf325)
        del arg227_1
        buf326 = reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg229_1, (1024, 1024), (1, 1024), 0), out=buf326)
        del arg229_1
        buf327 = reinterpret_tensor(buf306, (2048, 1024), (1024, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg231_1, (1024, 1024), (1, 1024), 0), out=buf327)
        del arg231_1
        buf328 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [query_states_29, key_states_14, value_states_14, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf325, arg228_1, buf328, 2097152, grid=grid(2097152), stream=stream0)
        del arg228_1
        buf329 = reinterpret_tensor(buf325, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [query_states_29, key_states_14, value_states_14, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf326, arg230_1, buf329, 2097152, grid=grid(2097152), stream=stream0)
        del arg230_1
        buf330 = reinterpret_tensor(buf326, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [query_states_29, key_states_14, value_states_14, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf327, arg232_1, buf330, 2097152, grid=grid(2097152), stream=stream0)
        del arg232_1
        del buf327
        buf331 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [query_states_29, key_states_14, value_states_14, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_5.run(buf331, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_29, key_states_14, value_states_14, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf332 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf328, buf329, buf330, buf331, False)
        buf333 = buf332[0]
        del buf332
        buf337 = reinterpret_tensor(buf330, (2048, 1024), (1024, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf333, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg233_1, (1024, 1024), (1, 1024), 0), out=buf337)
        del arg233_1
        buf341 = reinterpret_tensor(buf333, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_127, hidden_states_128], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf324, buf337, arg234_1, arg235_1, arg236_1, buf341, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg234_1
        del arg235_1
        del arg236_1
        buf342 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf341, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg237_1, (1024, 1024), (1, 1024), 0), out=buf342)
        del arg237_1
        buf343 = reinterpret_tensor(buf324, (2048, 1024), (1024, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg239_1, (1024, 1024), (1, 1024), 0), out=buf343)
        del arg239_1
        buf344 = reinterpret_tensor(buf329, (2048, 1024), (1024, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg241_1, (1024, 1024), (1, 1024), 0), out=buf344)
        del arg241_1
        buf345 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [query_states_31, key_states_15, value_states_15, attn_output_60], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf342, arg238_1, buf345, 2097152, grid=grid(2097152), stream=stream0)
        del arg238_1
        buf346 = reinterpret_tensor(buf342, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [query_states_31, key_states_15, value_states_15, attn_output_60], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf343, arg240_1, buf346, 2097152, grid=grid(2097152), stream=stream0)
        del arg240_1
        buf347 = reinterpret_tensor(buf343, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [query_states_31, key_states_15, value_states_15, attn_output_60], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf344, arg242_1, buf347, 2097152, grid=grid(2097152), stream=stream0)
        del arg242_1
        del buf344
        # Topologically Sorted Source Nodes: [query_states_31, key_states_15, value_states_15, attn_output_60], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf348 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf345, buf346, buf347, None, False)
        buf349 = buf348[0]
        del buf348
        buf353 = reinterpret_tensor(buf347, (2048, 1024), (1024, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf349, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg243_1, (1024, 1024), (1, 1024), 0), out=buf353)
        del arg243_1
        buf357 = reinterpret_tensor(buf349, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf349  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_130, hidden_states_131], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf341, buf353, arg244_1, arg245_1, arg246_1, buf357, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg244_1
        del arg245_1
        del arg246_1
        buf358 = reinterpret_tensor(buf319, (2048, 4096), (4096, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf357, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg247_1, (1024, 4096), (1, 1024), 0), out=buf358)
        del arg247_1
        buf359 = reinterpret_tensor(buf358, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_132], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf359, arg248_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg248_1
        buf360 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf359, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg249_1, (4096, 1024), (1, 4096), 0), out=buf360)
        del arg249_1
        buf364 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_137], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf357, buf360, arg250_1, arg251_1, arg252_1, buf364, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg250_1
        del arg251_1
        del arg252_1
        buf365 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg253_1, (1024, 1024), (1, 1024), 0), out=buf365)
        del arg253_1
        buf366 = reinterpret_tensor(buf357, (2048, 1024), (1024, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg255_1, (1024, 1024), (1, 1024), 0), out=buf366)
        del arg255_1
        buf367 = reinterpret_tensor(buf346, (2048, 1024), (1024, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg257_1, (1024, 1024), (1, 1024), 0), out=buf367)
        del arg257_1
        buf368 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [query_states_33, key_states_16, value_states_16, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf365, arg254_1, buf368, 2097152, grid=grid(2097152), stream=stream0)
        del arg254_1
        buf369 = reinterpret_tensor(buf365, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [query_states_33, key_states_16, value_states_16, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf366, arg256_1, buf369, 2097152, grid=grid(2097152), stream=stream0)
        del arg256_1
        buf370 = reinterpret_tensor(buf366, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [query_states_33, key_states_16, value_states_16, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf367, arg258_1, buf370, 2097152, grid=grid(2097152), stream=stream0)
        del arg258_1
        del buf367
        buf371 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [query_states_33, key_states_16, value_states_16, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_5.run(buf371, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_33, key_states_16, value_states_16, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf372 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf368, buf369, buf370, buf371, False)
        buf373 = buf372[0]
        del buf372
        buf377 = reinterpret_tensor(buf370, (2048, 1024), (1024, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf373, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg259_1, (1024, 1024), (1, 1024), 0), out=buf377)
        del arg259_1
        buf381 = reinterpret_tensor(buf373, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_139, hidden_states_140], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf364, buf377, arg260_1, arg261_1, arg262_1, buf381, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg260_1
        del arg261_1
        del arg262_1
        buf382 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf381, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg263_1, (1024, 1024), (1, 1024), 0), out=buf382)
        del arg263_1
        buf383 = reinterpret_tensor(buf364, (2048, 1024), (1024, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg265_1, (1024, 1024), (1, 1024), 0), out=buf383)
        del arg265_1
        buf384 = reinterpret_tensor(buf369, (2048, 1024), (1024, 1), 0); del buf369  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg267_1, (1024, 1024), (1, 1024), 0), out=buf384)
        del arg267_1
        buf385 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [query_states_35, key_states_17, value_states_17, attn_output_68], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf382, arg264_1, buf385, 2097152, grid=grid(2097152), stream=stream0)
        del arg264_1
        buf386 = reinterpret_tensor(buf382, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [query_states_35, key_states_17, value_states_17, attn_output_68], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf383, arg266_1, buf386, 2097152, grid=grid(2097152), stream=stream0)
        del arg266_1
        buf387 = reinterpret_tensor(buf383, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [query_states_35, key_states_17, value_states_17, attn_output_68], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf384, arg268_1, buf387, 2097152, grid=grid(2097152), stream=stream0)
        del arg268_1
        del buf384
        # Topologically Sorted Source Nodes: [query_states_35, key_states_17, value_states_17, attn_output_68], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf388 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf385, buf386, buf387, None, False)
        buf389 = buf388[0]
        del buf388
        buf393 = reinterpret_tensor(buf387, (2048, 1024), (1024, 1), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg269_1, (1024, 1024), (1, 1024), 0), out=buf393)
        del arg269_1
        buf397 = reinterpret_tensor(buf389, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_142, hidden_states_143], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf381, buf393, arg270_1, arg271_1, arg272_1, buf397, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg270_1
        del arg271_1
        del arg272_1
        buf398 = reinterpret_tensor(buf359, (2048, 4096), (4096, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf397, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg273_1, (1024, 4096), (1, 1024), 0), out=buf398)
        del arg273_1
        buf399 = reinterpret_tensor(buf398, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_144], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf399, arg274_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg274_1
        buf400 = buf393; del buf393  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf399, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg275_1, (4096, 1024), (1, 4096), 0), out=buf400)
        del arg275_1
        buf404 = buf381; del buf381  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_148, hidden_states_149], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf397, buf400, arg276_1, arg277_1, arg278_1, buf404, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg276_1
        del arg277_1
        del arg278_1
        buf405 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg279_1, (1024, 1024), (1, 1024), 0), out=buf405)
        del arg279_1
        buf406 = reinterpret_tensor(buf397, (2048, 1024), (1024, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg281_1, (1024, 1024), (1, 1024), 0), out=buf406)
        del arg281_1
        buf407 = reinterpret_tensor(buf386, (2048, 1024), (1024, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg283_1, (1024, 1024), (1, 1024), 0), out=buf407)
        del arg283_1
        buf408 = buf385; del buf385  # reuse
        # Topologically Sorted Source Nodes: [query_states_37, key_states_18, value_states_18, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf405, arg280_1, buf408, 2097152, grid=grid(2097152), stream=stream0)
        del arg280_1
        buf409 = reinterpret_tensor(buf405, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [query_states_37, key_states_18, value_states_18, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf406, arg282_1, buf409, 2097152, grid=grid(2097152), stream=stream0)
        del arg282_1
        buf410 = reinterpret_tensor(buf406, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [query_states_37, key_states_18, value_states_18, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf407, arg284_1, buf410, 2097152, grid=grid(2097152), stream=stream0)
        del arg284_1
        del buf407
        buf411 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [query_states_37, key_states_18, value_states_18, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_5.run(buf411, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_37, key_states_18, value_states_18, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf412 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf408, buf409, buf410, buf411, False)
        buf413 = buf412[0]
        del buf412
        buf417 = reinterpret_tensor(buf410, (2048, 1024), (1024, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf413, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg285_1, (1024, 1024), (1, 1024), 0), out=buf417)
        del arg285_1
        buf421 = reinterpret_tensor(buf413, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_151, hidden_states_152], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf404, buf417, arg286_1, arg287_1, arg288_1, buf421, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg286_1
        del arg287_1
        del arg288_1
        buf422 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf421, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg289_1, (1024, 1024), (1, 1024), 0), out=buf422)
        del arg289_1
        buf423 = reinterpret_tensor(buf404, (2048, 1024), (1024, 1), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg291_1, (1024, 1024), (1, 1024), 0), out=buf423)
        del arg291_1
        buf424 = reinterpret_tensor(buf409, (2048, 1024), (1024, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg293_1, (1024, 1024), (1, 1024), 0), out=buf424)
        del arg293_1
        buf425 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [query_states_39, key_states_19, value_states_19, attn_output_76], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf422, arg290_1, buf425, 2097152, grid=grid(2097152), stream=stream0)
        del arg290_1
        buf426 = reinterpret_tensor(buf422, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf422  # reuse
        # Topologically Sorted Source Nodes: [query_states_39, key_states_19, value_states_19, attn_output_76], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf423, arg292_1, buf426, 2097152, grid=grid(2097152), stream=stream0)
        del arg292_1
        buf427 = reinterpret_tensor(buf423, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [query_states_39, key_states_19, value_states_19, attn_output_76], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf424, arg294_1, buf427, 2097152, grid=grid(2097152), stream=stream0)
        del arg294_1
        del buf424
        # Topologically Sorted Source Nodes: [query_states_39, key_states_19, value_states_19, attn_output_76], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf428 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf425, buf426, buf427, None, False)
        buf429 = buf428[0]
        del buf428
        buf433 = reinterpret_tensor(buf427, (2048, 1024), (1024, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf429, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg295_1, (1024, 1024), (1, 1024), 0), out=buf433)
        del arg295_1
        buf437 = reinterpret_tensor(buf429, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_154, hidden_states_155], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf421, buf433, arg296_1, arg297_1, arg298_1, buf437, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg296_1
        del arg297_1
        del arg298_1
        buf438 = reinterpret_tensor(buf399, (2048, 4096), (4096, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf437, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg299_1, (1024, 4096), (1, 1024), 0), out=buf438)
        del arg299_1
        buf439 = reinterpret_tensor(buf438, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf438  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_156], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf439, arg300_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg300_1
        buf440 = buf433; del buf433  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf439, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg301_1, (4096, 1024), (1, 4096), 0), out=buf440)
        del arg301_1
        buf444 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_160, hidden_states_161], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf437, buf440, arg302_1, arg303_1, arg304_1, buf444, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        buf445 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf444, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg305_1, (1024, 1024), (1, 1024), 0), out=buf445)
        del arg305_1
        buf446 = reinterpret_tensor(buf437, (2048, 1024), (1024, 1), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf444, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg307_1, (1024, 1024), (1, 1024), 0), out=buf446)
        del arg307_1
        buf447 = reinterpret_tensor(buf426, (2048, 1024), (1024, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf444, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg309_1, (1024, 1024), (1, 1024), 0), out=buf447)
        del arg309_1
        buf448 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [query_states_41, key_states_20, value_states_20, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf445, arg306_1, buf448, 2097152, grid=grid(2097152), stream=stream0)
        del arg306_1
        buf449 = reinterpret_tensor(buf445, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf445  # reuse
        # Topologically Sorted Source Nodes: [query_states_41, key_states_20, value_states_20, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf446, arg308_1, buf449, 2097152, grid=grid(2097152), stream=stream0)
        del arg308_1
        buf450 = reinterpret_tensor(buf446, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf446  # reuse
        # Topologically Sorted Source Nodes: [query_states_41, key_states_20, value_states_20, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf447, arg310_1, buf450, 2097152, grid=grid(2097152), stream=stream0)
        del arg310_1
        del buf447
        buf451 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [query_states_41, key_states_20, value_states_20, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_5.run(buf451, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_41, key_states_20, value_states_20, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf452 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf448, buf449, buf450, buf451, False)
        buf453 = buf452[0]
        del buf452
        buf457 = reinterpret_tensor(buf450, (2048, 1024), (1024, 1), 0); del buf450  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf453, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg311_1, (1024, 1024), (1, 1024), 0), out=buf457)
        del arg311_1
        buf461 = reinterpret_tensor(buf453, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_163, hidden_states_164], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf444, buf457, arg312_1, arg313_1, arg314_1, buf461, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg312_1
        del arg313_1
        del arg314_1
        buf462 = buf457; del buf457  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf461, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg315_1, (1024, 1024), (1, 1024), 0), out=buf462)
        del arg315_1
        buf463 = reinterpret_tensor(buf444, (2048, 1024), (1024, 1), 0); del buf444  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg317_1, (1024, 1024), (1, 1024), 0), out=buf463)
        del arg317_1
        buf464 = reinterpret_tensor(buf449, (2048, 1024), (1024, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg319_1, (1024, 1024), (1, 1024), 0), out=buf464)
        del arg319_1
        buf465 = buf448; del buf448  # reuse
        # Topologically Sorted Source Nodes: [query_states_43, key_states_21, value_states_21, attn_output_84], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf462, arg316_1, buf465, 2097152, grid=grid(2097152), stream=stream0)
        del arg316_1
        buf466 = reinterpret_tensor(buf462, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf462  # reuse
        # Topologically Sorted Source Nodes: [query_states_43, key_states_21, value_states_21, attn_output_84], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf463, arg318_1, buf466, 2097152, grid=grid(2097152), stream=stream0)
        del arg318_1
        buf467 = reinterpret_tensor(buf463, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf463  # reuse
        # Topologically Sorted Source Nodes: [query_states_43, key_states_21, value_states_21, attn_output_84], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf464, arg320_1, buf467, 2097152, grid=grid(2097152), stream=stream0)
        del arg320_1
        del buf464
        # Topologically Sorted Source Nodes: [query_states_43, key_states_21, value_states_21, attn_output_84], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf468 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf465, buf466, buf467, None, False)
        buf469 = buf468[0]
        del buf468
        buf473 = reinterpret_tensor(buf467, (2048, 1024), (1024, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf469, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg321_1, (1024, 1024), (1, 1024), 0), out=buf473)
        del arg321_1
        buf477 = reinterpret_tensor(buf469, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_166, hidden_states_167], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf461, buf473, arg322_1, arg323_1, arg324_1, buf477, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg322_1
        del arg323_1
        del arg324_1
        buf478 = reinterpret_tensor(buf439, (2048, 4096), (4096, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf477, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg325_1, (1024, 4096), (1, 1024), 0), out=buf478)
        del arg325_1
        buf479 = reinterpret_tensor(buf478, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_168], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf479, arg326_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg326_1
        buf480 = buf473; del buf473  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf479, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg327_1, (4096, 1024), (1, 4096), 0), out=buf480)
        del arg327_1
        buf484 = buf461; del buf461  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_172, hidden_states_173], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf477, buf480, arg328_1, arg329_1, arg330_1, buf484, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg328_1
        del arg329_1
        del arg330_1
        buf485 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf484, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg331_1, (1024, 1024), (1, 1024), 0), out=buf485)
        del arg331_1
        buf486 = reinterpret_tensor(buf477, (2048, 1024), (1024, 1), 0); del buf477  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf484, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg333_1, (1024, 1024), (1, 1024), 0), out=buf486)
        del arg333_1
        buf487 = reinterpret_tensor(buf466, (2048, 1024), (1024, 1), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf484, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg335_1, (1024, 1024), (1, 1024), 0), out=buf487)
        del arg335_1
        buf488 = buf465; del buf465  # reuse
        # Topologically Sorted Source Nodes: [query_states_45, key_states_22, value_states_22, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf485, arg332_1, buf488, 2097152, grid=grid(2097152), stream=stream0)
        del arg332_1
        buf489 = reinterpret_tensor(buf485, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf485  # reuse
        # Topologically Sorted Source Nodes: [query_states_45, key_states_22, value_states_22, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf486, arg334_1, buf489, 2097152, grid=grid(2097152), stream=stream0)
        del arg334_1
        buf490 = reinterpret_tensor(buf486, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf486  # reuse
        # Topologically Sorted Source Nodes: [query_states_45, key_states_22, value_states_22, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf487, arg336_1, buf490, 2097152, grid=grid(2097152), stream=stream0)
        del arg336_1
        del buf487
        buf491 = buf451; del buf451  # reuse
        # Topologically Sorted Source Nodes: [query_states_45, key_states_22, value_states_22, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_5.run(buf491, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_45, key_states_22, value_states_22, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf492 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf488, buf489, buf490, buf491, False)
        buf493 = buf492[0]
        del buf492
        buf497 = reinterpret_tensor(buf490, (2048, 1024), (1024, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf493, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg337_1, (1024, 1024), (1, 1024), 0), out=buf497)
        del arg337_1
        buf501 = reinterpret_tensor(buf493, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf493  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_175, hidden_states_176], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf484, buf497, arg338_1, arg339_1, arg340_1, buf501, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg338_1
        del arg339_1
        del arg340_1
        buf502 = buf497; del buf497  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf501, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg341_1, (1024, 1024), (1, 1024), 0), out=buf502)
        del arg341_1
        buf503 = reinterpret_tensor(buf484, (2048, 1024), (1024, 1), 0); del buf484  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg343_1, (1024, 1024), (1, 1024), 0), out=buf503)
        del arg343_1
        buf504 = reinterpret_tensor(buf489, (2048, 1024), (1024, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg345_1, (1024, 1024), (1, 1024), 0), out=buf504)
        del arg345_1
        buf505 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [query_states_47, key_states_23, value_states_23, attn_output_92], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf502, arg342_1, buf505, 2097152, grid=grid(2097152), stream=stream0)
        del arg342_1
        buf506 = reinterpret_tensor(buf502, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [query_states_47, key_states_23, value_states_23, attn_output_92], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf503, arg344_1, buf506, 2097152, grid=grid(2097152), stream=stream0)
        del arg344_1
        buf507 = reinterpret_tensor(buf503, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf503  # reuse
        # Topologically Sorted Source Nodes: [query_states_47, key_states_23, value_states_23, attn_output_92], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf504, arg346_1, buf507, 2097152, grid=grid(2097152), stream=stream0)
        del arg346_1
        del buf504
        # Topologically Sorted Source Nodes: [query_states_47, key_states_23, value_states_23, attn_output_92], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf508 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf505, buf506, buf507, None, False)
        buf509 = buf508[0]
        del buf508
        buf513 = reinterpret_tensor(buf507, (2048, 1024), (1024, 1), 0); del buf507  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf509, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg347_1, (1024, 1024), (1, 1024), 0), out=buf513)
        del arg347_1
        buf517 = reinterpret_tensor(buf509, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf509  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_178, hidden_states_179], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf501, buf513, arg348_1, arg349_1, arg350_1, buf517, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg348_1
        del arg349_1
        del arg350_1
        buf518 = reinterpret_tensor(buf479, (2048, 4096), (4096, 1), 0); del buf479  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf517, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg351_1, (1024, 4096), (1, 1024), 0), out=buf518)
        del arg351_1
        buf519 = reinterpret_tensor(buf518, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf518  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_180], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf519, arg352_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg352_1
        buf520 = buf513; del buf513  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf519, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg353_1, (4096, 1024), (1, 4096), 0), out=buf520)
        del arg353_1
        buf524 = buf501; del buf501  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_184, hidden_states_185], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf517, buf520, arg354_1, arg355_1, arg356_1, buf524, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg354_1
        del arg355_1
        del arg356_1
        buf525 = buf520; del buf520  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf524, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg357_1, (1024, 1024), (1, 1024), 0), out=buf525)
        del arg357_1
        buf526 = reinterpret_tensor(buf517, (2048, 1024), (1024, 1), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf524, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg359_1, (1024, 1024), (1, 1024), 0), out=buf526)
        del arg359_1
        buf527 = reinterpret_tensor(buf506, (2048, 1024), (1024, 1), 0); del buf506  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf524, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg361_1, (1024, 1024), (1, 1024), 0), out=buf527)
        del arg361_1
        buf528 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [query_states_49, key_states_24, value_states_24, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf525, arg358_1, buf528, 2097152, grid=grid(2097152), stream=stream0)
        del arg358_1
        buf529 = reinterpret_tensor(buf525, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf525  # reuse
        # Topologically Sorted Source Nodes: [query_states_49, key_states_24, value_states_24, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf526, arg360_1, buf529, 2097152, grid=grid(2097152), stream=stream0)
        del arg360_1
        buf530 = reinterpret_tensor(buf526, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf526  # reuse
        # Topologically Sorted Source Nodes: [query_states_49, key_states_24, value_states_24, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf527, arg362_1, buf530, 2097152, grid=grid(2097152), stream=stream0)
        del arg362_1
        del buf527
        buf531 = buf491; del buf491  # reuse
        # Topologically Sorted Source Nodes: [query_states_49, key_states_24, value_states_24, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_5.run(buf531, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_49, key_states_24, value_states_24, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf532 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf528, buf529, buf530, buf531, False)
        buf533 = buf532[0]
        del buf532
        buf537 = reinterpret_tensor(buf530, (2048, 1024), (1024, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf533, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg363_1, (1024, 1024), (1, 1024), 0), out=buf537)
        del arg363_1
        buf541 = reinterpret_tensor(buf533, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf533  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_187, hidden_states_188], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf524, buf537, arg364_1, arg365_1, arg366_1, buf541, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg364_1
        del arg365_1
        del arg366_1
        buf542 = buf537; del buf537  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf541, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg367_1, (1024, 1024), (1, 1024), 0), out=buf542)
        del arg367_1
        buf543 = reinterpret_tensor(buf524, (2048, 1024), (1024, 1), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg369_1, (1024, 1024), (1, 1024), 0), out=buf543)
        del arg369_1
        buf544 = reinterpret_tensor(buf529, (2048, 1024), (1024, 1), 0); del buf529  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg371_1, (1024, 1024), (1, 1024), 0), out=buf544)
        del arg371_1
        buf545 = buf528; del buf528  # reuse
        # Topologically Sorted Source Nodes: [query_states_51, key_states_25, value_states_25, attn_output_100], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf542, arg368_1, buf545, 2097152, grid=grid(2097152), stream=stream0)
        del arg368_1
        buf546 = reinterpret_tensor(buf542, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf542  # reuse
        # Topologically Sorted Source Nodes: [query_states_51, key_states_25, value_states_25, attn_output_100], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf543, arg370_1, buf546, 2097152, grid=grid(2097152), stream=stream0)
        del arg370_1
        buf547 = reinterpret_tensor(buf543, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [query_states_51, key_states_25, value_states_25, attn_output_100], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf544, arg372_1, buf547, 2097152, grid=grid(2097152), stream=stream0)
        del arg372_1
        del buf544
        # Topologically Sorted Source Nodes: [query_states_51, key_states_25, value_states_25, attn_output_100], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf548 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf545, buf546, buf547, None, False)
        buf549 = buf548[0]
        del buf548
        buf553 = reinterpret_tensor(buf547, (2048, 1024), (1024, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf549, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg373_1, (1024, 1024), (1, 1024), 0), out=buf553)
        del arg373_1
        buf557 = reinterpret_tensor(buf549, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf549  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_190, hidden_states_191], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf541, buf553, arg374_1, arg375_1, arg376_1, buf557, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg374_1
        del arg375_1
        del arg376_1
        buf558 = reinterpret_tensor(buf519, (2048, 4096), (4096, 1), 0); del buf519  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf557, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg377_1, (1024, 4096), (1, 1024), 0), out=buf558)
        del arg377_1
        buf559 = reinterpret_tensor(buf558, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf558  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_192], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf559, arg378_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg378_1
        buf560 = buf553; del buf553  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf559, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg379_1, (4096, 1024), (1, 4096), 0), out=buf560)
        del arg379_1
        buf564 = buf541; del buf541  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_196, hidden_states_197], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf557, buf560, arg380_1, arg381_1, arg382_1, buf564, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg380_1
        del arg381_1
        del arg382_1
        buf565 = buf560; del buf560  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf564, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg383_1, (1024, 1024), (1, 1024), 0), out=buf565)
        del arg383_1
        buf566 = reinterpret_tensor(buf557, (2048, 1024), (1024, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf564, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg385_1, (1024, 1024), (1, 1024), 0), out=buf566)
        del arg385_1
        buf567 = reinterpret_tensor(buf546, (2048, 1024), (1024, 1), 0); del buf546  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf564, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg387_1, (1024, 1024), (1, 1024), 0), out=buf567)
        del arg387_1
        buf568 = buf545; del buf545  # reuse
        # Topologically Sorted Source Nodes: [query_states_53, key_states_26, value_states_26, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf565, arg384_1, buf568, 2097152, grid=grid(2097152), stream=stream0)
        del arg384_1
        buf569 = reinterpret_tensor(buf565, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf565  # reuse
        # Topologically Sorted Source Nodes: [query_states_53, key_states_26, value_states_26, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf566, arg386_1, buf569, 2097152, grid=grid(2097152), stream=stream0)
        del arg386_1
        buf570 = reinterpret_tensor(buf566, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf566  # reuse
        # Topologically Sorted Source Nodes: [query_states_53, key_states_26, value_states_26, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf567, arg388_1, buf570, 2097152, grid=grid(2097152), stream=stream0)
        del arg388_1
        del buf567
        buf571 = buf531; del buf531  # reuse
        # Topologically Sorted Source Nodes: [query_states_53, key_states_26, value_states_26, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_5.run(buf571, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_53, key_states_26, value_states_26, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf572 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf568, buf569, buf570, buf571, False)
        buf573 = buf572[0]
        del buf572
        buf577 = reinterpret_tensor(buf570, (2048, 1024), (1024, 1), 0); del buf570  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf573, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg389_1, (1024, 1024), (1, 1024), 0), out=buf577)
        del arg389_1
        buf581 = reinterpret_tensor(buf573, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf573  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_199, hidden_states_200], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf564, buf577, arg390_1, arg391_1, arg392_1, buf581, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg390_1
        del arg391_1
        del arg392_1
        buf582 = buf577; del buf577  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf581, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg393_1, (1024, 1024), (1, 1024), 0), out=buf582)
        del arg393_1
        buf583 = reinterpret_tensor(buf564, (2048, 1024), (1024, 1), 0); del buf564  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg395_1, (1024, 1024), (1, 1024), 0), out=buf583)
        del arg395_1
        buf584 = reinterpret_tensor(buf569, (2048, 1024), (1024, 1), 0); del buf569  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg397_1, (1024, 1024), (1, 1024), 0), out=buf584)
        del arg397_1
        buf585 = buf568; del buf568  # reuse
        # Topologically Sorted Source Nodes: [query_states_55, key_states_27, value_states_27, attn_output_108], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf582, arg394_1, buf585, 2097152, grid=grid(2097152), stream=stream0)
        del arg394_1
        buf586 = reinterpret_tensor(buf582, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf582  # reuse
        # Topologically Sorted Source Nodes: [query_states_55, key_states_27, value_states_27, attn_output_108], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf583, arg396_1, buf586, 2097152, grid=grid(2097152), stream=stream0)
        del arg396_1
        buf587 = reinterpret_tensor(buf583, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [query_states_55, key_states_27, value_states_27, attn_output_108], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf584, arg398_1, buf587, 2097152, grid=grid(2097152), stream=stream0)
        del arg398_1
        del buf584
        # Topologically Sorted Source Nodes: [query_states_55, key_states_27, value_states_27, attn_output_108], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf588 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf585, buf586, buf587, None, False)
        buf589 = buf588[0]
        del buf588
        buf593 = reinterpret_tensor(buf587, (2048, 1024), (1024, 1), 0); del buf587  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf589, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg399_1, (1024, 1024), (1, 1024), 0), out=buf593)
        del arg399_1
        buf597 = reinterpret_tensor(buf589, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf589  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_202, hidden_states_203], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf581, buf593, arg400_1, arg401_1, arg402_1, buf597, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg400_1
        del arg401_1
        del arg402_1
        buf598 = reinterpret_tensor(buf559, (2048, 4096), (4096, 1), 0); del buf559  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf597, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg403_1, (1024, 4096), (1, 1024), 0), out=buf598)
        del arg403_1
        buf599 = reinterpret_tensor(buf598, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf598  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_204], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf599, arg404_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg404_1
        buf600 = buf593; del buf593  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf599, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg405_1, (4096, 1024), (1, 4096), 0), out=buf600)
        del arg405_1
        buf604 = buf581; del buf581  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_208, hidden_states_209], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf597, buf600, arg406_1, arg407_1, arg408_1, buf604, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg406_1
        del arg407_1
        del arg408_1
        buf605 = buf600; del buf600  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf604, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg409_1, (1024, 1024), (1, 1024), 0), out=buf605)
        del arg409_1
        buf606 = reinterpret_tensor(buf597, (2048, 1024), (1024, 1), 0); del buf597  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf604, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg411_1, (1024, 1024), (1, 1024), 0), out=buf606)
        del arg411_1
        buf607 = reinterpret_tensor(buf586, (2048, 1024), (1024, 1), 0); del buf586  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf604, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg413_1, (1024, 1024), (1, 1024), 0), out=buf607)
        del arg413_1
        buf608 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [query_states_57, key_states_28, value_states_28, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf605, arg410_1, buf608, 2097152, grid=grid(2097152), stream=stream0)
        del arg410_1
        buf609 = reinterpret_tensor(buf605, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [query_states_57, key_states_28, value_states_28, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf606, arg412_1, buf609, 2097152, grid=grid(2097152), stream=stream0)
        del arg412_1
        buf610 = reinterpret_tensor(buf606, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf606  # reuse
        # Topologically Sorted Source Nodes: [query_states_57, key_states_28, value_states_28, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf607, arg414_1, buf610, 2097152, grid=grid(2097152), stream=stream0)
        del arg414_1
        del buf607
        buf611 = buf571; del buf571  # reuse
        # Topologically Sorted Source Nodes: [query_states_57, key_states_28, value_states_28, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_5.run(buf611, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_57, key_states_28, value_states_28, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf612 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf608, buf609, buf610, buf611, False)
        buf613 = buf612[0]
        del buf612
        buf617 = reinterpret_tensor(buf610, (2048, 1024), (1024, 1), 0); del buf610  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf613, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg415_1, (1024, 1024), (1, 1024), 0), out=buf617)
        del arg415_1
        buf621 = reinterpret_tensor(buf613, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_211, hidden_states_212], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf604, buf617, arg416_1, arg417_1, arg418_1, buf621, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg416_1
        del arg417_1
        del arg418_1
        buf622 = buf617; del buf617  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf621, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg419_1, (1024, 1024), (1, 1024), 0), out=buf622)
        del arg419_1
        buf623 = reinterpret_tensor(buf604, (2048, 1024), (1024, 1), 0); del buf604  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg421_1, (1024, 1024), (1, 1024), 0), out=buf623)
        del arg421_1
        buf624 = reinterpret_tensor(buf609, (2048, 1024), (1024, 1), 0); del buf609  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg423_1, (1024, 1024), (1, 1024), 0), out=buf624)
        del arg423_1
        buf625 = buf608; del buf608  # reuse
        # Topologically Sorted Source Nodes: [query_states_59, key_states_29, value_states_29, attn_output_116], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf622, arg420_1, buf625, 2097152, grid=grid(2097152), stream=stream0)
        del arg420_1
        buf626 = reinterpret_tensor(buf622, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf622  # reuse
        # Topologically Sorted Source Nodes: [query_states_59, key_states_29, value_states_29, attn_output_116], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf623, arg422_1, buf626, 2097152, grid=grid(2097152), stream=stream0)
        del arg422_1
        buf627 = reinterpret_tensor(buf623, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf623  # reuse
        # Topologically Sorted Source Nodes: [query_states_59, key_states_29, value_states_29, attn_output_116], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf624, arg424_1, buf627, 2097152, grid=grid(2097152), stream=stream0)
        del arg424_1
        del buf624
        # Topologically Sorted Source Nodes: [query_states_59, key_states_29, value_states_29, attn_output_116], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf628 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf625, buf626, buf627, None, False)
        buf629 = buf628[0]
        del buf628
        buf633 = reinterpret_tensor(buf627, (2048, 1024), (1024, 1), 0); del buf627  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf629, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg425_1, (1024, 1024), (1, 1024), 0), out=buf633)
        del arg425_1
        buf637 = reinterpret_tensor(buf629, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf629  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_214, hidden_states_215], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf621, buf633, arg426_1, arg427_1, arg428_1, buf637, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg426_1
        del arg427_1
        del arg428_1
        buf638 = reinterpret_tensor(buf599, (2048, 4096), (4096, 1), 0); del buf599  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf637, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg429_1, (1024, 4096), (1, 1024), 0), out=buf638)
        del arg429_1
        buf639 = reinterpret_tensor(buf638, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf638  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_216], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf639, arg430_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg430_1
        buf640 = buf633; del buf633  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf639, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg431_1, (4096, 1024), (1, 4096), 0), out=buf640)
        del arg431_1
        buf644 = buf621; del buf621  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_220, hidden_states_221], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf637, buf640, arg432_1, arg433_1, arg434_1, buf644, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg432_1
        del arg433_1
        del arg434_1
        buf645 = buf640; del buf640  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf644, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg435_1, (1024, 1024), (1, 1024), 0), out=buf645)
        del arg435_1
        buf646 = reinterpret_tensor(buf637, (2048, 1024), (1024, 1), 0); del buf637  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf644, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg437_1, (1024, 1024), (1, 1024), 0), out=buf646)
        del arg437_1
        buf647 = reinterpret_tensor(buf626, (2048, 1024), (1024, 1), 0); del buf626  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf644, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg439_1, (1024, 1024), (1, 1024), 0), out=buf647)
        del arg439_1
        buf648 = buf625; del buf625  # reuse
        # Topologically Sorted Source Nodes: [query_states_61, key_states_30, value_states_30, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf645, arg436_1, buf648, 2097152, grid=grid(2097152), stream=stream0)
        del arg436_1
        buf649 = reinterpret_tensor(buf645, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf645  # reuse
        # Topologically Sorted Source Nodes: [query_states_61, key_states_30, value_states_30, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf646, arg438_1, buf649, 2097152, grid=grid(2097152), stream=stream0)
        del arg438_1
        buf650 = reinterpret_tensor(buf646, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf646  # reuse
        # Topologically Sorted Source Nodes: [query_states_61, key_states_30, value_states_30, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf647, arg440_1, buf650, 2097152, grid=grid(2097152), stream=stream0)
        del arg440_1
        del buf647
        buf651 = buf611; del buf611  # reuse
        # Topologically Sorted Source Nodes: [query_states_61, key_states_30, value_states_30, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_5.run(buf651, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_61, key_states_30, value_states_30, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf652 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf648, buf649, buf650, buf651, False)
        buf653 = buf652[0]
        del buf652
        buf657 = reinterpret_tensor(buf650, (2048, 1024), (1024, 1), 0); del buf650  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf653, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg441_1, (1024, 1024), (1, 1024), 0), out=buf657)
        del arg441_1
        buf661 = reinterpret_tensor(buf653, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf653  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_223, hidden_states_224], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf644, buf657, arg442_1, arg443_1, arg444_1, buf661, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg442_1
        del arg443_1
        del arg444_1
        buf662 = buf657; del buf657  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf661, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg445_1, (1024, 1024), (1, 1024), 0), out=buf662)
        del arg445_1
        buf663 = reinterpret_tensor(buf644, (2048, 1024), (1024, 1), 0); del buf644  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg447_1, (1024, 1024), (1, 1024), 0), out=buf663)
        del arg447_1
        buf664 = reinterpret_tensor(buf649, (2048, 1024), (1024, 1), 0); del buf649  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg449_1, (1024, 1024), (1, 1024), 0), out=buf664)
        del arg449_1
        buf665 = buf648; del buf648  # reuse
        # Topologically Sorted Source Nodes: [query_states_63, key_states_31, value_states_31, attn_output_124], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf662, arg446_1, buf665, 2097152, grid=grid(2097152), stream=stream0)
        del arg446_1
        buf666 = reinterpret_tensor(buf662, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf662  # reuse
        # Topologically Sorted Source Nodes: [query_states_63, key_states_31, value_states_31, attn_output_124], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf663, arg448_1, buf666, 2097152, grid=grid(2097152), stream=stream0)
        del arg448_1
        buf667 = reinterpret_tensor(buf663, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf663  # reuse
        # Topologically Sorted Source Nodes: [query_states_63, key_states_31, value_states_31, attn_output_124], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf664, arg450_1, buf667, 2097152, grid=grid(2097152), stream=stream0)
        del arg450_1
        del buf664
        # Topologically Sorted Source Nodes: [query_states_63, key_states_31, value_states_31, attn_output_124], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf668 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf665, buf666, buf667, None, False)
        buf669 = buf668[0]
        del buf668
        buf673 = reinterpret_tensor(buf667, (2048, 1024), (1024, 1), 0); del buf667  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf669, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg451_1, (1024, 1024), (1, 1024), 0), out=buf673)
        del arg451_1
        buf677 = reinterpret_tensor(buf669, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf669  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_226, hidden_states_227], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf661, buf673, arg452_1, arg453_1, arg454_1, buf677, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg452_1
        del arg453_1
        del arg454_1
        buf678 = reinterpret_tensor(buf639, (2048, 4096), (4096, 1), 0); del buf639  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf677, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg455_1, (1024, 4096), (1, 1024), 0), out=buf678)
        del arg455_1
        buf679 = reinterpret_tensor(buf678, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf678  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_228], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf679, arg456_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg456_1
        buf680 = buf673; del buf673  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf679, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg457_1, (4096, 1024), (1, 4096), 0), out=buf680)
        del arg457_1
        buf684 = buf661; del buf661  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_232, hidden_states_233], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf677, buf680, arg458_1, arg459_1, arg460_1, buf684, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg458_1
        del arg459_1
        del arg460_1
        buf685 = buf680; del buf680  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf684, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg461_1, (1024, 1024), (1, 1024), 0), out=buf685)
        del arg461_1
        buf686 = reinterpret_tensor(buf677, (2048, 1024), (1024, 1), 0); del buf677  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf684, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg463_1, (1024, 1024), (1, 1024), 0), out=buf686)
        del arg463_1
        buf687 = reinterpret_tensor(buf666, (2048, 1024), (1024, 1), 0); del buf666  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf684, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg465_1, (1024, 1024), (1, 1024), 0), out=buf687)
        del arg465_1
        buf688 = buf665; del buf665  # reuse
        # Topologically Sorted Source Nodes: [query_states_65, key_states_32, value_states_32, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf685, arg462_1, buf688, 2097152, grid=grid(2097152), stream=stream0)
        del arg462_1
        buf689 = reinterpret_tensor(buf685, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf685  # reuse
        # Topologically Sorted Source Nodes: [query_states_65, key_states_32, value_states_32, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf686, arg464_1, buf689, 2097152, grid=grid(2097152), stream=stream0)
        del arg464_1
        buf690 = reinterpret_tensor(buf686, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf686  # reuse
        # Topologically Sorted Source Nodes: [query_states_65, key_states_32, value_states_32, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf687, arg466_1, buf690, 2097152, grid=grid(2097152), stream=stream0)
        del arg466_1
        del buf687
        buf691 = buf651; del buf651  # reuse
        # Topologically Sorted Source Nodes: [query_states_65, key_states_32, value_states_32, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_5.run(buf691, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_65, key_states_32, value_states_32, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf692 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf688, buf689, buf690, buf691, False)
        buf693 = buf692[0]
        del buf692
        buf697 = reinterpret_tensor(buf690, (2048, 1024), (1024, 1), 0); del buf690  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf693, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg467_1, (1024, 1024), (1, 1024), 0), out=buf697)
        del arg467_1
        buf701 = reinterpret_tensor(buf693, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf693  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_235, hidden_states_236], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf684, buf697, arg468_1, arg469_1, arg470_1, buf701, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg468_1
        del arg469_1
        del arg470_1
        buf702 = buf697; del buf697  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf701, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg471_1, (1024, 1024), (1, 1024), 0), out=buf702)
        del arg471_1
        buf703 = reinterpret_tensor(buf684, (2048, 1024), (1024, 1), 0); del buf684  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg473_1, (1024, 1024), (1, 1024), 0), out=buf703)
        del arg473_1
        buf704 = reinterpret_tensor(buf689, (2048, 1024), (1024, 1), 0); del buf689  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg475_1, (1024, 1024), (1, 1024), 0), out=buf704)
        del arg475_1
        buf705 = buf688; del buf688  # reuse
        # Topologically Sorted Source Nodes: [query_states_67, key_states_33, value_states_33, attn_output_132], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf702, arg472_1, buf705, 2097152, grid=grid(2097152), stream=stream0)
        del arg472_1
        buf706 = reinterpret_tensor(buf702, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf702  # reuse
        # Topologically Sorted Source Nodes: [query_states_67, key_states_33, value_states_33, attn_output_132], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf703, arg474_1, buf706, 2097152, grid=grid(2097152), stream=stream0)
        del arg474_1
        buf707 = reinterpret_tensor(buf703, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf703  # reuse
        # Topologically Sorted Source Nodes: [query_states_67, key_states_33, value_states_33, attn_output_132], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf704, arg476_1, buf707, 2097152, grid=grid(2097152), stream=stream0)
        del arg476_1
        del buf704
        # Topologically Sorted Source Nodes: [query_states_67, key_states_33, value_states_33, attn_output_132], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf708 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf705, buf706, buf707, None, False)
        buf709 = buf708[0]
        del buf708
        buf713 = reinterpret_tensor(buf707, (2048, 1024), (1024, 1), 0); del buf707  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf709, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg477_1, (1024, 1024), (1, 1024), 0), out=buf713)
        del arg477_1
        buf717 = reinterpret_tensor(buf709, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf709  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_238, hidden_states_239], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf701, buf713, arg478_1, arg479_1, arg480_1, buf717, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg478_1
        del arg479_1
        del arg480_1
        buf718 = reinterpret_tensor(buf679, (2048, 4096), (4096, 1), 0); del buf679  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf717, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg481_1, (1024, 4096), (1, 1024), 0), out=buf718)
        del arg481_1
        buf719 = reinterpret_tensor(buf718, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf718  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_240], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf719, arg482_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg482_1
        buf720 = buf713; del buf713  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf719, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg483_1, (4096, 1024), (1, 4096), 0), out=buf720)
        del arg483_1
        buf724 = buf701; del buf701  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_244, hidden_states_245], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf717, buf720, arg484_1, arg485_1, arg486_1, buf724, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg484_1
        del arg485_1
        del arg486_1
        buf725 = buf720; del buf720  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf724, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg487_1, (1024, 1024), (1, 1024), 0), out=buf725)
        del arg487_1
        buf726 = reinterpret_tensor(buf717, (2048, 1024), (1024, 1), 0); del buf717  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf724, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg489_1, (1024, 1024), (1, 1024), 0), out=buf726)
        del arg489_1
        buf727 = reinterpret_tensor(buf706, (2048, 1024), (1024, 1), 0); del buf706  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf724, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg491_1, (1024, 1024), (1, 1024), 0), out=buf727)
        del arg491_1
        buf728 = buf705; del buf705  # reuse
        # Topologically Sorted Source Nodes: [query_states_69, key_states_34, value_states_34, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf725, arg488_1, buf728, 2097152, grid=grid(2097152), stream=stream0)
        del arg488_1
        buf729 = reinterpret_tensor(buf725, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf725  # reuse
        # Topologically Sorted Source Nodes: [query_states_69, key_states_34, value_states_34, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf726, arg490_1, buf729, 2097152, grid=grid(2097152), stream=stream0)
        del arg490_1
        buf730 = reinterpret_tensor(buf726, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf726  # reuse
        # Topologically Sorted Source Nodes: [query_states_69, key_states_34, value_states_34, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf727, arg492_1, buf730, 2097152, grid=grid(2097152), stream=stream0)
        del arg492_1
        del buf727
        buf731 = buf691; del buf691  # reuse
        # Topologically Sorted Source Nodes: [query_states_69, key_states_34, value_states_34, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_5.run(buf731, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_69, key_states_34, value_states_34, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf732 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf728, buf729, buf730, buf731, False)
        del buf731
        buf733 = buf732[0]
        del buf732
        buf737 = reinterpret_tensor(buf730, (2048, 1024), (1024, 1), 0); del buf730  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf733, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg493_1, (1024, 1024), (1, 1024), 0), out=buf737)
        del arg493_1
        buf741 = reinterpret_tensor(buf733, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf733  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_247, hidden_states_248], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf724, buf737, arg494_1, arg495_1, arg496_1, buf741, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg494_1
        del arg495_1
        del arg496_1
        buf742 = buf737; del buf737  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf741, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg497_1, (1024, 1024), (1, 1024), 0), out=buf742)
        del arg497_1
        buf743 = reinterpret_tensor(buf724, (2048, 1024), (1024, 1), 0); del buf724  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg499_1, (1024, 1024), (1, 1024), 0), out=buf743)
        del arg499_1
        buf744 = reinterpret_tensor(buf729, (2048, 1024), (1024, 1), 0); del buf729  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg501_1, (1024, 1024), (1, 1024), 0), out=buf744)
        del arg501_1
        buf745 = buf728; del buf728  # reuse
        # Topologically Sorted Source Nodes: [query_states_71, key_states_35, value_states_35, attn_output_140], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf742, arg498_1, buf745, 2097152, grid=grid(2097152), stream=stream0)
        del arg498_1
        buf746 = reinterpret_tensor(buf742, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf742  # reuse
        # Topologically Sorted Source Nodes: [query_states_71, key_states_35, value_states_35, attn_output_140], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf743, arg500_1, buf746, 2097152, grid=grid(2097152), stream=stream0)
        del arg500_1
        buf747 = reinterpret_tensor(buf743, (2, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf743  # reuse
        # Topologically Sorted Source Nodes: [query_states_71, key_states_35, value_states_35, attn_output_140], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_1.run(buf744, arg502_1, buf747, 2097152, grid=grid(2097152), stream=stream0)
        del arg502_1
        del buf744
        # Topologically Sorted Source Nodes: [query_states_71, key_states_35, value_states_35, attn_output_140], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf748 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf745, buf746, buf747, None, False)
        del buf745
        del buf746
        buf749 = buf748[0]
        del buf748
        buf753 = reinterpret_tensor(buf747, (2048, 1024), (1024, 1), 0); del buf747  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf749, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg503_1, (1024, 1024), (1, 1024), 0), out=buf753)
        del arg503_1
        buf757 = reinterpret_tensor(buf749, (2, 1024, 1024), (1048576, 1024, 1), 0); del buf749  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_250, hidden_states_251], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf741, buf753, arg504_1, arg505_1, arg506_1, buf757, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg504_1
        del arg505_1
        del arg506_1
        buf758 = reinterpret_tensor(buf719, (2048, 4096), (4096, 1), 0); del buf719  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf757, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg507_1, (1024, 4096), (1, 1024), 0), out=buf758)
        del arg507_1
        buf759 = reinterpret_tensor(buf758, (2, 1024, 4096), (4194304, 4096, 1), 0); del buf758  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_252], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf759, arg508_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg508_1
        buf760 = buf753; del buf753  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf759, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg509_1, (4096, 1024), (1, 4096), 0), out=buf760)
        del arg509_1
        del buf759
        buf764 = buf741; del buf741  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_256, hidden_states_257], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf757, buf760, arg510_1, arg511_1, arg512_1, buf764, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg510_1
        del arg511_1
        del arg512_1
        del buf757
        del buf760
        buf765 = empty_strided_cuda((1024, 50268), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(arg2_1, buf765, 51474432, grid=grid(51474432), stream=stream0)
        del arg2_1
        buf766 = empty_strided_cuda((2048, 50268), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf764, (2048, 1024), (1024, 1), 0), buf765, out=buf766)
        del buf764
        del buf765
        buf767 = empty_strided_cuda((2, 1024, 50265), (51471360, 50265, 1), torch.float32)
        buf768 = empty_strided_cuda((2048, 1), (1, 2048), torch.float32)
        buf769 = empty_strided_cuda((2048, 1), (1, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [lm_logits_1, masked_lm_loss], Original ATen: [aten.add, aten._log_softmax]
        triton_red_fused__log_softmax_add_7.run(buf766, arg513_1, buf767, buf768, buf769, 2048, 50265, grid=grid(2048), stream=stream0)
        del arg513_1
        del buf766
        buf770 = empty_strided_cuda((), (), torch.float32)
        buf772 = buf770; del buf770  # reuse
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_8.run(buf772, arg0_1, buf767, buf768, buf769, 1, 2048, grid=grid(1), stream=stream0)
        del arg0_1
        del buf768
        del buf769
    return (buf772, buf767, buf302, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((2, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((1, 50265), (50265, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BartForConditionalGeneration', benchmark_compiled_module)
