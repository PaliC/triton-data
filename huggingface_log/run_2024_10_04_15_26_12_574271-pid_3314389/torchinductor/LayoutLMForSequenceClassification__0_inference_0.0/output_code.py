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


# kernel path: /tmp/torchinductor_sahanp/mh/cmh7l5mzr2wgxrejo7iju7a2stgemeg6ssxc33j4pcin4m2f26ml.py
# Topologically Sorted Source Nodes: [inputs_embeds, position_embeddings, add, left_position_embeddings, add_1, upper_position_embeddings, add_2, right_position_embeddings, add_3, lower_position_embeddings, add_4, sub_1, h_position_embeddings, add_5, sub_2, w_position_embeddings, add_6, token_type_ids, token_type_embeddings, embeddings, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.sub, aten.zeros, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   add_6 => add_6
#   embeddings => add_7
#   embeddings_1 => add_8, add_9, mul_1, mul_2, rsqrt, sub_3, var_mean
#   h_position_embeddings => embedding_6
#   inputs_embeds => embedding
#   left_position_embeddings => embedding_2
#   lower_position_embeddings => embedding_5
#   position_embeddings => embedding_1
#   right_position_embeddings => embedding_4
#   sub_1 => sub_1
#   sub_2 => sub_2
#   token_type_embeddings => embedding_8
#   token_type_ids => full_default
#   upper_position_embeddings => embedding_3
#   w_position_embeddings => embedding_7
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1, 0), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %arg204_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %select), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %embedding_3 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %select_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %embedding_3), kwargs = {})
#   %embedding_4 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %select_2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %embedding_4), kwargs = {})
#   %embedding_5 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %select_3), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %embedding_5), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_4, %select_5), kwargs = {})
#   %embedding_6 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %sub_1), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %embedding_6), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_6, %select_7), kwargs = {})
#   %embedding_7 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg6_1, %sub_2), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %embedding_7), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([16, 512], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %embedding_8 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg7_1, %full_default), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %embedding_8), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_7, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_1), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg8_1), kwargs = {})
#   %add_9 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg9_1), kwargs = {})
triton_per_fused_add_embedding_native_layer_norm_sub_zeros_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_sub_zeros_0', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 9, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr3, xnumel, rnumel):
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
    x3 = xindex
    r2 = rindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr9 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr10 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([RBLOCK], 30522, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 30522), "index out of bounds: 0 <= tmp4 < 30522")
    tmp6 = tl.load(in_ptr1 + (r2 + (768*tmp4)), rmask, other=0.0)
    tmp8 = tl.full([RBLOCK], 512, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 512), "index out of bounds: 0 <= tmp11 < 512")
    tmp13 = tl.load(in_ptr3 + (r2 + (768*tmp11)), rmask, other=0.0)
    tmp14 = tmp6 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp18 + tmp15
    tmp20 = tmp19 + tmp17
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask, tmp27, 0)
    tmp30 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp32 = tl.where(rmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = tl.full([1], 768, tl.int32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 / tmp35
    tmp37 = tmp27 - tmp36
    tmp38 = tmp37 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [RBLOCK])
    tmp41 = tl.where(rmask, tmp39, 0)
    tmp42 = triton_helpers.promote_to_tensor(tl.sum(tmp41, 0))
    tmp43 = tmp26 - tmp36
    tmp44 = 768.0
    tmp45 = tmp42 / tmp44
    tmp46 = 1e-12
    tmp47 = tmp45 + tmp46
    tmp48 = libdevice.rsqrt(tmp47)
    tmp49 = tmp43 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp22, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp53, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sx/csxbkuj4m7qzmz2iplste5s5cjuxlrf2n4bh44rr3mvzkfngali3.py
# Topologically Sorted Source Nodes: [add_9, hidden_states_2], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_9 => add_11
#   hidden_states_2 => add_12, add_13, mul_3, mul_4, rsqrt_1, sub_5, var_mean_1
# Graph fragment:
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %add_9), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_3), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-12), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg18_1), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg19_1), kwargs = {})
triton_per_fused_add_native_layer_norm_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_1', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ej/cejzuxyupw7uhdfqjv6uf24h36plommh6rn2jtlakoi6yhxlefen.py
# Topologically Sorted Source Nodes: [hidden_states_4], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_4 => add_14, erf, mul_5, mul_6, mul_7
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_14), kwargs = {})
triton_poi_fused_gelu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lt/cltxx2aissfdkzymvxyw6zi6hnib53kc7vnrv25b3l3btcrjo7ni.py
# Topologically Sorted Source Nodes: [pooled_output_1], Original ATen: [aten.tanh]
# Source node to ATen node mapping:
#   pooled_output_1 => tanh
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg203_1), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_tensor,), kwargs = {})
triton_poi_fused_tanh_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tanh_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.tanh(tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ws/cwsygfma43jtxzlz2bfjzv7ard6upu56cq4fdjayciq2wfmofnj2.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_133, %full_default_5], 1), kwargs = {})
triton_poi_fused_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + (768*x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 4, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nw/cnwpc6s47mwwopcprl25vbvwa4pfiy2yiztz4v54zkuellxnwt2n.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg206_1, %full_default_6],), kwargs = {})
triton_poi_fused_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 4, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wc/cwcjiv32ngc7hcnmf33ujmveohlksmdq2vws3vynwerdwp3kenf3.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type, div_24, full_default_4, ne_1, ne_2, neg, sum_14, sum_15, where_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_265, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_4), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_265, -100), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_14, torch.float32), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_15, %convert_element_type), kwargs = {})
triton_per_fused_nll_loss_forward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {3: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp11 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (1 + (4*r0)), None, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1, 1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tl.full([XBLOCK, RBLOCK], 2, tl.int32)
    tmp6 = tmp4 + tmp5
    tmp7 = tmp4 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp4)
    tl.device_assert((0 <= tmp8) & (tmp8 < 2), "index out of bounds: 0 <= tmp8 < 2")
    tmp10 = tl.load(in_ptr1 + (tmp8 + (4*r0)), None, eviction_policy='evict_last')
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = tmp10 - tmp13
    tmp15 = tmp11 - tmp13
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tmp12 - tmp13
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tmp16 + tmp18
    tmp20 = tl_math.log(tmp19)
    tmp21 = tmp14 - tmp20
    tmp22 = -tmp21
    tmp23 = 0.0
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None]
    tmp28 = tmp2.to(tl.int64)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.sum(tmp29, 1)[:, None]
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp27 / tmp32
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp33, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 512), (512, 1))
    assert_size_stride(arg1_1, (30522, 768), (768, 1))
    assert_size_stride(arg2_1, (512, 768), (768, 1))
    assert_size_stride(arg3_1, (1024, 768), (768, 1))
    assert_size_stride(arg4_1, (1024, 768), (768, 1))
    assert_size_stride(arg5_1, (1024, 768), (768, 1))
    assert_size_stride(arg6_1, (1024, 768), (768, 1))
    assert_size_stride(arg7_1, (2, 768), (768, 1))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, 768), (768, 1))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, 768), (768, 1))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, 768), (768, 1))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, 768), (768, 1))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (3072, 768), (768, 1))
    assert_size_stride(arg21_1, (3072, ), (1, ))
    assert_size_stride(arg22_1, (768, 3072), (3072, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, 768), (768, 1))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, 768), (768, 1))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, 768), (768, 1))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, 768), (768, 1))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (3072, 768), (768, 1))
    assert_size_stride(arg37_1, (3072, ), (1, ))
    assert_size_stride(arg38_1, (768, 3072), (3072, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, 768), (768, 1))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, 768), (768, 1))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, 768), (768, 1))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, 768), (768, 1))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (3072, 768), (768, 1))
    assert_size_stride(arg53_1, (3072, ), (1, ))
    assert_size_stride(arg54_1, (768, 3072), (3072, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, 768), (768, 1))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, 768), (768, 1))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, 768), (768, 1))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, 768), (768, 1))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (3072, 768), (768, 1))
    assert_size_stride(arg69_1, (3072, ), (1, ))
    assert_size_stride(arg70_1, (768, 3072), (3072, 1))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, 768), (768, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, 768), (768, 1))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, 768), (768, 1))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, 768), (768, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (3072, 768), (768, 1))
    assert_size_stride(arg85_1, (3072, ), (1, ))
    assert_size_stride(arg86_1, (768, 3072), (3072, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, 768), (768, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, 768), (768, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, 768), (768, 1))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, 768), (768, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (3072, 768), (768, 1))
    assert_size_stride(arg101_1, (3072, ), (1, ))
    assert_size_stride(arg102_1, (768, 3072), (3072, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, 768), (768, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, 768), (768, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, 768), (768, 1))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, 768), (768, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (3072, 768), (768, 1))
    assert_size_stride(arg117_1, (3072, ), (1, ))
    assert_size_stride(arg118_1, (768, 3072), (3072, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, 768), (768, 1))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, 768), (768, 1))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, 768), (768, 1))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, 768), (768, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (3072, 768), (768, 1))
    assert_size_stride(arg133_1, (3072, ), (1, ))
    assert_size_stride(arg134_1, (768, 3072), (3072, 1))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, 768), (768, 1))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, 768), (768, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, 768), (768, 1))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, 768), (768, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (3072, 768), (768, 1))
    assert_size_stride(arg149_1, (3072, ), (1, ))
    assert_size_stride(arg150_1, (768, 3072), (3072, 1))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (768, ), (1, ))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, 768), (768, 1))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, 768), (768, 1))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, 768), (768, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, 768), (768, 1))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (3072, 768), (768, 1))
    assert_size_stride(arg165_1, (3072, ), (1, ))
    assert_size_stride(arg166_1, (768, 3072), (3072, 1))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, 768), (768, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, 768), (768, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, 768), (768, 1))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (768, 768), (768, 1))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (3072, 768), (768, 1))
    assert_size_stride(arg181_1, (3072, ), (1, ))
    assert_size_stride(arg182_1, (768, 3072), (3072, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, 768), (768, 1))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, 768), (768, 1))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, 768), (768, 1))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (768, 768), (768, 1))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (3072, 768), (768, 1))
    assert_size_stride(arg197_1, (3072, ), (1, ))
    assert_size_stride(arg198_1, (768, 3072), (3072, 1))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (768, 768), (768, 1))
    assert_size_stride(arg203_1, (768, ), (1, ))
    assert_size_stride(arg204_1, (1, 512), (512, 1))
    assert_size_stride(arg205_1, (2, 768), (768, 1))
    assert_size_stride(arg206_1, (2, ), (1, ))
    assert_size_stride(arg207_1, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 512, 768), (393216, 768, 1), torch.float32)
        buf4 = empty_strided_cuda((16, 512, 768), (393216, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, position_embeddings, add, left_position_embeddings, add_1, upper_position_embeddings, add_2, right_position_embeddings, add_3, lower_position_embeddings, add_4, sub_1, h_position_embeddings, add_5, sub_2, w_position_embeddings, add_6, token_type_ids, token_type_embeddings, embeddings, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.sub, aten.zeros, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_sub_zeros_0.run(arg0_1, arg1_1, arg204_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, buf0, buf4, 8192, 768, grid=grid(8192), stream=stream0)
        del arg0_1
        del arg1_1
        del arg204_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf5 = reinterpret_tensor(buf0, (8192, 768), (768, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf4, (8192, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf5)
        del arg10_1
        del arg11_1
        buf6 = empty_strided_cuda((8192, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf4, (8192, 768), (768, 1), 0), reinterpret_tensor(arg12_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
        del arg12_1
        del arg13_1
        buf7 = empty_strided_cuda((8192, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf4, (8192, 768), (768, 1), 0), reinterpret_tensor(arg14_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
        del arg14_1
        del arg15_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf5, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf6, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf7, (16, 12, 512, 64), (393216, 64, 768, 1), 0), None, False, scale=0.125)
        del buf5
        buf9 = buf8[0]
        del buf8
        buf13 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf9, (8192, 768), (768, 1), 0), reinterpret_tensor(arg16_1, (768, 768), (1, 768), 0), out=buf13)
        del arg16_1
        buf17 = reinterpret_tensor(buf9, (16, 512, 768), (393216, 768, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [add_9, hidden_states_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf13, arg17_1, buf4, arg18_1, arg19_1, buf17, 8192, 768, grid=grid(8192), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        buf18 = empty_strided_cuda((8192, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf17, (8192, 768), (768, 1), 0), reinterpret_tensor(arg20_1, (768, 3072), (1, 768), 0), out=buf18)
        del arg20_1
        buf19 = reinterpret_tensor(buf18, (16, 512, 3072), (1572864, 3072, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf19, arg21_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg21_1
        buf20 = reinterpret_tensor(buf4, (8192, 768), (768, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg22_1, (3072, 768), (1, 3072), 0), out=buf20)
        del arg22_1
        buf24 = reinterpret_tensor(buf13, (16, 512, 768), (393216, 768, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [add_10, hidden_states_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf20, arg23_1, buf17, arg24_1, arg25_1, buf24, 8192, 768, grid=grid(8192), stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        buf25 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf24, (8192, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf25)
        del arg26_1
        del arg27_1
        buf26 = reinterpret_tensor(buf17, (8192, 768), (768, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg29_1, reinterpret_tensor(buf24, (8192, 768), (768, 1), 0), reinterpret_tensor(arg28_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf26)
        del arg28_1
        del arg29_1
        buf27 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg31_1, reinterpret_tensor(buf24, (8192, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf27)
        del arg30_1
        del arg31_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf28 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf25, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf26, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf27, (16, 12, 512, 64), (393216, 64, 768, 1), 0), None, False, scale=0.125)
        del buf25
        buf29 = buf28[0]
        del buf28
        buf33 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (8192, 768), (768, 1), 0), reinterpret_tensor(arg32_1, (768, 768), (1, 768), 0), out=buf33)
        del arg32_1
        buf37 = reinterpret_tensor(buf29, (16, 512, 768), (393216, 768, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [add_12, hidden_states_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf33, arg33_1, buf24, arg34_1, arg35_1, buf37, 8192, 768, grid=grid(8192), stream=stream0)
        del arg33_1
        del arg34_1
        del arg35_1
        buf38 = reinterpret_tensor(buf19, (8192, 3072), (3072, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf37, (8192, 768), (768, 1), 0), reinterpret_tensor(arg36_1, (768, 3072), (1, 768), 0), out=buf38)
        del arg36_1
        buf39 = reinterpret_tensor(buf38, (16, 512, 3072), (1572864, 3072, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_12], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf39, arg37_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg37_1
        buf40 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg38_1, (3072, 768), (1, 3072), 0), out=buf40)
        del arg38_1
        buf44 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [add_13, hidden_states_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf40, arg39_1, buf37, arg40_1, arg41_1, buf44, 8192, 768, grid=grid(8192), stream=stream0)
        del arg39_1
        del arg40_1
        del arg41_1
        buf45 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg43_1, reinterpret_tensor(buf44, (8192, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf45)
        del arg42_1
        del arg43_1
        buf46 = reinterpret_tensor(buf37, (8192, 768), (768, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg45_1, reinterpret_tensor(buf44, (8192, 768), (768, 1), 0), reinterpret_tensor(arg44_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf46)
        del arg44_1
        del arg45_1
        buf47 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg47_1, reinterpret_tensor(buf44, (8192, 768), (768, 1), 0), reinterpret_tensor(arg46_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf47)
        del arg46_1
        del arg47_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf48 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf45, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf46, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf47, (16, 12, 512, 64), (393216, 64, 768, 1), 0), None, False, scale=0.125)
        del buf45
        buf49 = buf48[0]
        del buf48
        buf53 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (8192, 768), (768, 1), 0), reinterpret_tensor(arg48_1, (768, 768), (1, 768), 0), out=buf53)
        del arg48_1
        buf57 = reinterpret_tensor(buf49, (16, 512, 768), (393216, 768, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [add_15, hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf53, arg49_1, buf44, arg50_1, arg51_1, buf57, 8192, 768, grid=grid(8192), stream=stream0)
        del arg49_1
        del arg50_1
        del arg51_1
        buf58 = reinterpret_tensor(buf39, (8192, 3072), (3072, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (8192, 768), (768, 1), 0), reinterpret_tensor(arg52_1, (768, 3072), (1, 768), 0), out=buf58)
        del arg52_1
        buf59 = reinterpret_tensor(buf58, (16, 512, 3072), (1572864, 3072, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf59, arg53_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg53_1
        buf60 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg54_1, (3072, 768), (1, 3072), 0), out=buf60)
        del arg54_1
        buf64 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [add_16, hidden_states_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf60, arg55_1, buf57, arg56_1, arg57_1, buf64, 8192, 768, grid=grid(8192), stream=stream0)
        del arg55_1
        del arg56_1
        del arg57_1
        buf65 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg59_1, reinterpret_tensor(buf64, (8192, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf65)
        del arg58_1
        del arg59_1
        buf66 = reinterpret_tensor(buf57, (8192, 768), (768, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg61_1, reinterpret_tensor(buf64, (8192, 768), (768, 1), 0), reinterpret_tensor(arg60_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf66)
        del arg60_1
        del arg61_1
        buf67 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg63_1, reinterpret_tensor(buf64, (8192, 768), (768, 1), 0), reinterpret_tensor(arg62_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf67)
        del arg62_1
        del arg63_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf68 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf65, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf66, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf67, (16, 12, 512, 64), (393216, 64, 768, 1), 0), None, False, scale=0.125)
        del buf65
        buf69 = buf68[0]
        del buf68
        buf73 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (8192, 768), (768, 1), 0), reinterpret_tensor(arg64_1, (768, 768), (1, 768), 0), out=buf73)
        del arg64_1
        buf77 = reinterpret_tensor(buf69, (16, 512, 768), (393216, 768, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [add_18, hidden_states_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf73, arg65_1, buf64, arg66_1, arg67_1, buf77, 8192, 768, grid=grid(8192), stream=stream0)
        del arg65_1
        del arg66_1
        del arg67_1
        buf78 = reinterpret_tensor(buf59, (8192, 3072), (3072, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (8192, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 3072), (1, 768), 0), out=buf78)
        del arg68_1
        buf79 = reinterpret_tensor(buf78, (16, 512, 3072), (1572864, 3072, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf79, arg69_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg69_1
        buf80 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg70_1, (3072, 768), (1, 3072), 0), out=buf80)
        del arg70_1
        buf84 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [add_19, hidden_states_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf80, arg71_1, buf77, arg72_1, arg73_1, buf84, 8192, 768, grid=grid(8192), stream=stream0)
        del arg71_1
        del arg72_1
        del arg73_1
        buf85 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg75_1, reinterpret_tensor(buf84, (8192, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf85)
        del arg74_1
        del arg75_1
        buf86 = reinterpret_tensor(buf77, (8192, 768), (768, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg77_1, reinterpret_tensor(buf84, (8192, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf86)
        del arg76_1
        del arg77_1
        buf87 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg79_1, reinterpret_tensor(buf84, (8192, 768), (768, 1), 0), reinterpret_tensor(arg78_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf87)
        del arg78_1
        del arg79_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf88 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf85, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf86, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf87, (16, 12, 512, 64), (393216, 64, 768, 1), 0), None, False, scale=0.125)
        del buf85
        buf89 = buf88[0]
        del buf88
        buf93 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf89, (8192, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 768), (1, 768), 0), out=buf93)
        del arg80_1
        buf97 = reinterpret_tensor(buf89, (16, 512, 768), (393216, 768, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [add_21, hidden_states_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf93, arg81_1, buf84, arg82_1, arg83_1, buf97, 8192, 768, grid=grid(8192), stream=stream0)
        del arg81_1
        del arg82_1
        del arg83_1
        buf98 = reinterpret_tensor(buf79, (8192, 3072), (3072, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (8192, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 3072), (1, 768), 0), out=buf98)
        del arg84_1
        buf99 = reinterpret_tensor(buf98, (16, 512, 3072), (1572864, 3072, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_36], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf99, arg85_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg85_1
        buf100 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg86_1, (3072, 768), (1, 3072), 0), out=buf100)
        del arg86_1
        buf104 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [add_22, hidden_states_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf100, arg87_1, buf97, arg88_1, arg89_1, buf104, 8192, 768, grid=grid(8192), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        buf105 = reinterpret_tensor(buf97, (8192, 768), (768, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg91_1, reinterpret_tensor(buf104, (8192, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf105)
        del arg90_1
        del arg91_1
        buf106 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg93_1, reinterpret_tensor(buf104, (8192, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf106)
        del arg92_1
        del arg93_1
        buf107 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg95_1, reinterpret_tensor(buf104, (8192, 768), (768, 1), 0), reinterpret_tensor(arg94_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf107)
        del arg94_1
        del arg95_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf108 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf105, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf106, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf107, (16, 12, 512, 64), (393216, 64, 768, 1), 0), None, False, scale=0.125)
        del buf105
        buf109 = buf108[0]
        del buf108
        buf113 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf109, (8192, 768), (768, 1), 0), reinterpret_tensor(arg96_1, (768, 768), (1, 768), 0), out=buf113)
        del arg96_1
        buf117 = reinterpret_tensor(buf109, (16, 512, 768), (393216, 768, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [add_24, hidden_states_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf113, arg97_1, buf104, arg98_1, arg99_1, buf117, 8192, 768, grid=grid(8192), stream=stream0)
        del arg97_1
        del arg98_1
        del arg99_1
        buf118 = reinterpret_tensor(buf99, (8192, 3072), (3072, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf117, (8192, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 3072), (1, 768), 0), out=buf118)
        del arg100_1
        buf119 = reinterpret_tensor(buf118, (16, 512, 3072), (1572864, 3072, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_44], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf119, arg101_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg101_1
        buf120 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg102_1, (3072, 768), (1, 3072), 0), out=buf120)
        del arg102_1
        buf124 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [add_25, hidden_states_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf120, arg103_1, buf117, arg104_1, arg105_1, buf124, 8192, 768, grid=grid(8192), stream=stream0)
        del arg103_1
        del arg104_1
        del arg105_1
        buf125 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg107_1, reinterpret_tensor(buf124, (8192, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf125)
        del arg106_1
        del arg107_1
        buf126 = reinterpret_tensor(buf117, (8192, 768), (768, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg109_1, reinterpret_tensor(buf124, (8192, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf126)
        del arg108_1
        del arg109_1
        buf127 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg111_1, reinterpret_tensor(buf124, (8192, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf127)
        del arg110_1
        del arg111_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf128 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf125, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf126, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf127, (16, 12, 512, 64), (393216, 64, 768, 1), 0), None, False, scale=0.125)
        del buf125
        buf129 = buf128[0]
        del buf128
        buf133 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (8192, 768), (768, 1), 0), reinterpret_tensor(arg112_1, (768, 768), (1, 768), 0), out=buf133)
        del arg112_1
        buf137 = reinterpret_tensor(buf129, (16, 512, 768), (393216, 768, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [add_27, hidden_states_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf133, arg113_1, buf124, arg114_1, arg115_1, buf137, 8192, 768, grid=grid(8192), stream=stream0)
        del arg113_1
        del arg114_1
        del arg115_1
        buf138 = reinterpret_tensor(buf119, (8192, 3072), (3072, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf137, (8192, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 3072), (1, 768), 0), out=buf138)
        del arg116_1
        buf139 = reinterpret_tensor(buf138, (16, 512, 3072), (1572864, 3072, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_52], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf139, arg117_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg117_1
        buf140 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf139, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg118_1, (3072, 768), (1, 3072), 0), out=buf140)
        del arg118_1
        buf144 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [add_28, hidden_states_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf140, arg119_1, buf137, arg120_1, arg121_1, buf144, 8192, 768, grid=grid(8192), stream=stream0)
        del arg119_1
        del arg120_1
        del arg121_1
        buf145 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg123_1, reinterpret_tensor(buf144, (8192, 768), (768, 1), 0), reinterpret_tensor(arg122_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf145)
        del arg122_1
        del arg123_1
        buf146 = reinterpret_tensor(buf137, (8192, 768), (768, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg125_1, reinterpret_tensor(buf144, (8192, 768), (768, 1), 0), reinterpret_tensor(arg124_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf146)
        del arg124_1
        del arg125_1
        buf147 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg127_1, reinterpret_tensor(buf144, (8192, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf147)
        del arg126_1
        del arg127_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf148 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf145, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf146, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf147, (16, 12, 512, 64), (393216, 64, 768, 1), 0), None, False, scale=0.125)
        del buf145
        buf149 = buf148[0]
        del buf148
        buf153 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (8192, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 768), (1, 768), 0), out=buf153)
        del arg128_1
        buf157 = reinterpret_tensor(buf149, (16, 512, 768), (393216, 768, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [add_30, hidden_states_58], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf153, arg129_1, buf144, arg130_1, arg131_1, buf157, 8192, 768, grid=grid(8192), stream=stream0)
        del arg129_1
        del arg130_1
        del arg131_1
        buf158 = reinterpret_tensor(buf139, (8192, 3072), (3072, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (8192, 768), (768, 1), 0), reinterpret_tensor(arg132_1, (768, 3072), (1, 768), 0), out=buf158)
        del arg132_1
        buf159 = reinterpret_tensor(buf158, (16, 512, 3072), (1572864, 3072, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_60], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf159, arg133_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg133_1
        buf160 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg134_1, (3072, 768), (1, 3072), 0), out=buf160)
        del arg134_1
        buf164 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [add_31, hidden_states_63], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf160, arg135_1, buf157, arg136_1, arg137_1, buf164, 8192, 768, grid=grid(8192), stream=stream0)
        del arg135_1
        del arg136_1
        del arg137_1
        buf165 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg139_1, reinterpret_tensor(buf164, (8192, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf165)
        del arg138_1
        del arg139_1
        buf166 = reinterpret_tensor(buf157, (8192, 768), (768, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg141_1, reinterpret_tensor(buf164, (8192, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf166)
        del arg140_1
        del arg141_1
        buf167 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg143_1, reinterpret_tensor(buf164, (8192, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf167)
        del arg142_1
        del arg143_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf168 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf165, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf166, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf167, (16, 12, 512, 64), (393216, 64, 768, 1), 0), None, False, scale=0.125)
        del buf165
        buf169 = buf168[0]
        del buf168
        buf173 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (8192, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 768), (1, 768), 0), out=buf173)
        del arg144_1
        buf177 = reinterpret_tensor(buf169, (16, 512, 768), (393216, 768, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [add_33, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf173, arg145_1, buf164, arg146_1, arg147_1, buf177, 8192, 768, grid=grid(8192), stream=stream0)
        del arg145_1
        del arg146_1
        del arg147_1
        buf178 = reinterpret_tensor(buf159, (8192, 3072), (3072, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf177, (8192, 768), (768, 1), 0), reinterpret_tensor(arg148_1, (768, 3072), (1, 768), 0), out=buf178)
        del arg148_1
        buf179 = reinterpret_tensor(buf178, (16, 512, 3072), (1572864, 3072, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_68], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf179, arg149_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg149_1
        buf180 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg150_1, (3072, 768), (1, 3072), 0), out=buf180)
        del arg150_1
        buf184 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [add_34, hidden_states_71], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf180, arg151_1, buf177, arg152_1, arg153_1, buf184, 8192, 768, grid=grid(8192), stream=stream0)
        del arg151_1
        del arg152_1
        del arg153_1
        buf185 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg155_1, reinterpret_tensor(buf184, (8192, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf185)
        del arg154_1
        del arg155_1
        buf186 = reinterpret_tensor(buf177, (8192, 768), (768, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg157_1, reinterpret_tensor(buf184, (8192, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf186)
        del arg156_1
        del arg157_1
        buf187 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg159_1, reinterpret_tensor(buf184, (8192, 768), (768, 1), 0), reinterpret_tensor(arg158_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf187)
        del arg158_1
        del arg159_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf188 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf185, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf186, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf187, (16, 12, 512, 64), (393216, 64, 768, 1), 0), None, False, scale=0.125)
        del buf185
        buf189 = buf188[0]
        del buf188
        buf193 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (8192, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 768), (1, 768), 0), out=buf193)
        del arg160_1
        buf197 = reinterpret_tensor(buf189, (16, 512, 768), (393216, 768, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [add_36, hidden_states_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf193, arg161_1, buf184, arg162_1, arg163_1, buf197, 8192, 768, grid=grid(8192), stream=stream0)
        del arg161_1
        del arg162_1
        del arg163_1
        buf198 = reinterpret_tensor(buf179, (8192, 3072), (3072, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (8192, 768), (768, 1), 0), reinterpret_tensor(arg164_1, (768, 3072), (1, 768), 0), out=buf198)
        del arg164_1
        buf199 = reinterpret_tensor(buf198, (16, 512, 3072), (1572864, 3072, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf199, arg165_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg165_1
        buf200 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg166_1, (3072, 768), (1, 3072), 0), out=buf200)
        del arg166_1
        buf204 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [add_37, hidden_states_79], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf200, arg167_1, buf197, arg168_1, arg169_1, buf204, 8192, 768, grid=grid(8192), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        buf205 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg171_1, reinterpret_tensor(buf204, (8192, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf205)
        del arg170_1
        del arg171_1
        buf206 = reinterpret_tensor(buf197, (8192, 768), (768, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg173_1, reinterpret_tensor(buf204, (8192, 768), (768, 1), 0), reinterpret_tensor(arg172_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf206)
        del arg172_1
        del arg173_1
        buf207 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg175_1, reinterpret_tensor(buf204, (8192, 768), (768, 1), 0), reinterpret_tensor(arg174_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf207)
        del arg174_1
        del arg175_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf208 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf205, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf206, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf207, (16, 12, 512, 64), (393216, 64, 768, 1), 0), None, False, scale=0.125)
        del buf205
        buf209 = buf208[0]
        del buf208
        buf213 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf209, (8192, 768), (768, 1), 0), reinterpret_tensor(arg176_1, (768, 768), (1, 768), 0), out=buf213)
        del arg176_1
        buf217 = reinterpret_tensor(buf209, (16, 512, 768), (393216, 768, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [add_39, hidden_states_82], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf213, arg177_1, buf204, arg178_1, arg179_1, buf217, 8192, 768, grid=grid(8192), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        buf218 = reinterpret_tensor(buf199, (8192, 3072), (3072, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 768), (768, 1), 0), reinterpret_tensor(arg180_1, (768, 3072), (1, 768), 0), out=buf218)
        del arg180_1
        buf219 = reinterpret_tensor(buf218, (16, 512, 3072), (1572864, 3072, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf219, arg181_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg181_1
        buf220 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg182_1, (3072, 768), (1, 3072), 0), out=buf220)
        del arg182_1
        buf224 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [add_40, hidden_states_87], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf220, arg183_1, buf217, arg184_1, arg185_1, buf224, 8192, 768, grid=grid(8192), stream=stream0)
        del arg183_1
        del arg184_1
        del arg185_1
        buf225 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg187_1, reinterpret_tensor(buf224, (8192, 768), (768, 1), 0), reinterpret_tensor(arg186_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf225)
        del arg186_1
        del arg187_1
        buf226 = reinterpret_tensor(buf217, (8192, 768), (768, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg189_1, reinterpret_tensor(buf224, (8192, 768), (768, 1), 0), reinterpret_tensor(arg188_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf226)
        del arg188_1
        del arg189_1
        buf227 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg191_1, reinterpret_tensor(buf224, (8192, 768), (768, 1), 0), reinterpret_tensor(arg190_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf227)
        del arg190_1
        del arg191_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf228 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf225, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf226, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf227, (16, 12, 512, 64), (393216, 64, 768, 1), 0), None, False, scale=0.125)
        del buf225
        del buf226
        buf229 = buf228[0]
        del buf228
        buf233 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (8192, 768), (768, 1), 0), reinterpret_tensor(arg192_1, (768, 768), (1, 768), 0), out=buf233)
        del arg192_1
        buf237 = reinterpret_tensor(buf229, (16, 512, 768), (393216, 768, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [add_42, hidden_states_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf233, arg193_1, buf224, arg194_1, arg195_1, buf237, 8192, 768, grid=grid(8192), stream=stream0)
        del arg193_1
        del arg194_1
        del arg195_1
        buf238 = reinterpret_tensor(buf219, (8192, 3072), (3072, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf237, (8192, 768), (768, 1), 0), reinterpret_tensor(arg196_1, (768, 3072), (1, 768), 0), out=buf238)
        del arg196_1
        buf239 = reinterpret_tensor(buf238, (16, 512, 3072), (1572864, 3072, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_92], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf239, arg197_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg197_1
        buf240 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg198_1, (3072, 768), (1, 3072), 0), out=buf240)
        del arg198_1
        del buf239
        buf244 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [add_43, hidden_states_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf240, arg199_1, buf237, arg200_1, arg201_1, buf244, 8192, 768, grid=grid(8192), stream=stream0)
        del arg199_1
        del arg200_1
        del arg201_1
        del buf237
        del buf240
        buf245 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (16, 768), (393216, 1), 0), reinterpret_tensor(arg202_1, (768, 768), (1, 768), 0), out=buf245)
        del arg202_1
        del buf244
        buf246 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [pooled_output_1], Original ATen: [aten.tanh]
        triton_poi_fused_tanh_3.run(buf246, arg203_1, 12288, grid=grid(12288), stream=stream0)
        del arg203_1
        buf247 = empty_strided_cuda((768, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(arg205_1, buf247, 3072, grid=grid(3072), stream=stream0)
        del arg205_1
        buf248 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(arg206_1, buf248, 4, grid=grid(4), stream=stream0)
        del arg206_1
        buf249 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pooled_output_1], Original ATen: [aten.tanh]
        extern_kernels.addmm(buf248, buf246, buf247, alpha=1, beta=1, out=buf249)
        del buf246
        del buf247
        del buf248
        buf250 = empty_strided_cuda((), (), torch.float32)
        buf252 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_6.run(buf252, arg207_1, buf249, 1, 16, grid=grid(1), stream=stream0)
        del arg207_1
    return (buf252, reinterpret_tensor(buf249, (16, 2), (4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg205_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('LayoutLMForSequenceClassification', benchmark_compiled_module)
