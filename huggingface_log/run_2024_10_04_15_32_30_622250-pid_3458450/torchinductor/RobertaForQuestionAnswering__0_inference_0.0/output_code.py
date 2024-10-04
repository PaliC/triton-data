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


# kernel path: /tmp/torchinductor_sahanp/75/c75r7xkscdaihxwjhjmbs6suefpo2nmxdi4ebm7rwfzqkhhxpqxf.py
# Topologically Sorted Source Nodes: [ne, mask, cumsum], Original ATen: [aten.ne, aten._to_copy, aten.cumsum]
# Source node to ATen node mapping:
#   cumsum => cumsum
#   mask => convert_element_type
#   ne => ne
# Graph fragment:
#   %ne : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%arg0_1, 0), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne, torch.int32), kwargs = {})
#   %cumsum : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%convert_element_type, 1), kwargs = {})
triton_per_fused__to_copy_cumsum_ne_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints=[16, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_cumsum_ne_0', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 16
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
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int32)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tmp4.to(tl.int64)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp7, = tl.associative_scan((tmp6,), 0, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/c6/cc6nzzdryjcmkx2cn23ao3xko6uo2mgkqd77zxcntl5v5u6w2nfo.py
# Topologically Sorted Source Nodes: [inputs_embeds, token_type_embeddings, embeddings, ne, mask, type_as, add, incremental_indices, long, position_ids, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.ne, aten._to_copy, aten.mul, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add
#   embeddings => add_2
#   embeddings_1 => add_3
#   embeddings_2 => add_4, add_5, mul_1, mul_2, rsqrt, sub, var_mean
#   incremental_indices => mul
#   inputs_embeds => embedding
#   long => convert_element_type_2
#   mask => convert_element_type
#   ne => ne
#   position_embeddings => embedding_2
#   position_ids => add_1
#   token_type_embeddings => embedding_1
#   type_as => convert_element_type_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %arg0_1, 0), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %expand), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %ne : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%arg0_1, 0), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne, torch.int32), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cumsum, torch.int32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %convert_element_type), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.int64), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 0), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %add_1, 0), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %embedding_2), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   %add_5 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg6_1), kwargs = {})
triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_1', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel):
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
    tmp15 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([RBLOCK], 50265, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 50265), "index out of bounds: 0 <= tmp4 < 50265")
    tmp6 = tl.load(in_ptr1 + (r2 + (768*tmp4)), rmask, other=0.0)
    tmp8 = tl.full([RBLOCK], 2, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 2), "index out of bounds: 0 <= tmp11 < 2")
    tmp13 = tl.load(in_ptr3 + (r2 + (768*tmp11)), rmask, other=0.0)
    tmp14 = tmp6 + tmp13
    tmp16 = tmp15.to(tl.int32)
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([1], 0, tl.int64)
    tmp20 = tmp0 != tmp19
    tmp21 = tmp20.to(tl.int32)
    tmp22 = tmp18 * tmp21
    tmp23 = tmp22.to(tl.int64)
    tmp24 = tmp23 + tmp19
    tmp25 = tl.full([RBLOCK], 512, tl.int32)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp24 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp24)
    tl.device_assert((0 <= tmp28) & (tmp28 < 512), "index out of bounds: 0 <= tmp28 < 512")
    tmp30 = tl.load(in_ptr5 + (r2 + (768*tmp28)), rmask, other=0.0)
    tmp31 = tmp14 + tmp30
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = tl.where(rmask, tmp32, 0)
    tmp35 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp37 = tl.where(rmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = tl.full([1], 768, tl.int32)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp38 / tmp40
    tmp42 = tmp32 - tmp41
    tmp43 = tmp42 * tmp42
    tmp44 = tl.broadcast_to(tmp43, [RBLOCK])
    tmp46 = tl.where(rmask, tmp44, 0)
    tmp47 = triton_helpers.promote_to_tensor(tl.sum(tmp46, 0))
    tmp48 = tmp31 - tmp41
    tmp49 = 768.0
    tmp50 = tmp47 / tmp49
    tmp51 = 1e-12
    tmp52 = tmp50 + tmp51
    tmp53 = libdevice.rsqrt(tmp52)
    tmp54 = tmp48 * tmp53
    tmp56 = tmp54 * tmp55
    tmp58 = tmp56 + tmp57
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp31, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp58, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ur/curwbngmbfwb3z4g6p5mqujdthoqfl3g5ihslufclzjotimzp26i.py
# Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_1, %permute_3, %permute_5, %expand_2, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], False, tl.int1)
    tmp1 = -3.4028234663852886e+38
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zn/czngnp2vd2744vc4tdxd33mabtdgqrw4qygeh3us5nwqdwputkxf.py
# Topologically Sorted Source Nodes: [add_3, hidden_states_2], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_3 => add_6
#   hidden_states_2 => add_7, add_8, mul_3, mul_4, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %add_5), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_6, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %getitem_7), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-12), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg15_1), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg16_1), kwargs = {})
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/2l/c2l3qn3qcujzcaywy73psey2hi7m5llsrvlj5smdgz4hbcb4h4gi.py
# Topologically Sorted Source Nodes: [hidden_states_4], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_4 => add_9, erf, mul_5, mul_6, mul_7
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_9), kwargs = {})
triton_poi_fused_gelu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/on/cona5nwhcayiyh65tfrhpxjaoeihtmzusipagihkduo337a5cigs.py
# Topologically Sorted Source Nodes: [start_logits_1, start_loss], Original ATen: [aten.clone, aten._log_softmax]
# Source node to ATen node mapping:
#   start_logits_1 => clone_25
#   start_loss => amax, exp, sub_26, sum_1
# Graph fragment:
#   %clone_25 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_25, [1], True), kwargs = {})
#   %sub_26 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_25, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_26,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_red_fused__log_softmax_clone_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[16, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_clone_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*r1) + (1024*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp0 + tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tl.store(out_ptr0 + (r1 + (512*x0)), tmp3, rmask & xmask)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp5, xmask)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(out_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp7 - tmp5
        tmp9 = tl_math.exp(tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/63/c63rg7fhpjm7y4zr2spjhkv5uxtcgayt7xqikqmwcgtrhnwrvqgg.py
# Topologically Sorted Source Nodes: [end_logits_1, end_loss], Original ATen: [aten.clone, aten._log_softmax]
# Source node to ATen node mapping:
#   end_logits_1 => clone_26
#   end_loss => amax_1, exp_1, sub_28, sum_4
# Graph fragment:
#   %clone_26 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_1,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_26, [1], True), kwargs = {})
#   %sub_28 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_26, %amax_1), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_28,), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})
triton_red_fused__log_softmax_clone_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[16, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_clone_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (1))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (1 + (2*r1) + (1024*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp0 + tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tl.store(out_ptr0 + (r1 + (512*x0)), tmp3, rmask & xmask)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp5, xmask)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(out_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp7 - tmp5
        tmp9 = tl_math.exp(tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ch/cch2lk5akun473cazhh46ynaknfijpthturzlkksw7d6ibr5usmg.py
# Topologically Sorted Source Nodes: [start_positions, start_loss, end_positions, end_loss, add_27, total_loss], Original ATen: [aten.clamp, aten.nll_loss_forward, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_27 => add_90
#   end_loss => convert_element_type_5, div_1, full_default_3, ne_5, ne_6, neg_1, sum_5, sum_6, where_4
#   end_positions => clamp_max_1, clamp_min_1
#   start_loss => convert_element_type_4, div, full_default_1, ne_2, ne_3, neg, sum_2, sum_3, where_2
#   start_positions => clamp_max, clamp_min
#   total_loss => div_2
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg201_1, 0), kwargs = {})
#   %clamp_max : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 512), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max, 512), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_2,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_2, %neg, %full_default_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_2,), kwargs = {})
#   %ne_3 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max, 512), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_3,), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type_4), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg202_1, 0), kwargs = {})
#   %clamp_max_1 : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 512), kwargs = {})
#   %ne_5 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max_1, 512), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_3,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_5, %neg_1, %full_default_3), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_4,), kwargs = {})
#   %ne_6 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max_1, 512), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_6,), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_5, torch.float32), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_6, %convert_element_type_5), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %div_1), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_90, 2), kwargs = {})
triton_per_fused_add_clamp_div_nll_loss_forward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {9: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_div_nll_loss_forward_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp13 = tl.load(in_ptr2 + (r0), None)
    tmp15 = tl.load(in_ptr3 + (r0), None)
    tmp28 = tl.load(in_ptr4 + (r0), None)
    tmp38 = tl.load(in_ptr6 + (r0), None)
    tmp40 = tl.load(in_ptr7 + (r0), None)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tmp4 != tmp3
    tmp6 = tl.where(tmp5, tmp4, tmp1)
    tmp7 = tl.full([XBLOCK, RBLOCK], 512, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 512), "index out of bounds: 0 <= tmp10 < 512")
    tmp12 = tl.load(in_ptr1 + (tmp10 + (512*r0)), None, eviction_policy='evict_last')
    tmp14 = tmp12 - tmp13
    tmp16 = tl_math.log(tmp15)
    tmp17 = tmp14 - tmp16
    tmp18 = -tmp17
    tmp19 = 0.0
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tmp5.to(tl.int64)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None]
    tmp29 = triton_helpers.maximum(tmp28, tmp1)
    tmp30 = triton_helpers.minimum(tmp29, tmp3)
    tmp31 = tmp30 != tmp3
    tmp32 = tl.where(tmp31, tmp30, tmp1)
    tmp33 = tmp32 + tmp7
    tmp34 = tmp32 < 0
    tmp35 = tl.where(tmp34, tmp33, tmp32)
    tl.device_assert((0 <= tmp35) & (tmp35 < 512), "index out of bounds: 0 <= tmp35 < 512")
    tmp37 = tl.load(in_ptr5 + (tmp35 + (512*r0)), None, eviction_policy='evict_last')
    tmp39 = tmp37 - tmp38
    tmp41 = tl_math.log(tmp40)
    tmp42 = tmp39 - tmp41
    tmp43 = -tmp42
    tmp44 = tl.where(tmp31, tmp43, tmp19)
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
    tmp47 = tl.sum(tmp45, 1)[:, None]
    tmp48 = tmp31.to(tl.int64)
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, RBLOCK])
    tmp51 = tl.sum(tmp49, 1)[:, None]
    tmp52 = tmp27.to(tl.float32)
    tmp53 = tmp23 / tmp52
    tmp54 = tmp51.to(tl.float32)
    tmp55 = tmp47 / tmp54
    tmp56 = tmp53 + tmp55
    tmp57 = 0.5
    tmp58 = tmp56 * tmp57
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp58, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 512), (512, 1))
    assert_size_stride(arg1_1, (1, 512), (512, 1))
    assert_size_stride(arg2_1, (50265, 768), (768, 1))
    assert_size_stride(arg3_1, (2, 768), (768, 1))
    assert_size_stride(arg4_1, (512, 768), (768, 1))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, 768), (768, 1))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, 768), (768, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, 768), (768, 1))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (3072, 768), (768, 1))
    assert_size_stride(arg18_1, (3072, ), (1, ))
    assert_size_stride(arg19_1, (768, 3072), (3072, 1))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, 768), (768, 1))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, 768), (768, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, 768), (768, 1))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (3072, 768), (768, 1))
    assert_size_stride(arg34_1, (3072, ), (1, ))
    assert_size_stride(arg35_1, (768, 3072), (3072, 1))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, 768), (768, 1))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, 768), (768, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, 768), (768, 1))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (3072, 768), (768, 1))
    assert_size_stride(arg50_1, (3072, ), (1, ))
    assert_size_stride(arg51_1, (768, 3072), (3072, 1))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, 768), (768, 1))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, 768), (768, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, 768), (768, 1))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, 768), (768, 1))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (3072, 768), (768, 1))
    assert_size_stride(arg66_1, (3072, ), (1, ))
    assert_size_stride(arg67_1, (768, 3072), (3072, 1))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, 768), (768, 1))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, 768), (768, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, 768), (768, 1))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, 768), (768, 1))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (3072, 768), (768, 1))
    assert_size_stride(arg82_1, (3072, ), (1, ))
    assert_size_stride(arg83_1, (768, 3072), (3072, 1))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, 768), (768, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, 768), (768, 1))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, 768), (768, 1))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (3072, 768), (768, 1))
    assert_size_stride(arg98_1, (3072, ), (1, ))
    assert_size_stride(arg99_1, (768, 3072), (3072, 1))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, 768), (768, 1))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, 768), (768, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, 768), (768, 1))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, 768), (768, 1))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (3072, 768), (768, 1))
    assert_size_stride(arg114_1, (3072, ), (1, ))
    assert_size_stride(arg115_1, (768, 3072), (3072, 1))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, 768), (768, 1))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, 768), (768, 1))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, 768), (768, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, 768), (768, 1))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (3072, 768), (768, 1))
    assert_size_stride(arg130_1, (3072, ), (1, ))
    assert_size_stride(arg131_1, (768, 3072), (3072, 1))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, 768), (768, 1))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, 768), (768, 1))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, 768), (768, 1))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, 768), (768, 1))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (3072, 768), (768, 1))
    assert_size_stride(arg146_1, (3072, ), (1, ))
    assert_size_stride(arg147_1, (768, 3072), (3072, 1))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (768, 768), (768, 1))
    assert_size_stride(arg152_1, (768, ), (1, ))
    assert_size_stride(arg153_1, (768, 768), (768, 1))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, 768), (768, 1))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, 768), (768, 1))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (3072, 768), (768, 1))
    assert_size_stride(arg162_1, (3072, ), (1, ))
    assert_size_stride(arg163_1, (768, 3072), (3072, 1))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, 768), (768, 1))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, 768), (768, 1))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (768, 768), (768, 1))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, 768), (768, 1))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (768, ), (1, ))
    assert_size_stride(arg177_1, (3072, 768), (768, 1))
    assert_size_stride(arg178_1, (3072, ), (1, ))
    assert_size_stride(arg179_1, (768, 3072), (3072, 1))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, 768), (768, 1))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, 768), (768, 1))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, 768), (768, 1))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (768, 768), (768, 1))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (768, ), (1, ))
    assert_size_stride(arg193_1, (3072, 768), (768, 1))
    assert_size_stride(arg194_1, (3072, ), (1, ))
    assert_size_stride(arg195_1, (768, 3072), (3072, 1))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (768, ), (1, ))
    assert_size_stride(arg199_1, (2, 768), (768, 1))
    assert_size_stride(arg200_1, (2, ), (1, ))
    assert_size_stride(arg201_1, (16, ), (1, ))
    assert_size_stride(arg202_1, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 512), (512, 1), torch.int64)
        # Topologically Sorted Source Nodes: [ne, mask, cumsum], Original ATen: [aten.ne, aten._to_copy, aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_cumsum_ne_0.run(arg0_1, buf0, 16, 512, grid=grid(16), stream=stream0)
        buf1 = empty_strided_cuda((16, 512, 768), (393216, 768, 1), torch.float32)
        buf5 = empty_strided_cuda((16, 512, 768), (393216, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, token_type_embeddings, embeddings, ne, mask, type_as, add, incremental_indices, long, position_ids, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.ne, aten._to_copy, aten.mul, aten.native_layer_norm]
        triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_1.run(arg0_1, arg2_1, arg1_1, arg3_1, buf0, arg4_1, arg5_1, arg6_1, buf1, buf5, 8192, 768, grid=grid(8192), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del buf0
        buf6 = reinterpret_tensor(buf1, (8192, 768), (768, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf5, (8192, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
        del arg7_1
        del arg8_1
        buf7 = empty_strided_cuda((8192, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf5, (8192, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
        del arg10_1
        del arg9_1
        buf8 = empty_strided_cuda((8192, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf5, (8192, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf8)
        del arg11_1
        del arg12_1
        buf9 = empty_strided_cuda((16, 12, 512, 512), (3145728, 262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf9, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf6, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf7, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf8, (16, 12, 512, 64), (393216, 64, 768, 1), 0), buf9, False)
        del buf6
        buf11 = buf10[0]
        del buf10
        buf15 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (8192, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf15)
        del arg13_1
        buf19 = reinterpret_tensor(buf11, (16, 512, 768), (393216, 768, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [add_3, hidden_states_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf15, arg14_1, buf5, arg15_1, arg16_1, buf19, 8192, 768, grid=grid(8192), stream=stream0)
        del arg14_1
        del arg15_1
        del arg16_1
        buf20 = empty_strided_cuda((8192, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (8192, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf20)
        del arg17_1
        buf21 = reinterpret_tensor(buf20, (16, 512, 3072), (1572864, 3072, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf21, arg18_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg18_1
        buf22 = reinterpret_tensor(buf5, (8192, 768), (768, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf22)
        del arg19_1
        buf26 = reinterpret_tensor(buf15, (16, 512, 768), (393216, 768, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [add_4, hidden_states_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf22, arg20_1, buf19, arg21_1, arg22_1, buf26, 8192, 768, grid=grid(8192), stream=stream0)
        del arg20_1
        del arg21_1
        del arg22_1
        buf27 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg24_1, reinterpret_tensor(buf26, (8192, 768), (768, 1), 0), reinterpret_tensor(arg23_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf27)
        del arg23_1
        del arg24_1
        buf28 = reinterpret_tensor(buf19, (8192, 768), (768, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg26_1, reinterpret_tensor(buf26, (8192, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf28)
        del arg25_1
        del arg26_1
        buf29 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg28_1, reinterpret_tensor(buf26, (8192, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf29)
        del arg27_1
        del arg28_1
        buf30 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf30, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf31 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf27, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf28, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf29, (16, 12, 512, 64), (393216, 64, 768, 1), 0), buf30, False)
        del buf27
        buf32 = buf31[0]
        del buf31
        buf36 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (8192, 768), (768, 1), 0), reinterpret_tensor(arg29_1, (768, 768), (1, 768), 0), out=buf36)
        del arg29_1
        buf40 = reinterpret_tensor(buf32, (16, 512, 768), (393216, 768, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [add_5, hidden_states_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf36, arg30_1, buf26, arg31_1, arg32_1, buf40, 8192, 768, grid=grid(8192), stream=stream0)
        del arg30_1
        del arg31_1
        del arg32_1
        buf41 = reinterpret_tensor(buf21, (8192, 3072), (3072, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (8192, 768), (768, 1), 0), reinterpret_tensor(arg33_1, (768, 3072), (1, 768), 0), out=buf41)
        del arg33_1
        buf42 = reinterpret_tensor(buf41, (16, 512, 3072), (1572864, 3072, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_12], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf42, arg34_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg34_1
        buf43 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg35_1, (3072, 768), (1, 3072), 0), out=buf43)
        del arg35_1
        buf47 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [add_6, hidden_states_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf43, arg36_1, buf40, arg37_1, arg38_1, buf47, 8192, 768, grid=grid(8192), stream=stream0)
        del arg36_1
        del arg37_1
        del arg38_1
        buf48 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg40_1, reinterpret_tensor(buf47, (8192, 768), (768, 1), 0), reinterpret_tensor(arg39_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf48)
        del arg39_1
        del arg40_1
        buf49 = reinterpret_tensor(buf40, (8192, 768), (768, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg42_1, reinterpret_tensor(buf47, (8192, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf49)
        del arg41_1
        del arg42_1
        buf50 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg44_1, reinterpret_tensor(buf47, (8192, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf50)
        del arg43_1
        del arg44_1
        buf51 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [attn_output_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf51, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf52 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf48, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf49, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf50, (16, 12, 512, 64), (393216, 64, 768, 1), 0), buf51, False)
        del buf48
        buf53 = buf52[0]
        del buf52
        buf57 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (8192, 768), (768, 1), 0), reinterpret_tensor(arg45_1, (768, 768), (1, 768), 0), out=buf57)
        del arg45_1
        buf61 = reinterpret_tensor(buf53, (16, 512, 768), (393216, 768, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [add_7, hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf57, arg46_1, buf47, arg47_1, arg48_1, buf61, 8192, 768, grid=grid(8192), stream=stream0)
        del arg46_1
        del arg47_1
        del arg48_1
        buf62 = reinterpret_tensor(buf42, (8192, 3072), (3072, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (8192, 768), (768, 1), 0), reinterpret_tensor(arg49_1, (768, 3072), (1, 768), 0), out=buf62)
        del arg49_1
        buf63 = reinterpret_tensor(buf62, (16, 512, 3072), (1572864, 3072, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf63, arg50_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg50_1
        buf64 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg51_1, (3072, 768), (1, 3072), 0), out=buf64)
        del arg51_1
        buf68 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [add_8, hidden_states_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf64, arg52_1, buf61, arg53_1, arg54_1, buf68, 8192, 768, grid=grid(8192), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        buf69 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg56_1, reinterpret_tensor(buf68, (8192, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf69)
        del arg55_1
        del arg56_1
        buf70 = reinterpret_tensor(buf61, (8192, 768), (768, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg58_1, reinterpret_tensor(buf68, (8192, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf70)
        del arg57_1
        del arg58_1
        buf71 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg60_1, reinterpret_tensor(buf68, (8192, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf71)
        del arg59_1
        del arg60_1
        buf72 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [attn_output_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf72, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf73 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf69, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf70, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf71, (16, 12, 512, 64), (393216, 64, 768, 1), 0), buf72, False)
        del buf69
        buf74 = buf73[0]
        del buf73
        buf78 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (8192, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 768), (1, 768), 0), out=buf78)
        del arg61_1
        buf82 = reinterpret_tensor(buf74, (16, 512, 768), (393216, 768, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [add_9, hidden_states_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf78, arg62_1, buf68, arg63_1, arg64_1, buf82, 8192, 768, grid=grid(8192), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        buf83 = reinterpret_tensor(buf63, (8192, 3072), (3072, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (8192, 768), (768, 1), 0), reinterpret_tensor(arg65_1, (768, 3072), (1, 768), 0), out=buf83)
        del arg65_1
        buf84 = reinterpret_tensor(buf83, (16, 512, 3072), (1572864, 3072, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf84, arg66_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg66_1
        buf85 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg67_1, (3072, 768), (1, 3072), 0), out=buf85)
        del arg67_1
        buf89 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [add_10, hidden_states_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf85, arg68_1, buf82, arg69_1, arg70_1, buf89, 8192, 768, grid=grid(8192), stream=stream0)
        del arg68_1
        del arg69_1
        del arg70_1
        buf90 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg72_1, reinterpret_tensor(buf89, (8192, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf90)
        del arg71_1
        del arg72_1
        buf91 = reinterpret_tensor(buf82, (8192, 768), (768, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg74_1, reinterpret_tensor(buf89, (8192, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf91)
        del arg73_1
        del arg74_1
        buf92 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg76_1, reinterpret_tensor(buf89, (8192, 768), (768, 1), 0), reinterpret_tensor(arg75_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf92)
        del arg75_1
        del arg76_1
        buf93 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [attn_output_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf93, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf94 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf90, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf91, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf92, (16, 12, 512, 64), (393216, 64, 768, 1), 0), buf93, False)
        del buf90
        buf95 = buf94[0]
        del buf94
        buf99 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (8192, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 768), (1, 768), 0), out=buf99)
        del arg77_1
        buf103 = reinterpret_tensor(buf95, (16, 512, 768), (393216, 768, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [add_11, hidden_states_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf99, arg78_1, buf89, arg79_1, arg80_1, buf103, 8192, 768, grid=grid(8192), stream=stream0)
        del arg78_1
        del arg79_1
        del arg80_1
        buf104 = reinterpret_tensor(buf84, (8192, 3072), (3072, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (8192, 768), (768, 1), 0), reinterpret_tensor(arg81_1, (768, 3072), (1, 768), 0), out=buf104)
        del arg81_1
        buf105 = reinterpret_tensor(buf104, (16, 512, 3072), (1572864, 3072, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_36], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf105, arg82_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg82_1
        buf106 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg83_1, (3072, 768), (1, 3072), 0), out=buf106)
        del arg83_1
        buf110 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [add_12, hidden_states_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf106, arg84_1, buf103, arg85_1, arg86_1, buf110, 8192, 768, grid=grid(8192), stream=stream0)
        del arg84_1
        del arg85_1
        del arg86_1
        buf111 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg88_1, reinterpret_tensor(buf110, (8192, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf111)
        del arg87_1
        del arg88_1
        buf112 = reinterpret_tensor(buf103, (8192, 768), (768, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg90_1, reinterpret_tensor(buf110, (8192, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf112)
        del arg89_1
        del arg90_1
        buf113 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg92_1, reinterpret_tensor(buf110, (8192, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf113)
        del arg91_1
        del arg92_1
        buf114 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [attn_output_15], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf114, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_15], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf115 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf111, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf112, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf113, (16, 12, 512, 64), (393216, 64, 768, 1), 0), buf114, False)
        del buf111
        buf116 = buf115[0]
        del buf115
        buf120 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (8192, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 768), (1, 768), 0), out=buf120)
        del arg93_1
        buf124 = reinterpret_tensor(buf116, (16, 512, 768), (393216, 768, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [add_13, hidden_states_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf120, arg94_1, buf110, arg95_1, arg96_1, buf124, 8192, 768, grid=grid(8192), stream=stream0)
        del arg94_1
        del arg95_1
        del arg96_1
        buf125 = reinterpret_tensor(buf105, (8192, 3072), (3072, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (8192, 768), (768, 1), 0), reinterpret_tensor(arg97_1, (768, 3072), (1, 768), 0), out=buf125)
        del arg97_1
        buf126 = reinterpret_tensor(buf125, (16, 512, 3072), (1572864, 3072, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_44], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf126, arg98_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg98_1
        buf127 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg99_1, (3072, 768), (1, 3072), 0), out=buf127)
        del arg99_1
        buf131 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [add_14, hidden_states_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf127, arg100_1, buf124, arg101_1, arg102_1, buf131, 8192, 768, grid=grid(8192), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        buf132 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg104_1, reinterpret_tensor(buf131, (8192, 768), (768, 1), 0), reinterpret_tensor(arg103_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf132)
        del arg103_1
        del arg104_1
        buf133 = reinterpret_tensor(buf124, (8192, 768), (768, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg106_1, reinterpret_tensor(buf131, (8192, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf133)
        del arg105_1
        del arg106_1
        buf134 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg108_1, reinterpret_tensor(buf131, (8192, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf134)
        del arg107_1
        del arg108_1
        buf135 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [attn_output_18], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf135, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_18], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf136 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf132, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf133, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf134, (16, 12, 512, 64), (393216, 64, 768, 1), 0), buf135, False)
        del buf132
        buf137 = buf136[0]
        del buf136
        buf141 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf137, (8192, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 768), (1, 768), 0), out=buf141)
        del arg109_1
        buf145 = reinterpret_tensor(buf137, (16, 512, 768), (393216, 768, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [add_15, hidden_states_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf141, arg110_1, buf131, arg111_1, arg112_1, buf145, 8192, 768, grid=grid(8192), stream=stream0)
        del arg110_1
        del arg111_1
        del arg112_1
        buf146 = reinterpret_tensor(buf126, (8192, 3072), (3072, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (8192, 768), (768, 1), 0), reinterpret_tensor(arg113_1, (768, 3072), (1, 768), 0), out=buf146)
        del arg113_1
        buf147 = reinterpret_tensor(buf146, (16, 512, 3072), (1572864, 3072, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_52], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf147, arg114_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg114_1
        buf148 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg115_1, (3072, 768), (1, 3072), 0), out=buf148)
        del arg115_1
        buf152 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [add_16, hidden_states_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf148, arg116_1, buf145, arg117_1, arg118_1, buf152, 8192, 768, grid=grid(8192), stream=stream0)
        del arg116_1
        del arg117_1
        del arg118_1
        buf153 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg120_1, reinterpret_tensor(buf152, (8192, 768), (768, 1), 0), reinterpret_tensor(arg119_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf153)
        del arg119_1
        del arg120_1
        buf154 = reinterpret_tensor(buf145, (8192, 768), (768, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg122_1, reinterpret_tensor(buf152, (8192, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf154)
        del arg121_1
        del arg122_1
        buf155 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg124_1, reinterpret_tensor(buf152, (8192, 768), (768, 1), 0), reinterpret_tensor(arg123_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf155)
        del arg123_1
        del arg124_1
        buf156 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [attn_output_21], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf156, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_21], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf157 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf153, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf154, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf155, (16, 12, 512, 64), (393216, 64, 768, 1), 0), buf156, False)
        del buf153
        buf158 = buf157[0]
        del buf157
        buf162 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (8192, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 768), (1, 768), 0), out=buf162)
        del arg125_1
        buf166 = reinterpret_tensor(buf158, (16, 512, 768), (393216, 768, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [add_17, hidden_states_58], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf162, arg126_1, buf152, arg127_1, arg128_1, buf166, 8192, 768, grid=grid(8192), stream=stream0)
        del arg126_1
        del arg127_1
        del arg128_1
        buf167 = reinterpret_tensor(buf147, (8192, 3072), (3072, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (8192, 768), (768, 1), 0), reinterpret_tensor(arg129_1, (768, 3072), (1, 768), 0), out=buf167)
        del arg129_1
        buf168 = reinterpret_tensor(buf167, (16, 512, 3072), (1572864, 3072, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_60], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf168, arg130_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg130_1
        buf169 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg131_1, (3072, 768), (1, 3072), 0), out=buf169)
        del arg131_1
        buf173 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [add_18, hidden_states_63], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf169, arg132_1, buf166, arg133_1, arg134_1, buf173, 8192, 768, grid=grid(8192), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        buf174 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg136_1, reinterpret_tensor(buf173, (8192, 768), (768, 1), 0), reinterpret_tensor(arg135_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf174)
        del arg135_1
        del arg136_1
        buf175 = reinterpret_tensor(buf166, (8192, 768), (768, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg138_1, reinterpret_tensor(buf173, (8192, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf175)
        del arg137_1
        del arg138_1
        buf176 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg140_1, reinterpret_tensor(buf173, (8192, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf176)
        del arg139_1
        del arg140_1
        buf177 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [attn_output_24], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf177, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_24], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf178 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf174, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf175, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf176, (16, 12, 512, 64), (393216, 64, 768, 1), 0), buf177, False)
        del buf174
        buf179 = buf178[0]
        del buf178
        buf183 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (8192, 768), (768, 1), 0), reinterpret_tensor(arg141_1, (768, 768), (1, 768), 0), out=buf183)
        del arg141_1
        buf187 = reinterpret_tensor(buf179, (16, 512, 768), (393216, 768, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [add_19, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf183, arg142_1, buf173, arg143_1, arg144_1, buf187, 8192, 768, grid=grid(8192), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        buf188 = reinterpret_tensor(buf168, (8192, 3072), (3072, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (8192, 768), (768, 1), 0), reinterpret_tensor(arg145_1, (768, 3072), (1, 768), 0), out=buf188)
        del arg145_1
        buf189 = reinterpret_tensor(buf188, (16, 512, 3072), (1572864, 3072, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_68], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf189, arg146_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg146_1
        buf190 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg147_1, (3072, 768), (1, 3072), 0), out=buf190)
        del arg147_1
        buf194 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [add_20, hidden_states_71], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf190, arg148_1, buf187, arg149_1, arg150_1, buf194, 8192, 768, grid=grid(8192), stream=stream0)
        del arg148_1
        del arg149_1
        del arg150_1
        buf195 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg152_1, reinterpret_tensor(buf194, (8192, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf195)
        del arg151_1
        del arg152_1
        buf196 = reinterpret_tensor(buf187, (8192, 768), (768, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg154_1, reinterpret_tensor(buf194, (8192, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf196)
        del arg153_1
        del arg154_1
        buf197 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg156_1, reinterpret_tensor(buf194, (8192, 768), (768, 1), 0), reinterpret_tensor(arg155_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf197)
        del arg155_1
        del arg156_1
        buf198 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [attn_output_27], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf198, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_27], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf199 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf195, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf196, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf197, (16, 12, 512, 64), (393216, 64, 768, 1), 0), buf198, False)
        del buf195
        buf200 = buf199[0]
        del buf199
        buf204 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (8192, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 768), (1, 768), 0), out=buf204)
        del arg157_1
        buf208 = reinterpret_tensor(buf200, (16, 512, 768), (393216, 768, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [add_21, hidden_states_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf204, arg158_1, buf194, arg159_1, arg160_1, buf208, 8192, 768, grid=grid(8192), stream=stream0)
        del arg158_1
        del arg159_1
        del arg160_1
        buf209 = reinterpret_tensor(buf189, (8192, 3072), (3072, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf208, (8192, 768), (768, 1), 0), reinterpret_tensor(arg161_1, (768, 3072), (1, 768), 0), out=buf209)
        del arg161_1
        buf210 = reinterpret_tensor(buf209, (16, 512, 3072), (1572864, 3072, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf210, arg162_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg162_1
        buf211 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg163_1, (3072, 768), (1, 3072), 0), out=buf211)
        del arg163_1
        buf215 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [add_22, hidden_states_79], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf211, arg164_1, buf208, arg165_1, arg166_1, buf215, 8192, 768, grid=grid(8192), stream=stream0)
        del arg164_1
        del arg165_1
        del arg166_1
        buf216 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg168_1, reinterpret_tensor(buf215, (8192, 768), (768, 1), 0), reinterpret_tensor(arg167_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf216)
        del arg167_1
        del arg168_1
        buf217 = reinterpret_tensor(buf208, (8192, 768), (768, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg170_1, reinterpret_tensor(buf215, (8192, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf217)
        del arg169_1
        del arg170_1
        buf218 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg172_1, reinterpret_tensor(buf215, (8192, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf218)
        del arg171_1
        del arg172_1
        buf219 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [attn_output_30], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf219, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_30], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf220 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf216, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf217, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf218, (16, 12, 512, 64), (393216, 64, 768, 1), 0), buf219, False)
        del buf216
        buf221 = buf220[0]
        del buf220
        buf225 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (8192, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 768), (1, 768), 0), out=buf225)
        del arg173_1
        buf229 = reinterpret_tensor(buf221, (16, 512, 768), (393216, 768, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [add_23, hidden_states_82], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf225, arg174_1, buf215, arg175_1, arg176_1, buf229, 8192, 768, grid=grid(8192), stream=stream0)
        del arg174_1
        del arg175_1
        del arg176_1
        buf230 = reinterpret_tensor(buf210, (8192, 3072), (3072, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (8192, 768), (768, 1), 0), reinterpret_tensor(arg177_1, (768, 3072), (1, 768), 0), out=buf230)
        del arg177_1
        buf231 = reinterpret_tensor(buf230, (16, 512, 3072), (1572864, 3072, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf231, arg178_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg178_1
        buf232 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg179_1, (3072, 768), (1, 3072), 0), out=buf232)
        del arg179_1
        buf236 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [add_24, hidden_states_87], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf232, arg180_1, buf229, arg181_1, arg182_1, buf236, 8192, 768, grid=grid(8192), stream=stream0)
        del arg180_1
        del arg181_1
        del arg182_1
        buf237 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [linear_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg184_1, reinterpret_tensor(buf236, (8192, 768), (768, 1), 0), reinterpret_tensor(arg183_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf237)
        del arg183_1
        del arg184_1
        buf238 = reinterpret_tensor(buf229, (8192, 768), (768, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg186_1, reinterpret_tensor(buf236, (8192, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf238)
        del arg185_1
        del arg186_1
        buf239 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg188_1, reinterpret_tensor(buf236, (8192, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf239)
        del arg187_1
        del arg188_1
        buf240 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [attn_output_33], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_2.run(buf240, 50331648, grid=grid(50331648), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_33], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf241 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf237, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf238, (16, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf239, (16, 12, 512, 64), (393216, 64, 768, 1), 0), buf240, False)
        del buf237
        del buf238
        del buf240
        buf242 = buf241[0]
        del buf241
        buf246 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (8192, 768), (768, 1), 0), reinterpret_tensor(arg189_1, (768, 768), (1, 768), 0), out=buf246)
        del arg189_1
        buf250 = reinterpret_tensor(buf242, (16, 512, 768), (393216, 768, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [add_25, hidden_states_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf246, arg190_1, buf236, arg191_1, arg192_1, buf250, 8192, 768, grid=grid(8192), stream=stream0)
        del arg190_1
        del arg191_1
        del arg192_1
        buf251 = reinterpret_tensor(buf231, (8192, 3072), (3072, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (8192, 768), (768, 1), 0), reinterpret_tensor(arg193_1, (768, 3072), (1, 768), 0), out=buf251)
        del arg193_1
        buf252 = reinterpret_tensor(buf251, (16, 512, 3072), (1572864, 3072, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_92], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf252, arg194_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg194_1
        buf253 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf252, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg195_1, (3072, 768), (1, 3072), 0), out=buf253)
        del arg195_1
        del buf252
        buf257 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [add_26, hidden_states_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf253, arg196_1, buf250, arg197_1, arg198_1, buf257, 8192, 768, grid=grid(8192), stream=stream0)
        del arg196_1
        del arg197_1
        del arg198_1
        del buf250
        del buf253
        buf258 = empty_strided_cuda((8192, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (8192, 768), (768, 1), 0), reinterpret_tensor(arg199_1, (768, 2), (1, 768), 0), out=buf258)
        del arg199_1
        del buf257
        buf259 = empty_strided_cuda((16, 512), (512, 1), torch.float32)
        buf260 = empty_strided_cuda((16, 1), (1, 16), torch.float32)
        buf261 = empty_strided_cuda((16, 1), (1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [start_logits_1, start_loss], Original ATen: [aten.clone, aten._log_softmax]
        triton_red_fused__log_softmax_clone_5.run(buf258, arg200_1, buf259, buf260, buf261, 16, 512, grid=grid(16), stream=stream0)
        buf264 = empty_strided_cuda((16, 512), (512, 1), torch.float32)
        buf265 = empty_strided_cuda((16, 1), (1, 16), torch.float32)
        buf266 = empty_strided_cuda((16, 1), (1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [end_logits_1, end_loss], Original ATen: [aten.clone, aten._log_softmax]
        triton_red_fused__log_softmax_clone_6.run(buf258, arg200_1, buf264, buf265, buf266, 16, 512, grid=grid(16), stream=stream0)
        del arg200_1
        del buf258
        buf262 = empty_strided_cuda((), (), torch.float32)
        buf269 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [start_positions, start_loss, end_positions, end_loss, add_27, total_loss], Original ATen: [aten.clamp, aten.nll_loss_forward, aten.add, aten.div]
        triton_per_fused_add_clamp_div_nll_loss_forward_7.run(buf269, arg201_1, buf259, buf260, buf261, arg202_1, buf264, buf265, buf266, 1, 16, grid=grid(1), stream=stream0)
        del arg201_1
        del arg202_1
        del buf260
        del buf261
        del buf265
        del buf266
    return (buf269, buf259, buf264, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg202_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('RobertaForQuestionAnswering', benchmark_compiled_module)
