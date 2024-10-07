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


# kernel path: /tmp/torchinductor_sahanp/id/cidiuq6pguxspbptwdljvombsv2jboequ67bo7pdqq5cq3lm3sv2.py
# Topologically Sorted Source Nodes: [inputs_embeds, token_type_embeddings, embeddings, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embeddings => add
#   embeddings_1 => add_1
#   embeddings_2 => add_2, add_3, mul, mul_1, rsqrt, sub, var_mean
#   inputs_embeds => embedding
#   position_embeddings => embedding_2
#   token_type_embeddings => embedding_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1, 0), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %expand), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %arg2_1), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg6_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg7_1), kwargs = {})
triton_per_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x3 = xindex
    r2 = rindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr6 + (r2), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr7 + (r2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, RBLOCK], 30000, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 30000), "index out of bounds: 0 <= tmp4 < 30000")
    tmp6 = tl.load(in_ptr1 + (r2 + (128*tmp4)), None)
    tmp8 = tl.full([XBLOCK, RBLOCK], 2, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 2), "index out of bounds: 0 <= tmp11 < 2")
    tmp13 = tl.load(in_ptr3 + (r2 + (128*tmp11)), None)
    tmp14 = tmp6 + tmp13
    tmp16 = tl.full([XBLOCK, RBLOCK], 512, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert((0 <= tmp19) & (tmp19 < 512), "index out of bounds: 0 <= tmp19 < 512")
    tmp21 = tl.load(in_ptr5 + (r2 + (128*tmp19)), None)
    tmp22 = tmp14 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None]
    tmp28 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp23 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.sum(tmp33, 1)[:, None]
    tmp36 = tmp22 - tmp30
    tmp37 = 128.0
    tmp38 = tmp35 / tmp37
    tmp39 = 1e-12
    tmp40 = tmp38 + tmp39
    tmp41 = libdevice.rsqrt(tmp40)
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tl.store(out_ptr0 + (r2 + (128*x3)), tmp22, None)
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp46, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/a2/ca2w2sepg2u3nltut5yk3m7p4e7mk2vuwm5ls5jauadp5fykwmd2.py
# Topologically Sorted Source Nodes: [attention_output], Original ATen: [aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attention_output => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_2, %permute_4, %permute_6, %expand_2, False), kwargs = {})
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
    xnumel = 67108864
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


# kernel path: /tmp/torchinductor_sahanp/g2/cg2rdhbuci3g6qgwysrzvigo3tskfan6otbyx6etxq4a4apqzugv.py
# Topologically Sorted Source Nodes: [add_1, layernormed_context_layer], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_1 => add_4
#   layernormed_context_layer => add_5, add_6, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %view_13), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_7), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-12), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg18_1), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg19_1), kwargs = {})
triton_red_fused_add_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp9 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 4096.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-12
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp24, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nf/cnfac73y7vd4j4z2u63kdy6hfghjx35u7mcyzegsuyuxr6gbemnt.py
# Topologically Sorted Source Nodes: [mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, ffn_output_1], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_2 => add_7
#   add_3 => add_8
#   ffn_output_1 => mul_7
#   mul => mul_4
#   mul_1 => mul_5
#   mul_2 => mul_6
#   pow_1 => pow_1
#   tanh => tanh
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, 0.5), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_15, 3.0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_15, %mul_5), kwargs = {})
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
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 16384
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


# kernel path: /tmp/torchinductor_sahanp/al/calkyf5rvbiy7l6dr2lktskx4llxlhz67nfq4vag44btoxjwk6dg.py
# Topologically Sorted Source Nodes: [add_4, hidden_states_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_4 => add_9
#   hidden_states_1 => add_10, add_11, mul_8, mul_9, rsqrt_2, sub_3, var_mean_2
# Graph fragment:
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %add_6), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_9), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-12), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %arg24_1), kwargs = {})
#   %add_11 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg25_1), kwargs = {})
triton_red_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r1 + (4096*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 4096.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-12
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp24, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/iu/ciuqv7ipca5jjwlj4kffpefiahk5aaowavodfwv5ofxsz7ow5zqf.py
# Topologically Sorted Source Nodes: [mul_48, pow_13, mul_49, add_49, mul_50, tanh_12, add_50, hidden_states_14, hidden_states_15], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_49 => add_100
#   add_50 => add_101
#   hidden_states_14 => mul_101
#   hidden_states_15 => add_102, add_103, mul_102, mul_103, rsqrt_25, sub_26, var_mean_25
#   mul_48 => mul_98
#   mul_49 => mul_99
#   mul_50 => mul_100
#   pow_13 => pow_13
#   tanh_12 => tanh_12
# Graph fragment:
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_195, 0.5), kwargs = {})
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_195, 3.0), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_13, 0.044715), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_195, %mul_99), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_100, 0.7978845608028654), kwargs = {})
#   %tanh_12 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_100,), kwargs = {})
#   %add_101 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh_12, 1.0), kwargs = {})
#   %mul_101 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_98, %add_101), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_101, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_101, %getitem_99), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_98, 1e-12), kwargs = {})
#   %rsqrt_25 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_102,), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %rsqrt_25), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, %arg28_1), kwargs = {})
#   %add_103 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_103, %arg29_1), kwargs = {})
triton_per_fused_add_mul_native_layer_norm_pow_tanh_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_pow_tanh_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
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
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp20 = tl.sum(tmp18, 1)[:, None]
    tmp21 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp16 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.sum(tmp26, 1)[:, None]
    tmp29 = tmp15 - tmp23
    tmp30 = 128.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-12
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp39, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ih/cihdtvbyaewja47gns4p47ebafed3uje43uptp6jc37rare6tlnf.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   masked_lm_loss => amax, exp, sub_27, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_198, [1], True), kwargs = {})
#   %sub_27 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_198, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_27,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_red_fused__log_softmax_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 30000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (30016*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, None)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (30016*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/iw/ciwh2soz3ll5iyx5iwixl75alpc2csyizso7odgib2kzyb2iyupt.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   masked_lm_loss => convert_element_type_1, div, full_default_1, ne_1, ne_2, neg, sum_2, sum_3, where_2
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_199, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_2,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_199, -100), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type_1), kwargs = {})
triton_red_fused_nll_loss_forward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
        tmp5 = tl.full([XBLOCK, RBLOCK], 30000, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 30000)) | ~(rmask), "index out of bounds: 0 <= tmp8 < 30000")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (30016*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 512), (512, 1))
    assert_size_stride(arg1_1, (1, 512), (512, 1))
    assert_size_stride(arg2_1, (1, 512), (512, 1))
    assert_size_stride(arg3_1, (30000, 128), (128, 1))
    assert_size_stride(arg4_1, (2, 128), (128, 1))
    assert_size_stride(arg5_1, (512, 128), (128, 1))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (4096, 128), (128, 1))
    assert_size_stride(arg9_1, (4096, ), (1, ))
    assert_size_stride(arg10_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg11_1, (4096, ), (1, ))
    assert_size_stride(arg12_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg13_1, (4096, ), (1, ))
    assert_size_stride(arg14_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg15_1, (4096, ), (1, ))
    assert_size_stride(arg16_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg17_1, (4096, ), (1, ))
    assert_size_stride(arg18_1, (4096, ), (1, ))
    assert_size_stride(arg19_1, (4096, ), (1, ))
    assert_size_stride(arg20_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg21_1, (16384, ), (1, ))
    assert_size_stride(arg22_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg23_1, (4096, ), (1, ))
    assert_size_stride(arg24_1, (4096, ), (1, ))
    assert_size_stride(arg25_1, (4096, ), (1, ))
    assert_size_stride(arg26_1, (128, 4096), (4096, 1))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (30000, ), (1, ))
    assert_size_stride(arg31_1, (4, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 512, 128), (65536, 128, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 512, 128), (65536, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, token_type_embeddings, embeddings, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg0_1, arg3_1, arg1_1, arg4_1, arg2_1, arg5_1, arg6_1, arg7_1, buf0, buf4, 2048, 128, grid=grid(2048), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        buf5 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg9_1, reinterpret_tensor(buf4, (2048, 128), (128, 1), 0), reinterpret_tensor(arg8_1, (128, 4096), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del arg8_1
        del arg9_1
        buf6 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, buf5, reinterpret_tensor(arg10_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf6)
        buf7 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, buf5, reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf7)
        buf8 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, buf5, reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf8)
        buf9 = empty_strided_cuda((4, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention_output], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf9, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [attention_output], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf6, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf7, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf8, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), buf9, False)
        del buf6
        buf11 = buf10[0]
        del buf10
        buf15 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf15)
        buf19 = reinterpret_tensor(buf11, (4, 512, 4096), (2097152, 4096, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [add_1, layernormed_context_layer], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf5, buf15, arg17_1, arg18_1, arg19_1, buf19, 2048, 4096, grid=grid(2048), stream=stream0)
        buf20 = empty_strided_cuda((2048, 16384), (16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 16384), (1, 4096), 0), out=buf20)
        buf21 = reinterpret_tensor(buf20, (4, 512, 16384), (8388608, 16384, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, ffn_output_1], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf21, arg21_1, 33554432, grid=grid(33554432), stream=stream0)
        buf22 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (2048, 16384), (16384, 1), 0), reinterpret_tensor(arg22_1, (16384, 4096), (1, 16384), 0), out=buf22)
        buf26 = reinterpret_tensor(buf15, (4, 512, 4096), (2097152, 4096, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [add_4, hidden_states_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf22, arg23_1, buf19, arg24_1, arg25_1, buf26, 2048, 4096, grid=grid(2048), stream=stream0)
        buf27 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf26, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg10_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf27)
        buf28 = reinterpret_tensor(buf19, (2048, 4096), (4096, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf26, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf28)
        buf29 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf26, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf29)
        buf30 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [attention_output_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf30, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [attention_output_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf31 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf27, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf28, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf29, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), buf30, False)
        del buf27
        buf32 = buf31[0]
        del buf31
        buf36 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf36)
        buf40 = reinterpret_tensor(buf32, (4, 512, 4096), (2097152, 4096, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [add_5, layernormed_context_layer_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf26, buf36, arg17_1, arg18_1, arg19_1, buf40, 2048, 4096, grid=grid(2048), stream=stream0)
        buf41 = reinterpret_tensor(buf21, (2048, 16384), (16384, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 16384), (1, 4096), 0), out=buf41)
        buf42 = reinterpret_tensor(buf41, (4, 512, 16384), (8388608, 16384, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, ffn_output_4], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf42, arg21_1, 33554432, grid=grid(33554432), stream=stream0)
        buf43 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (2048, 16384), (16384, 1), 0), reinterpret_tensor(arg22_1, (16384, 4096), (1, 16384), 0), out=buf43)
        buf47 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [add_8, hidden_states_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf43, arg23_1, buf40, arg24_1, arg25_1, buf47, 2048, 4096, grid=grid(2048), stream=stream0)
        buf48 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf47, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg10_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf48)
        buf49 = reinterpret_tensor(buf40, (2048, 4096), (4096, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf47, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf49)
        buf50 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf47, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf50)
        buf51 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [attention_output_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf51, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [attention_output_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf52 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf48, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf49, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf50, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), buf51, False)
        del buf48
        buf53 = buf52[0]
        del buf52
        buf57 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf57)
        buf61 = reinterpret_tensor(buf53, (4, 512, 4096), (2097152, 4096, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [add_9, layernormed_context_layer_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf47, buf57, arg17_1, arg18_1, arg19_1, buf61, 2048, 4096, grid=grid(2048), stream=stream0)
        buf62 = reinterpret_tensor(buf42, (2048, 16384), (16384, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 16384), (1, 4096), 0), out=buf62)
        buf63 = reinterpret_tensor(buf62, (4, 512, 16384), (8388608, 16384, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, ffn_output_7], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf63, arg21_1, 33554432, grid=grid(33554432), stream=stream0)
        buf64 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (2048, 16384), (16384, 1), 0), reinterpret_tensor(arg22_1, (16384, 4096), (1, 16384), 0), out=buf64)
        buf68 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [add_12, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf64, arg23_1, buf61, arg24_1, arg25_1, buf68, 2048, 4096, grid=grid(2048), stream=stream0)
        buf69 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf68, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg10_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf69)
        buf70 = reinterpret_tensor(buf61, (2048, 4096), (4096, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf68, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf70)
        buf71 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf68, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf71)
        buf72 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [attention_output_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf72, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [attention_output_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf73 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf69, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf70, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf71, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), buf72, False)
        del buf69
        buf74 = buf73[0]
        del buf73
        buf78 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf78)
        buf82 = reinterpret_tensor(buf74, (4, 512, 4096), (2097152, 4096, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [add_13, layernormed_context_layer_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf68, buf78, arg17_1, arg18_1, arg19_1, buf82, 2048, 4096, grid=grid(2048), stream=stream0)
        buf83 = reinterpret_tensor(buf63, (2048, 16384), (16384, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 16384), (1, 4096), 0), out=buf83)
        buf84 = reinterpret_tensor(buf83, (4, 512, 16384), (8388608, 16384, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, ffn_output_10], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf84, arg21_1, 33554432, grid=grid(33554432), stream=stream0)
        buf85 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (2048, 16384), (16384, 1), 0), reinterpret_tensor(arg22_1, (16384, 4096), (1, 16384), 0), out=buf85)
        buf89 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [add_16, hidden_states_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf85, arg23_1, buf82, arg24_1, arg25_1, buf89, 2048, 4096, grid=grid(2048), stream=stream0)
        buf90 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf89, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg10_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf90)
        buf91 = reinterpret_tensor(buf82, (2048, 4096), (4096, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf89, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf91)
        buf92 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf89, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf92)
        buf93 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [attention_output_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf93, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [attention_output_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf94 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf90, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf91, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf92, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), buf93, False)
        del buf90
        buf95 = buf94[0]
        del buf94
        buf99 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf99)
        buf103 = reinterpret_tensor(buf95, (4, 512, 4096), (2097152, 4096, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [add_17, layernormed_context_layer_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf89, buf99, arg17_1, arg18_1, arg19_1, buf103, 2048, 4096, grid=grid(2048), stream=stream0)
        buf104 = reinterpret_tensor(buf84, (2048, 16384), (16384, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 16384), (1, 4096), 0), out=buf104)
        buf105 = reinterpret_tensor(buf104, (4, 512, 16384), (8388608, 16384, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, ffn_output_13], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf105, arg21_1, 33554432, grid=grid(33554432), stream=stream0)
        buf106 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (2048, 16384), (16384, 1), 0), reinterpret_tensor(arg22_1, (16384, 4096), (1, 16384), 0), out=buf106)
        buf110 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [add_20, hidden_states_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf106, arg23_1, buf103, arg24_1, arg25_1, buf110, 2048, 4096, grid=grid(2048), stream=stream0)
        buf111 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf110, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg10_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf111)
        buf112 = reinterpret_tensor(buf103, (2048, 4096), (4096, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf110, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf112)
        buf113 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf110, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf113)
        buf114 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [attention_output_15], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf114, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [attention_output_15], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf115 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf111, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf112, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf113, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), buf114, False)
        del buf111
        buf116 = buf115[0]
        del buf115
        buf120 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf120)
        buf124 = reinterpret_tensor(buf116, (4, 512, 4096), (2097152, 4096, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [add_21, layernormed_context_layer_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf110, buf120, arg17_1, arg18_1, arg19_1, buf124, 2048, 4096, grid=grid(2048), stream=stream0)
        buf125 = reinterpret_tensor(buf105, (2048, 16384), (16384, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 16384), (1, 4096), 0), out=buf125)
        buf126 = reinterpret_tensor(buf125, (4, 512, 16384), (8388608, 16384, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, ffn_output_16], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf126, arg21_1, 33554432, grid=grid(33554432), stream=stream0)
        buf127 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (2048, 16384), (16384, 1), 0), reinterpret_tensor(arg22_1, (16384, 4096), (1, 16384), 0), out=buf127)
        buf131 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [add_24, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf127, arg23_1, buf124, arg24_1, arg25_1, buf131, 2048, 4096, grid=grid(2048), stream=stream0)
        buf132 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf131, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg10_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf132)
        buf133 = reinterpret_tensor(buf124, (2048, 4096), (4096, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf131, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf133)
        buf134 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf131, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf134)
        buf135 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [attention_output_18], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf135, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [attention_output_18], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf136 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf132, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf133, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf134, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), buf135, False)
        del buf132
        buf137 = buf136[0]
        del buf136
        buf141 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf137, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf141)
        buf145 = reinterpret_tensor(buf137, (4, 512, 4096), (2097152, 4096, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [add_25, layernormed_context_layer_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf131, buf141, arg17_1, arg18_1, arg19_1, buf145, 2048, 4096, grid=grid(2048), stream=stream0)
        buf146 = reinterpret_tensor(buf126, (2048, 16384), (16384, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 16384), (1, 4096), 0), out=buf146)
        buf147 = reinterpret_tensor(buf146, (4, 512, 16384), (8388608, 16384, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, ffn_output_19], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf147, arg21_1, 33554432, grid=grid(33554432), stream=stream0)
        buf148 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (2048, 16384), (16384, 1), 0), reinterpret_tensor(arg22_1, (16384, 4096), (1, 16384), 0), out=buf148)
        buf152 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [add_28, hidden_states_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf148, arg23_1, buf145, arg24_1, arg25_1, buf152, 2048, 4096, grid=grid(2048), stream=stream0)
        buf153 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf152, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg10_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf153)
        buf154 = reinterpret_tensor(buf145, (2048, 4096), (4096, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf152, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf154)
        buf155 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf152, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf155)
        buf156 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [attention_output_21], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf156, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [attention_output_21], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf157 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf153, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf154, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf155, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), buf156, False)
        del buf153
        buf158 = buf157[0]
        del buf157
        buf162 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf162)
        buf166 = reinterpret_tensor(buf158, (4, 512, 4096), (2097152, 4096, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [add_29, layernormed_context_layer_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf152, buf162, arg17_1, arg18_1, arg19_1, buf166, 2048, 4096, grid=grid(2048), stream=stream0)
        buf167 = reinterpret_tensor(buf147, (2048, 16384), (16384, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 16384), (1, 4096), 0), out=buf167)
        buf168 = reinterpret_tensor(buf167, (4, 512, 16384), (8388608, 16384, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, ffn_output_22], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf168, arg21_1, 33554432, grid=grid(33554432), stream=stream0)
        buf169 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (2048, 16384), (16384, 1), 0), reinterpret_tensor(arg22_1, (16384, 4096), (1, 16384), 0), out=buf169)
        buf173 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [add_32, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf169, arg23_1, buf166, arg24_1, arg25_1, buf173, 2048, 4096, grid=grid(2048), stream=stream0)
        buf174 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf173, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg10_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf174)
        buf175 = reinterpret_tensor(buf166, (2048, 4096), (4096, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf173, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf175)
        buf176 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf173, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf176)
        buf177 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [attention_output_24], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf177, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [attention_output_24], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf178 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf174, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf175, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf176, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), buf177, False)
        del buf174
        buf179 = buf178[0]
        del buf178
        buf183 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf183)
        buf187 = reinterpret_tensor(buf179, (4, 512, 4096), (2097152, 4096, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [add_33, layernormed_context_layer_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf173, buf183, arg17_1, arg18_1, arg19_1, buf187, 2048, 4096, grid=grid(2048), stream=stream0)
        buf188 = reinterpret_tensor(buf168, (2048, 16384), (16384, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 16384), (1, 4096), 0), out=buf188)
        buf189 = reinterpret_tensor(buf188, (4, 512, 16384), (8388608, 16384, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, ffn_output_25], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf189, arg21_1, 33554432, grid=grid(33554432), stream=stream0)
        buf190 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (2048, 16384), (16384, 1), 0), reinterpret_tensor(arg22_1, (16384, 4096), (1, 16384), 0), out=buf190)
        buf194 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [add_36, hidden_states_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf190, arg23_1, buf187, arg24_1, arg25_1, buf194, 2048, 4096, grid=grid(2048), stream=stream0)
        buf195 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf194, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg10_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf195)
        buf196 = reinterpret_tensor(buf187, (2048, 4096), (4096, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf194, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf196)
        buf197 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf194, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf197)
        buf198 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [attention_output_27], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf198, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [attention_output_27], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf199 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf195, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf196, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf197, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), buf198, False)
        del buf195
        buf200 = buf199[0]
        del buf199
        buf204 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf204)
        buf208 = reinterpret_tensor(buf200, (4, 512, 4096), (2097152, 4096, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [add_37, layernormed_context_layer_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf194, buf204, arg17_1, arg18_1, arg19_1, buf208, 2048, 4096, grid=grid(2048), stream=stream0)
        buf209 = reinterpret_tensor(buf189, (2048, 16384), (16384, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf208, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 16384), (1, 4096), 0), out=buf209)
        buf210 = reinterpret_tensor(buf209, (4, 512, 16384), (8388608, 16384, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, ffn_output_28], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf210, arg21_1, 33554432, grid=grid(33554432), stream=stream0)
        buf211 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (2048, 16384), (16384, 1), 0), reinterpret_tensor(arg22_1, (16384, 4096), (1, 16384), 0), out=buf211)
        buf215 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [add_40, hidden_states_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf211, arg23_1, buf208, arg24_1, arg25_1, buf215, 2048, 4096, grid=grid(2048), stream=stream0)
        buf216 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf215, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg10_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf216)
        buf217 = reinterpret_tensor(buf208, (2048, 4096), (4096, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf215, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf217)
        buf218 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf215, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf218)
        buf219 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [attention_output_30], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf219, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [attention_output_30], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf220 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf216, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf217, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf218, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), buf219, False)
        del buf216
        buf221 = buf220[0]
        del buf220
        buf225 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf225)
        buf229 = reinterpret_tensor(buf221, (4, 512, 4096), (2097152, 4096, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [add_41, layernormed_context_layer_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf215, buf225, arg17_1, arg18_1, arg19_1, buf229, 2048, 4096, grid=grid(2048), stream=stream0)
        buf230 = reinterpret_tensor(buf210, (2048, 16384), (16384, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 16384), (1, 4096), 0), out=buf230)
        buf231 = reinterpret_tensor(buf230, (4, 512, 16384), (8388608, 16384, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, ffn_output_31], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf231, arg21_1, 33554432, grid=grid(33554432), stream=stream0)
        buf232 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (2048, 16384), (16384, 1), 0), reinterpret_tensor(arg22_1, (16384, 4096), (1, 16384), 0), out=buf232)
        buf236 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [add_44, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf232, arg23_1, buf229, arg24_1, arg25_1, buf236, 2048, 4096, grid=grid(2048), stream=stream0)
        buf237 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf236, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg10_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf237)
        del arg10_1
        del arg11_1
        buf238 = reinterpret_tensor(buf229, (2048, 4096), (4096, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf236, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf238)
        del arg12_1
        del arg13_1
        buf239 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [linear_69], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf236, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf239)
        del arg14_1
        del arg15_1
        buf240 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [attention_output_33], Original ATen: [aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_1.run(buf240, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [attention_output_33], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf241 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf237, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf238, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), reinterpret_tensor(buf239, (4, 64, 512, 64), (2097152, 64, 4096, 1), 0), buf240, False)
        del buf237
        del buf238
        del buf240
        buf242 = buf241[0]
        del buf241
        buf246 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf246)
        del arg16_1
        buf250 = reinterpret_tensor(buf242, (4, 512, 4096), (2097152, 4096, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [add_45, layernormed_context_layer_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf236, buf246, arg17_1, arg18_1, arg19_1, buf250, 2048, 4096, grid=grid(2048), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        buf251 = reinterpret_tensor(buf231, (2048, 16384), (16384, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg20_1, (4096, 16384), (1, 4096), 0), out=buf251)
        del arg20_1
        buf252 = reinterpret_tensor(buf251, (4, 512, 16384), (8388608, 16384, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, ffn_output_34], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_3.run(buf252, arg21_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg21_1
        buf253 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf252, (2048, 16384), (16384, 1), 0), reinterpret_tensor(arg22_1, (16384, 4096), (1, 16384), 0), out=buf253)
        del arg22_1
        del buf252
        buf257 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [add_48, hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf253, arg23_1, buf250, arg24_1, arg25_1, buf257, 2048, 4096, grid=grid(2048), stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        del buf250
        del buf253
        buf258 = reinterpret_tensor(buf4, (2048, 128), (128, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg26_1, (4096, 128), (1, 4096), 0), out=buf258)
        del arg26_1
        del buf257
        buf262 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [mul_48, pow_13, mul_49, add_49, mul_50, tanh_12, add_50, hidden_states_14, hidden_states_15], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_pow_tanh_5.run(buf258, arg27_1, arg28_1, arg29_1, buf262, 2048, 128, grid=grid(2048), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del buf258
        buf263 = empty_strided_cuda((2048, 30000), (30016, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg30_1, reinterpret_tensor(buf262, (2048, 128), (128, 1), 0), reinterpret_tensor(arg3_1, (128, 30000), (1, 128), 0), alpha=1, beta=1, out=buf263)
        del arg30_1
        del arg3_1
        del buf262
        buf264 = empty_strided_cuda((2048, 1), (1, 2048), torch.float32)
        buf265 = empty_strided_cuda((2048, 1), (1, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_6.run(buf263, buf264, buf265, 2048, 30000, grid=grid(2048), stream=stream0)
        buf266 = empty_strided_cuda((), (), torch.float32)
        buf268 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_7.run(buf268, arg31_1, buf263, buf264, buf265, 1, 2048, grid=grid(1), stream=stream0)
        del arg31_1
        del buf264
        del buf265
    return (buf268, reinterpret_tensor(buf263, (4, 512, 30000), (15368192, 30016, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((16384, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((30000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AlbertForMaskedLM', benchmark_compiled_module)
