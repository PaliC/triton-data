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


# kernel path: /tmp/torchinductor_sahanp/nt/cntkraaoljsbj32xusytyoeevfm5gnijc3lkvila77hfw2nfp3v6.py
# Topologically Sorted Source Nodes: [inputs_embeds, token_type_ids, token_type_embeddings, embeddings, position_embeddings, embeddings_1, ln_outputs], Original ATen: [aten.embedding, aten.zeros, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embeddings => add
#   embeddings_1 => add_1
#   inputs_embeds => embedding
#   ln_outputs => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
#   position_embeddings => embedding_2
#   token_type_embeddings => embedding_1
#   token_type_ids => full_default
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %arg1_1, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 512], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %full_default), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg391_1), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg6_1), kwargs = {})
triton_per_fused_add_embedding_native_layer_norm_zeros_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_zeros_0', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
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
    x3 = xindex
    r2 = rindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr6 + (r2), None, eviction_policy='evict_last')
    tmp1 = tl.full([RBLOCK], 29056, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 29056), "index out of bounds: 0 <= tmp4 < 29056")
    tmp6 = tl.load(in_ptr1 + (r2 + (1024*tmp4)), None)
    tmp8 = tmp6 + tmp7
    tmp10 = tl.full([RBLOCK], 512, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tl.device_assert((0 <= tmp13) & (tmp13 < 512), "index out of bounds: 0 <= tmp13 < 512")
    tmp15 = tl.load(in_ptr4 + (r2 + (1024*tmp13)), None)
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tl.full([1], 1024, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp17 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp30 = tmp16 - tmp24
    tmp31 = 1024.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-12
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tl.store(out_ptr0 + (r2 + (1024*x3)), tmp16, None)
    tl.store(out_ptr3 + (r2 + (1024*x3)), tmp40, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sm/csmlqp2g4fmpshil3utxhubckhmppvnq4rxvonoiw3gmivvdqhyn.py
# Topologically Sorted Source Nodes: [attention_output, ln_output], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attention_output => add_5
#   ln_output => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
# Graph fragment:
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_17), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_3), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-12), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg15_1), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg16_1), kwargs = {})
triton_per_fused_add_native_layer_norm_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_1', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    tmp21 = 1e-12
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n5/cn5nqpzy3fevxp4w6x24jqn6xi7y7mmj32go7ce4ovb2joflpoez.py
# Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_3 => add_8, erf, mul_5, mul_6, mul_7
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_8), kwargs = {})
triton_poi_fused_gelu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/yj/cyj7fwta7b2tk7zm5cfdbyqlf3rxesi5fl5yod3nrep4b4sykqiz.py
# Topologically Sorted Source Nodes: [attention_output, layer_output, ln_outputs_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attention_output => add_5
#   layer_output => add_9
#   ln_outputs_1 => add_10, add_11, mul_8, mul_9, rsqrt_2, sub_4, var_mean_2
# Graph fragment:
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_17), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_21), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_5), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-12), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_2), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %arg21_1), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg22_1), kwargs = {})
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    tmp25 = 1e-12
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, None)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ww/cwwdy47nn5ap4wjm7nctc5bhxsrdaymsypejaaxjkcglms36g7eu.py
# Topologically Sorted Source Nodes: [hidden_states_146, hidden_states_147], Original ATen: [aten.gelu, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_146 => add_196, erf_24, mul_171, mul_172, mul_173
#   hidden_states_147 => add_197, add_198, mul_174, mul_175, rsqrt_49, sub_74, var_mean_49
# Graph fragment:
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_529, 0.5), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_529, 0.7071067811865476), kwargs = {})
#   %erf_24 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_172,), kwargs = {})
#   %add_196 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_24, 1), kwargs = {})
#   %mul_173 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_171, %add_196), kwargs = {})
#   %var_mean_49 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_173, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_74 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_173, %getitem_99), kwargs = {})
#   %add_197 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_98, 1e-12), kwargs = {})
#   %rsqrt_49 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_197,), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_74, %rsqrt_49), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_174, %arg394_1), kwargs = {})
#   %add_198 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_175, %arg395_1), kwargs = {})
triton_per_fused_gelu_native_layer_norm_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_4', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tl.full([1], 1024, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp11 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tmp10 - tmp18
    tmp25 = 1024.0
    tmp26 = tmp23 / tmp25
    tmp27 = 1e-12
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp34, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ai/cai24awatpulfwgaj5lpmrvwjjk25a7oxon66khyicpdo4lszphd.py
# Topologically Sorted Source Nodes: [lm_loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   lm_loss => amax_24, exp_24, sub_75, sum_25
# Graph fragment:
#   %amax_24 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_532, [1], True), kwargs = {})
#   %sub_75 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_532, %amax_24), kwargs = {})
#   %exp_24 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_75,), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [1], True), kwargs = {})
triton_red_fused__log_softmax_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2044
    rnumel = 29056
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
        tmp0 = tl.load(in_ptr0 + (r1 + (29056*(x0 % 511)) + (14876672*(x0 // 511))), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (29056*(x0 % 511)) + (14876672*(x0 // 511))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nx/cnxfu7qsnu2xxdjkv56ibgw7aqotqfwdkxqmgozsroibft5xwasg.py
# Topologically Sorted Source Nodes: [lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   lm_loss => convert_element_type, div_48, full_default_3, ne_1, ne_2, neg, sum_26, sum_27, where_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_533, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_3), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_533, -100), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_26, torch.float32), kwargs = {})
#   %div_48 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_27, %convert_element_type), kwargs = {})
triton_red_fused_nll_loss_forward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {5: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2044
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
        tmp5 = tl.full([XBLOCK, RBLOCK], 29056, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 29056)) | ~(rmask), "index out of bounds: 0 <= tmp8 < 29056")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (29056*(r0 % 511)) + (14876672*(r0 // 511))), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 512), (512, 1))
    assert_size_stride(arg1_1, (4, 512), (512, 1))
    assert_size_stride(arg2_1, (29056, 1024), (1024, 1))
    assert_size_stride(arg3_1, (512, 1024), (1024, 1))
    assert_size_stride(arg4_1, (2, 1024), (1024, 1))
    assert_size_stride(arg5_1, (1024, ), (1, ))
    assert_size_stride(arg6_1, (1024, ), (1, ))
    assert_size_stride(arg7_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg8_1, (1024, ), (1, ))
    assert_size_stride(arg9_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg10_1, (1024, ), (1, ))
    assert_size_stride(arg11_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg12_1, (1024, ), (1, ))
    assert_size_stride(arg13_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg14_1, (1024, ), (1, ))
    assert_size_stride(arg15_1, (1024, ), (1, ))
    assert_size_stride(arg16_1, (1024, ), (1, ))
    assert_size_stride(arg17_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg18_1, (4096, ), (1, ))
    assert_size_stride(arg19_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg20_1, (1024, ), (1, ))
    assert_size_stride(arg21_1, (1024, ), (1, ))
    assert_size_stride(arg22_1, (1024, ), (1, ))
    assert_size_stride(arg23_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg24_1, (1024, ), (1, ))
    assert_size_stride(arg25_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg26_1, (1024, ), (1, ))
    assert_size_stride(arg27_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg28_1, (1024, ), (1, ))
    assert_size_stride(arg29_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg30_1, (1024, ), (1, ))
    assert_size_stride(arg31_1, (1024, ), (1, ))
    assert_size_stride(arg32_1, (1024, ), (1, ))
    assert_size_stride(arg33_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg34_1, (4096, ), (1, ))
    assert_size_stride(arg35_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg40_1, (1024, ), (1, ))
    assert_size_stride(arg41_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg42_1, (1024, ), (1, ))
    assert_size_stride(arg43_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (1024, ), (1, ))
    assert_size_stride(arg48_1, (1024, ), (1, ))
    assert_size_stride(arg49_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg50_1, (4096, ), (1, ))
    assert_size_stride(arg51_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg52_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (1024, ), (1, ))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg57_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg63_1, (1024, ), (1, ))
    assert_size_stride(arg64_1, (1024, ), (1, ))
    assert_size_stride(arg65_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg66_1, (4096, ), (1, ))
    assert_size_stride(arg67_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (1024, ), (1, ))
    assert_size_stride(arg70_1, (1024, ), (1, ))
    assert_size_stride(arg71_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg72_1, (1024, ), (1, ))
    assert_size_stride(arg73_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg74_1, (1024, ), (1, ))
    assert_size_stride(arg75_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg78_1, (1024, ), (1, ))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg82_1, (4096, ), (1, ))
    assert_size_stride(arg83_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, ), (1, ))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg87_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg88_1, (1024, ), (1, ))
    assert_size_stride(arg89_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg90_1, (1024, ), (1, ))
    assert_size_stride(arg91_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg94_1, (1024, ), (1, ))
    assert_size_stride(arg95_1, (1024, ), (1, ))
    assert_size_stride(arg96_1, (1024, ), (1, ))
    assert_size_stride(arg97_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg98_1, (4096, ), (1, ))
    assert_size_stride(arg99_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (1024, ), (1, ))
    assert_size_stride(arg103_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg104_1, (1024, ), (1, ))
    assert_size_stride(arg105_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg106_1, (1024, ), (1, ))
    assert_size_stride(arg107_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (1024, ), (1, ))
    assert_size_stride(arg112_1, (1024, ), (1, ))
    assert_size_stride(arg113_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg114_1, (4096, ), (1, ))
    assert_size_stride(arg115_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg120_1, (1024, ), (1, ))
    assert_size_stride(arg121_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg122_1, (1024, ), (1, ))
    assert_size_stride(arg123_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg124_1, (1024, ), (1, ))
    assert_size_stride(arg125_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg126_1, (1024, ), (1, ))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg130_1, (4096, ), (1, ))
    assert_size_stride(arg131_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg142_1, (1024, ), (1, ))
    assert_size_stride(arg143_1, (1024, ), (1, ))
    assert_size_stride(arg144_1, (1024, ), (1, ))
    assert_size_stride(arg145_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg146_1, (4096, ), (1, ))
    assert_size_stride(arg147_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg158_1, (1024, ), (1, ))
    assert_size_stride(arg159_1, (1024, ), (1, ))
    assert_size_stride(arg160_1, (1024, ), (1, ))
    assert_size_stride(arg161_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg162_1, (4096, ), (1, ))
    assert_size_stride(arg163_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (1024, ), (1, ))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg174_1, (1024, ), (1, ))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg178_1, (4096, ), (1, ))
    assert_size_stride(arg179_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg186_1, (1024, ), (1, ))
    assert_size_stride(arg187_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg188_1, (1024, ), (1, ))
    assert_size_stride(arg189_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (1024, ), (1, ))
    assert_size_stride(arg193_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg194_1, (4096, ), (1, ))
    assert_size_stride(arg195_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (1024, ), (1, ))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg202_1, (1024, ), (1, ))
    assert_size_stride(arg203_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg204_1, (1024, ), (1, ))
    assert_size_stride(arg205_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (1024, ), (1, ))
    assert_size_stride(arg208_1, (1024, ), (1, ))
    assert_size_stride(arg209_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg210_1, (4096, ), (1, ))
    assert_size_stride(arg211_1, (1024, 4096), (4096, 1))
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
    assert_size_stride(arg241_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg242_1, (4096, ), (1, ))
    assert_size_stride(arg243_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (1024, ), (1, ))
    assert_size_stride(arg246_1, (1024, ), (1, ))
    assert_size_stride(arg247_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg248_1, (1024, ), (1, ))
    assert_size_stride(arg249_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg250_1, (1024, ), (1, ))
    assert_size_stride(arg251_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg252_1, (1024, ), (1, ))
    assert_size_stride(arg253_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg254_1, (1024, ), (1, ))
    assert_size_stride(arg255_1, (1024, ), (1, ))
    assert_size_stride(arg256_1, (1024, ), (1, ))
    assert_size_stride(arg257_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg258_1, (4096, ), (1, ))
    assert_size_stride(arg259_1, (1024, 4096), (4096, 1))
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
    assert_size_stride(arg289_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg290_1, (4096, ), (1, ))
    assert_size_stride(arg291_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg292_1, (1024, ), (1, ))
    assert_size_stride(arg293_1, (1024, ), (1, ))
    assert_size_stride(arg294_1, (1024, ), (1, ))
    assert_size_stride(arg295_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg296_1, (1024, ), (1, ))
    assert_size_stride(arg297_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg298_1, (1024, ), (1, ))
    assert_size_stride(arg299_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg300_1, (1024, ), (1, ))
    assert_size_stride(arg301_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (1024, ), (1, ))
    assert_size_stride(arg304_1, (1024, ), (1, ))
    assert_size_stride(arg305_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg306_1, (4096, ), (1, ))
    assert_size_stride(arg307_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg308_1, (1024, ), (1, ))
    assert_size_stride(arg309_1, (1024, ), (1, ))
    assert_size_stride(arg310_1, (1024, ), (1, ))
    assert_size_stride(arg311_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg312_1, (1024, ), (1, ))
    assert_size_stride(arg313_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg314_1, (1024, ), (1, ))
    assert_size_stride(arg315_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg318_1, (1024, ), (1, ))
    assert_size_stride(arg319_1, (1024, ), (1, ))
    assert_size_stride(arg320_1, (1024, ), (1, ))
    assert_size_stride(arg321_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg322_1, (4096, ), (1, ))
    assert_size_stride(arg323_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg324_1, (1024, ), (1, ))
    assert_size_stride(arg325_1, (1024, ), (1, ))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg328_1, (1024, ), (1, ))
    assert_size_stride(arg329_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg330_1, (1024, ), (1, ))
    assert_size_stride(arg331_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg332_1, (1024, ), (1, ))
    assert_size_stride(arg333_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg334_1, (1024, ), (1, ))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (1024, ), (1, ))
    assert_size_stride(arg337_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg338_1, (4096, ), (1, ))
    assert_size_stride(arg339_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg340_1, (1024, ), (1, ))
    assert_size_stride(arg341_1, (1024, ), (1, ))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg344_1, (1024, ), (1, ))
    assert_size_stride(arg345_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg348_1, (1024, ), (1, ))
    assert_size_stride(arg349_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg350_1, (1024, ), (1, ))
    assert_size_stride(arg351_1, (1024, ), (1, ))
    assert_size_stride(arg352_1, (1024, ), (1, ))
    assert_size_stride(arg353_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg354_1, (4096, ), (1, ))
    assert_size_stride(arg355_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg356_1, (1024, ), (1, ))
    assert_size_stride(arg357_1, (1024, ), (1, ))
    assert_size_stride(arg358_1, (1024, ), (1, ))
    assert_size_stride(arg359_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg364_1, (1024, ), (1, ))
    assert_size_stride(arg365_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg366_1, (1024, ), (1, ))
    assert_size_stride(arg367_1, (1024, ), (1, ))
    assert_size_stride(arg368_1, (1024, ), (1, ))
    assert_size_stride(arg369_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg370_1, (4096, ), (1, ))
    assert_size_stride(arg371_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg372_1, (1024, ), (1, ))
    assert_size_stride(arg373_1, (1024, ), (1, ))
    assert_size_stride(arg374_1, (1024, ), (1, ))
    assert_size_stride(arg375_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg376_1, (1024, ), (1, ))
    assert_size_stride(arg377_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg380_1, (1024, ), (1, ))
    assert_size_stride(arg381_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg382_1, (1024, ), (1, ))
    assert_size_stride(arg383_1, (1024, ), (1, ))
    assert_size_stride(arg384_1, (1024, ), (1, ))
    assert_size_stride(arg385_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg386_1, (4096, ), (1, ))
    assert_size_stride(arg387_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg388_1, (1024, ), (1, ))
    assert_size_stride(arg389_1, (1024, ), (1, ))
    assert_size_stride(arg390_1, (1024, ), (1, ))
    assert_size_stride(arg391_1, (1, 512), (512, 1))
    assert_size_stride(arg392_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg393_1, (1024, ), (1, ))
    assert_size_stride(arg394_1, (1024, ), (1, ))
    assert_size_stride(arg395_1, (1024, ), (1, ))
    assert_size_stride(arg396_1, (29056, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 512, 1024), (524288, 1024, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 512, 1024), (524288, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, token_type_ids, token_type_embeddings, embeddings, position_embeddings, embeddings_1, ln_outputs], Original ATen: [aten.embedding, aten.zeros, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_zeros_0.run(arg1_1, arg2_1, arg4_1, arg391_1, arg3_1, arg5_1, arg6_1, buf0, buf4, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg1_1
        del arg391_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        buf5 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf4, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg7_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf5)
        del arg7_1
        del arg8_1
        buf6 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf4, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg9_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf6)
        del arg10_1
        del arg9_1
        buf7 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf4, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg11_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf7)
        del arg11_1
        del arg12_1
        del buf4
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf5, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf6, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf7, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf9 = buf8[0]
        del buf8
        buf13 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf9, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg13_1, (1024, 1024), (1, 1024), 0), out=buf13)
        del arg13_1
        buf17 = reinterpret_tensor(buf9, (4, 512, 1024), (524288, 1024, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [attention_output, ln_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf0, buf13, arg14_1, arg15_1, arg16_1, buf17, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg15_1
        del arg16_1
        buf18 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf17, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg17_1, (1024, 4096), (1, 1024), 0), out=buf18)
        del arg17_1
        buf19 = reinterpret_tensor(buf18, (4, 512, 4096), (2097152, 4096, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf19, arg18_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg18_1
        buf20 = reinterpret_tensor(buf17, (2048, 1024), (1024, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg19_1, (4096, 1024), (1, 4096), 0), out=buf20)
        del arg19_1
        buf21 = reinterpret_tensor(buf20, (4, 512, 1024), (524288, 1024, 1), 0); del buf20  # reuse
        buf25 = reinterpret_tensor(buf6, (4, 512, 1024), (524288, 1024, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [attention_output, layer_output, ln_outputs_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf21, buf0, buf13, arg14_1, arg20_1, arg21_1, arg22_1, buf25, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg14_1
        del arg20_1
        del arg21_1
        del arg22_1
        buf26 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg24_1, reinterpret_tensor(buf25, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg23_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf26)
        del arg23_1
        del arg24_1
        buf27 = reinterpret_tensor(buf0, (2048, 1024), (1024, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg26_1, reinterpret_tensor(buf25, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg25_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf27)
        del arg25_1
        del arg26_1
        buf28 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg28_1, reinterpret_tensor(buf25, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg27_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf28)
        del arg27_1
        del arg28_1
        del buf25
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf29 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf26, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf27, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf28, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf30 = buf29[0]
        del buf29
        buf34 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg29_1, (1024, 1024), (1, 1024), 0), out=buf34)
        del arg29_1
        buf38 = reinterpret_tensor(buf30, (4, 512, 1024), (524288, 1024, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [attention_output_1, ln_output_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf21, buf34, arg30_1, arg31_1, arg32_1, buf38, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg31_1
        del arg32_1
        buf39 = reinterpret_tensor(buf19, (2048, 4096), (4096, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg33_1, (1024, 4096), (1, 1024), 0), out=buf39)
        del arg33_1
        buf40 = reinterpret_tensor(buf39, (4, 512, 4096), (2097152, 4096, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf40, arg34_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg34_1
        buf41 = reinterpret_tensor(buf38, (2048, 1024), (1024, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg35_1, (4096, 1024), (1, 4096), 0), out=buf41)
        del arg35_1
        buf42 = reinterpret_tensor(buf41, (4, 512, 1024), (524288, 1024, 1), 0); del buf41  # reuse
        buf46 = reinterpret_tensor(buf27, (4, 512, 1024), (524288, 1024, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [attention_output_1, layer_output_1, ln_outputs_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf42, buf21, buf34, arg30_1, arg36_1, arg37_1, arg38_1, buf46, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg30_1
        del arg36_1
        del arg37_1
        del arg38_1
        buf47 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg40_1, reinterpret_tensor(buf46, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg39_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf47)
        del arg39_1
        del arg40_1
        buf48 = reinterpret_tensor(buf21, (2048, 1024), (1024, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg42_1, reinterpret_tensor(buf46, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg41_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf48)
        del arg41_1
        del arg42_1
        buf49 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg44_1, reinterpret_tensor(buf46, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg43_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf49)
        del arg43_1
        del arg44_1
        del buf46
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf50 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf47, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf48, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf49, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf51 = buf50[0]
        del buf50
        buf55 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg45_1, (1024, 1024), (1, 1024), 0), out=buf55)
        del arg45_1
        buf59 = reinterpret_tensor(buf51, (4, 512, 1024), (524288, 1024, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [attention_output_2, ln_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf42, buf55, arg46_1, arg47_1, arg48_1, buf59, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg47_1
        del arg48_1
        buf60 = reinterpret_tensor(buf40, (2048, 4096), (4096, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg49_1, (1024, 4096), (1, 1024), 0), out=buf60)
        del arg49_1
        buf61 = reinterpret_tensor(buf60, (4, 512, 4096), (2097152, 4096, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf61, arg50_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg50_1
        buf62 = reinterpret_tensor(buf59, (2048, 1024), (1024, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg51_1, (4096, 1024), (1, 4096), 0), out=buf62)
        del arg51_1
        buf63 = reinterpret_tensor(buf62, (4, 512, 1024), (524288, 1024, 1), 0); del buf62  # reuse
        buf67 = reinterpret_tensor(buf48, (4, 512, 1024), (524288, 1024, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [attention_output_2, layer_output_2, ln_outputs_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf63, buf42, buf55, arg46_1, arg52_1, arg53_1, arg54_1, buf67, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg46_1
        del arg52_1
        del arg53_1
        del arg54_1
        buf68 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg56_1, reinterpret_tensor(buf67, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg55_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf68)
        del arg55_1
        del arg56_1
        buf69 = reinterpret_tensor(buf42, (2048, 1024), (1024, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg58_1, reinterpret_tensor(buf67, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf69)
        del arg57_1
        del arg58_1
        buf70 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg60_1, reinterpret_tensor(buf67, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg59_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf70)
        del arg59_1
        del arg60_1
        del buf67
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf71 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf68, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf69, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf70, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf72 = buf71[0]
        del buf71
        buf76 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg61_1, (1024, 1024), (1, 1024), 0), out=buf76)
        del arg61_1
        buf80 = reinterpret_tensor(buf72, (4, 512, 1024), (524288, 1024, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [attention_output_3, ln_output_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf63, buf76, arg62_1, arg63_1, arg64_1, buf80, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg63_1
        del arg64_1
        buf81 = reinterpret_tensor(buf61, (2048, 4096), (4096, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg65_1, (1024, 4096), (1, 1024), 0), out=buf81)
        del arg65_1
        buf82 = reinterpret_tensor(buf81, (4, 512, 4096), (2097152, 4096, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf82, arg66_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg66_1
        buf83 = reinterpret_tensor(buf80, (2048, 1024), (1024, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg67_1, (4096, 1024), (1, 4096), 0), out=buf83)
        del arg67_1
        buf84 = reinterpret_tensor(buf83, (4, 512, 1024), (524288, 1024, 1), 0); del buf83  # reuse
        buf88 = reinterpret_tensor(buf69, (4, 512, 1024), (524288, 1024, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [attention_output_3, layer_output_3, ln_outputs_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf84, buf63, buf76, arg62_1, arg68_1, arg69_1, arg70_1, buf88, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg62_1
        del arg68_1
        del arg69_1
        del arg70_1
        buf89 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg72_1, reinterpret_tensor(buf88, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf89)
        del arg71_1
        del arg72_1
        buf90 = reinterpret_tensor(buf63, (2048, 1024), (1024, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg74_1, reinterpret_tensor(buf88, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf90)
        del arg73_1
        del arg74_1
        buf91 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg76_1, reinterpret_tensor(buf88, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg75_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf91)
        del arg75_1
        del arg76_1
        del buf88
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf92 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf89, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf90, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf91, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf93 = buf92[0]
        del buf92
        buf97 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg77_1, (1024, 1024), (1, 1024), 0), out=buf97)
        del arg77_1
        buf101 = reinterpret_tensor(buf93, (4, 512, 1024), (524288, 1024, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [attention_output_4, ln_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf84, buf97, arg78_1, arg79_1, arg80_1, buf101, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg79_1
        del arg80_1
        buf102 = reinterpret_tensor(buf82, (2048, 4096), (4096, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg81_1, (1024, 4096), (1, 1024), 0), out=buf102)
        del arg81_1
        buf103 = reinterpret_tensor(buf102, (4, 512, 4096), (2097152, 4096, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_27], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf103, arg82_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg82_1
        buf104 = reinterpret_tensor(buf101, (2048, 1024), (1024, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg83_1, (4096, 1024), (1, 4096), 0), out=buf104)
        del arg83_1
        buf105 = reinterpret_tensor(buf104, (4, 512, 1024), (524288, 1024, 1), 0); del buf104  # reuse
        buf109 = reinterpret_tensor(buf90, (4, 512, 1024), (524288, 1024, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [attention_output_4, layer_output_4, ln_outputs_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf105, buf84, buf97, arg78_1, arg84_1, arg85_1, arg86_1, buf109, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg78_1
        del arg84_1
        del arg85_1
        del arg86_1
        buf110 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg88_1, reinterpret_tensor(buf109, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg87_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf110)
        del arg87_1
        del arg88_1
        buf111 = reinterpret_tensor(buf84, (2048, 1024), (1024, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg90_1, reinterpret_tensor(buf109, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg89_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf111)
        del arg89_1
        del arg90_1
        buf112 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg92_1, reinterpret_tensor(buf109, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg91_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf112)
        del arg91_1
        del arg92_1
        del buf109
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf113 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf110, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf111, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf112, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf114 = buf113[0]
        del buf113
        buf118 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg93_1, (1024, 1024), (1, 1024), 0), out=buf118)
        del arg93_1
        buf122 = reinterpret_tensor(buf114, (4, 512, 1024), (524288, 1024, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [attention_output_5, ln_output_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf105, buf118, arg94_1, arg95_1, arg96_1, buf122, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg95_1
        del arg96_1
        buf123 = reinterpret_tensor(buf103, (2048, 4096), (4096, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg97_1, (1024, 4096), (1, 1024), 0), out=buf123)
        del arg97_1
        buf124 = reinterpret_tensor(buf123, (4, 512, 4096), (2097152, 4096, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_33], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf124, arg98_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg98_1
        buf125 = reinterpret_tensor(buf122, (2048, 1024), (1024, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg99_1, (4096, 1024), (1, 4096), 0), out=buf125)
        del arg99_1
        buf126 = reinterpret_tensor(buf125, (4, 512, 1024), (524288, 1024, 1), 0); del buf125  # reuse
        buf130 = reinterpret_tensor(buf111, (4, 512, 1024), (524288, 1024, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [attention_output_5, layer_output_5, ln_outputs_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf126, buf105, buf118, arg94_1, arg100_1, arg101_1, arg102_1, buf130, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        del arg94_1
        buf131 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg104_1, reinterpret_tensor(buf130, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg103_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf131)
        del arg103_1
        del arg104_1
        buf132 = reinterpret_tensor(buf105, (2048, 1024), (1024, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg106_1, reinterpret_tensor(buf130, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg105_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf132)
        del arg105_1
        del arg106_1
        buf133 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg108_1, reinterpret_tensor(buf130, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg107_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf133)
        del arg107_1
        del arg108_1
        del buf130
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf134 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf131, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf132, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf133, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf135 = buf134[0]
        del buf134
        buf139 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg109_1, (1024, 1024), (1, 1024), 0), out=buf139)
        del arg109_1
        buf143 = reinterpret_tensor(buf135, (4, 512, 1024), (524288, 1024, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [attention_output_6, ln_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf126, buf139, arg110_1, arg111_1, arg112_1, buf143, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg111_1
        del arg112_1
        buf144 = reinterpret_tensor(buf124, (2048, 4096), (4096, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg113_1, (1024, 4096), (1, 1024), 0), out=buf144)
        del arg113_1
        buf145 = reinterpret_tensor(buf144, (4, 512, 4096), (2097152, 4096, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf145, arg114_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg114_1
        buf146 = reinterpret_tensor(buf143, (2048, 1024), (1024, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg115_1, (4096, 1024), (1, 4096), 0), out=buf146)
        del arg115_1
        buf147 = reinterpret_tensor(buf146, (4, 512, 1024), (524288, 1024, 1), 0); del buf146  # reuse
        buf151 = reinterpret_tensor(buf132, (4, 512, 1024), (524288, 1024, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [attention_output_6, layer_output_6, ln_outputs_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf147, buf126, buf139, arg110_1, arg116_1, arg117_1, arg118_1, buf151, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg110_1
        del arg116_1
        del arg117_1
        del arg118_1
        buf152 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg120_1, reinterpret_tensor(buf151, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg119_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf152)
        del arg119_1
        del arg120_1
        buf153 = reinterpret_tensor(buf126, (2048, 1024), (1024, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg122_1, reinterpret_tensor(buf151, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg121_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf153)
        del arg121_1
        del arg122_1
        buf154 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg124_1, reinterpret_tensor(buf151, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg123_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf154)
        del arg123_1
        del arg124_1
        del buf151
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf155 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf152, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf153, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf154, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf156 = buf155[0]
        del buf155
        buf160 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg125_1, (1024, 1024), (1, 1024), 0), out=buf160)
        del arg125_1
        buf164 = reinterpret_tensor(buf156, (4, 512, 1024), (524288, 1024, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [attention_output_7, ln_output_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf147, buf160, arg126_1, arg127_1, arg128_1, buf164, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg127_1
        del arg128_1
        buf165 = reinterpret_tensor(buf145, (2048, 4096), (4096, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg129_1, (1024, 4096), (1, 1024), 0), out=buf165)
        del arg129_1
        buf166 = reinterpret_tensor(buf165, (4, 512, 4096), (2097152, 4096, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf166, arg130_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg130_1
        buf167 = reinterpret_tensor(buf164, (2048, 1024), (1024, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg131_1, (4096, 1024), (1, 4096), 0), out=buf167)
        del arg131_1
        buf168 = reinterpret_tensor(buf167, (4, 512, 1024), (524288, 1024, 1), 0); del buf167  # reuse
        buf172 = reinterpret_tensor(buf153, (4, 512, 1024), (524288, 1024, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [attention_output_7, layer_output_7, ln_outputs_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf168, buf147, buf160, arg126_1, arg132_1, arg133_1, arg134_1, buf172, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg126_1
        del arg132_1
        del arg133_1
        del arg134_1
        buf173 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg136_1, reinterpret_tensor(buf172, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg135_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf173)
        del arg135_1
        del arg136_1
        buf174 = reinterpret_tensor(buf147, (2048, 1024), (1024, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg138_1, reinterpret_tensor(buf172, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf174)
        del arg137_1
        del arg138_1
        buf175 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg140_1, reinterpret_tensor(buf172, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg139_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf175)
        del arg139_1
        del arg140_1
        del buf172
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf176 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf173, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf174, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf175, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf177 = buf176[0]
        del buf176
        buf181 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf177, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg141_1, (1024, 1024), (1, 1024), 0), out=buf181)
        del arg141_1
        buf185 = reinterpret_tensor(buf177, (4, 512, 1024), (524288, 1024, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [attention_output_8, ln_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf168, buf181, arg142_1, arg143_1, arg144_1, buf185, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg143_1
        del arg144_1
        buf186 = reinterpret_tensor(buf166, (2048, 4096), (4096, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf185, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg145_1, (1024, 4096), (1, 1024), 0), out=buf186)
        del arg145_1
        buf187 = reinterpret_tensor(buf186, (4, 512, 4096), (2097152, 4096, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf187, arg146_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg146_1
        buf188 = reinterpret_tensor(buf185, (2048, 1024), (1024, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg147_1, (4096, 1024), (1, 4096), 0), out=buf188)
        del arg147_1
        buf189 = reinterpret_tensor(buf188, (4, 512, 1024), (524288, 1024, 1), 0); del buf188  # reuse
        buf193 = reinterpret_tensor(buf174, (4, 512, 1024), (524288, 1024, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [attention_output_8, layer_output_8, ln_outputs_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf189, buf168, buf181, arg142_1, arg148_1, arg149_1, arg150_1, buf193, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg142_1
        del arg148_1
        del arg149_1
        del arg150_1
        buf194 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg152_1, reinterpret_tensor(buf193, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg151_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf194)
        del arg151_1
        del arg152_1
        buf195 = reinterpret_tensor(buf168, (2048, 1024), (1024, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg154_1, reinterpret_tensor(buf193, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg153_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf195)
        del arg153_1
        del arg154_1
        buf196 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg156_1, reinterpret_tensor(buf193, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg155_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf196)
        del arg155_1
        del arg156_1
        del buf193
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf197 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf194, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf195, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf196, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf198 = buf197[0]
        del buf197
        buf202 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg157_1, (1024, 1024), (1, 1024), 0), out=buf202)
        del arg157_1
        buf206 = reinterpret_tensor(buf198, (4, 512, 1024), (524288, 1024, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [attention_output_9, ln_output_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf189, buf202, arg158_1, arg159_1, arg160_1, buf206, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg159_1
        del arg160_1
        buf207 = reinterpret_tensor(buf187, (2048, 4096), (4096, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg161_1, (1024, 4096), (1, 1024), 0), out=buf207)
        del arg161_1
        buf208 = reinterpret_tensor(buf207, (4, 512, 4096), (2097152, 4096, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_57], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf208, arg162_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg162_1
        buf209 = reinterpret_tensor(buf206, (2048, 1024), (1024, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf208, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg163_1, (4096, 1024), (1, 4096), 0), out=buf209)
        del arg163_1
        buf210 = reinterpret_tensor(buf209, (4, 512, 1024), (524288, 1024, 1), 0); del buf209  # reuse
        buf214 = reinterpret_tensor(buf195, (4, 512, 1024), (524288, 1024, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [attention_output_9, layer_output_9, ln_outputs_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf210, buf189, buf202, arg158_1, arg164_1, arg165_1, arg166_1, buf214, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg158_1
        del arg164_1
        del arg165_1
        del arg166_1
        buf215 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg168_1, reinterpret_tensor(buf214, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg167_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf215)
        del arg167_1
        del arg168_1
        buf216 = reinterpret_tensor(buf189, (2048, 1024), (1024, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg170_1, reinterpret_tensor(buf214, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg169_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf216)
        del arg169_1
        del arg170_1
        buf217 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg172_1, reinterpret_tensor(buf214, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf217)
        del arg171_1
        del arg172_1
        del buf214
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf218 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf215, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf216, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf217, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf219 = buf218[0]
        del buf218
        buf223 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg173_1, (1024, 1024), (1, 1024), 0), out=buf223)
        del arg173_1
        buf227 = reinterpret_tensor(buf219, (4, 512, 1024), (524288, 1024, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [attention_output_10, ln_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf210, buf223, arg174_1, arg175_1, arg176_1, buf227, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg175_1
        del arg176_1
        buf228 = reinterpret_tensor(buf208, (2048, 4096), (4096, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf227, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg177_1, (1024, 4096), (1, 1024), 0), out=buf228)
        del arg177_1
        buf229 = reinterpret_tensor(buf228, (4, 512, 4096), (2097152, 4096, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_63], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf229, arg178_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg178_1
        buf230 = reinterpret_tensor(buf227, (2048, 1024), (1024, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg179_1, (4096, 1024), (1, 4096), 0), out=buf230)
        del arg179_1
        buf231 = reinterpret_tensor(buf230, (4, 512, 1024), (524288, 1024, 1), 0); del buf230  # reuse
        buf235 = reinterpret_tensor(buf216, (4, 512, 1024), (524288, 1024, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [attention_output_10, layer_output_10, ln_outputs_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf231, buf210, buf223, arg174_1, arg180_1, arg181_1, arg182_1, buf235, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg174_1
        del arg180_1
        del arg181_1
        del arg182_1
        buf236 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg184_1, reinterpret_tensor(buf235, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg183_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf236)
        del arg183_1
        del arg184_1
        buf237 = reinterpret_tensor(buf210, (2048, 1024), (1024, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg186_1, reinterpret_tensor(buf235, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg185_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf237)
        del arg185_1
        del arg186_1
        buf238 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg188_1, reinterpret_tensor(buf235, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg187_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf238)
        del arg187_1
        del arg188_1
        del buf235
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf239 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf236, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf237, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf238, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf240 = buf239[0]
        del buf239
        buf244 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg189_1, (1024, 1024), (1, 1024), 0), out=buf244)
        del arg189_1
        buf248 = reinterpret_tensor(buf240, (4, 512, 1024), (524288, 1024, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [attention_output_11, ln_output_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf231, buf244, arg190_1, arg191_1, arg192_1, buf248, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg191_1
        del arg192_1
        buf249 = reinterpret_tensor(buf229, (2048, 4096), (4096, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg193_1, (1024, 4096), (1, 1024), 0), out=buf249)
        del arg193_1
        buf250 = reinterpret_tensor(buf249, (4, 512, 4096), (2097152, 4096, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf250, arg194_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg194_1
        buf251 = reinterpret_tensor(buf248, (2048, 1024), (1024, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg195_1, (4096, 1024), (1, 4096), 0), out=buf251)
        del arg195_1
        buf252 = reinterpret_tensor(buf251, (4, 512, 1024), (524288, 1024, 1), 0); del buf251  # reuse
        buf256 = reinterpret_tensor(buf237, (4, 512, 1024), (524288, 1024, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [attention_output_11, layer_output_11, ln_outputs_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf252, buf231, buf244, arg190_1, arg196_1, arg197_1, arg198_1, buf256, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg190_1
        del arg196_1
        del arg197_1
        del arg198_1
        buf257 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg200_1, reinterpret_tensor(buf256, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg199_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf257)
        del arg199_1
        del arg200_1
        buf258 = reinterpret_tensor(buf231, (2048, 1024), (1024, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [linear_73], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg202_1, reinterpret_tensor(buf256, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg201_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf258)
        del arg201_1
        del arg202_1
        buf259 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [linear_74], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg204_1, reinterpret_tensor(buf256, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg203_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf259)
        del arg203_1
        del arg204_1
        del buf256
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf260 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf257, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf258, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf259, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf261 = buf260[0]
        del buf260
        buf265 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf261, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg205_1, (1024, 1024), (1, 1024), 0), out=buf265)
        del arg205_1
        buf269 = reinterpret_tensor(buf261, (4, 512, 1024), (524288, 1024, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [attention_output_12, ln_output_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf252, buf265, arg206_1, arg207_1, arg208_1, buf269, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg207_1
        del arg208_1
        buf270 = reinterpret_tensor(buf250, (2048, 4096), (4096, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg209_1, (1024, 4096), (1, 1024), 0), out=buf270)
        del arg209_1
        buf271 = reinterpret_tensor(buf270, (4, 512, 4096), (2097152, 4096, 1), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_75], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf271, arg210_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg210_1
        buf272 = reinterpret_tensor(buf269, (2048, 1024), (1024, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg211_1, (4096, 1024), (1, 4096), 0), out=buf272)
        del arg211_1
        buf273 = reinterpret_tensor(buf272, (4, 512, 1024), (524288, 1024, 1), 0); del buf272  # reuse
        buf277 = reinterpret_tensor(buf258, (4, 512, 1024), (524288, 1024, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [attention_output_12, layer_output_12, ln_outputs_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf273, buf252, buf265, arg206_1, arg212_1, arg213_1, arg214_1, buf277, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg206_1
        del arg212_1
        del arg213_1
        del arg214_1
        buf278 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg216_1, reinterpret_tensor(buf277, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg215_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf278)
        del arg215_1
        del arg216_1
        buf279 = reinterpret_tensor(buf252, (2048, 1024), (1024, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg218_1, reinterpret_tensor(buf277, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg217_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf279)
        del arg217_1
        del arg218_1
        buf280 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [linear_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg220_1, reinterpret_tensor(buf277, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg219_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf280)
        del arg219_1
        del arg220_1
        del buf277
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf281 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf278, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf279, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf280, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf282 = buf281[0]
        del buf281
        buf286 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg221_1, (1024, 1024), (1, 1024), 0), out=buf286)
        del arg221_1
        buf290 = reinterpret_tensor(buf282, (4, 512, 1024), (524288, 1024, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [attention_output_13, ln_output_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf273, buf286, arg222_1, arg223_1, arg224_1, buf290, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg223_1
        del arg224_1
        buf291 = reinterpret_tensor(buf271, (2048, 4096), (4096, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf290, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg225_1, (1024, 4096), (1, 1024), 0), out=buf291)
        del arg225_1
        buf292 = reinterpret_tensor(buf291, (4, 512, 4096), (2097152, 4096, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_81], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf292, arg226_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg226_1
        buf293 = reinterpret_tensor(buf290, (2048, 1024), (1024, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg227_1, (4096, 1024), (1, 4096), 0), out=buf293)
        del arg227_1
        buf294 = reinterpret_tensor(buf293, (4, 512, 1024), (524288, 1024, 1), 0); del buf293  # reuse
        buf298 = reinterpret_tensor(buf279, (4, 512, 1024), (524288, 1024, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [attention_output_13, layer_output_13, ln_outputs_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf294, buf273, buf286, arg222_1, arg228_1, arg229_1, arg230_1, buf298, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg222_1
        del arg228_1
        del arg229_1
        del arg230_1
        buf299 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg232_1, reinterpret_tensor(buf298, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg231_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf299)
        del arg231_1
        del arg232_1
        buf300 = reinterpret_tensor(buf273, (2048, 1024), (1024, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg234_1, reinterpret_tensor(buf298, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg233_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf300)
        del arg233_1
        del arg234_1
        buf301 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg236_1, reinterpret_tensor(buf298, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg235_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf301)
        del arg235_1
        del arg236_1
        del buf298
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf302 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf299, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf300, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf301, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf303 = buf302[0]
        del buf302
        buf307 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg237_1, (1024, 1024), (1, 1024), 0), out=buf307)
        del arg237_1
        buf311 = reinterpret_tensor(buf303, (4, 512, 1024), (524288, 1024, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [attention_output_14, ln_output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf294, buf307, arg238_1, arg239_1, arg240_1, buf311, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg239_1
        del arg240_1
        buf312 = reinterpret_tensor(buf292, (2048, 4096), (4096, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf311, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg241_1, (1024, 4096), (1, 1024), 0), out=buf312)
        del arg241_1
        buf313 = reinterpret_tensor(buf312, (4, 512, 4096), (2097152, 4096, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf313, arg242_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg242_1
        buf314 = reinterpret_tensor(buf311, (2048, 1024), (1024, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg243_1, (4096, 1024), (1, 4096), 0), out=buf314)
        del arg243_1
        buf315 = reinterpret_tensor(buf314, (4, 512, 1024), (524288, 1024, 1), 0); del buf314  # reuse
        buf319 = reinterpret_tensor(buf300, (4, 512, 1024), (524288, 1024, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [attention_output_14, layer_output_14, ln_outputs_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf315, buf294, buf307, arg238_1, arg244_1, arg245_1, arg246_1, buf319, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg238_1
        del arg244_1
        del arg245_1
        del arg246_1
        buf320 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg248_1, reinterpret_tensor(buf319, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg247_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf320)
        del arg247_1
        del arg248_1
        buf321 = reinterpret_tensor(buf294, (2048, 1024), (1024, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg250_1, reinterpret_tensor(buf319, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg249_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf321)
        del arg249_1
        del arg250_1
        buf322 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [linear_92], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg252_1, reinterpret_tensor(buf319, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg251_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf322)
        del arg251_1
        del arg252_1
        del buf319
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf323 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf320, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf321, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf322, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf324 = buf323[0]
        del buf323
        buf328 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg253_1, (1024, 1024), (1, 1024), 0), out=buf328)
        del arg253_1
        buf332 = reinterpret_tensor(buf324, (4, 512, 1024), (524288, 1024, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [attention_output_15, ln_output_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf315, buf328, arg254_1, arg255_1, arg256_1, buf332, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg255_1
        del arg256_1
        buf333 = reinterpret_tensor(buf313, (2048, 4096), (4096, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg257_1, (1024, 4096), (1, 1024), 0), out=buf333)
        del arg257_1
        buf334 = reinterpret_tensor(buf333, (4, 512, 4096), (2097152, 4096, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf334, arg258_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg258_1
        buf335 = reinterpret_tensor(buf332, (2048, 1024), (1024, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg259_1, (4096, 1024), (1, 4096), 0), out=buf335)
        del arg259_1
        buf336 = reinterpret_tensor(buf335, (4, 512, 1024), (524288, 1024, 1), 0); del buf335  # reuse
        buf340 = reinterpret_tensor(buf321, (4, 512, 1024), (524288, 1024, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [attention_output_15, layer_output_15, ln_outputs_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf336, buf315, buf328, arg254_1, arg260_1, arg261_1, arg262_1, buf340, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg254_1
        del arg260_1
        del arg261_1
        del arg262_1
        buf341 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg264_1, reinterpret_tensor(buf340, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg263_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf341)
        del arg263_1
        del arg264_1
        buf342 = reinterpret_tensor(buf315, (2048, 1024), (1024, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [linear_97], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg266_1, reinterpret_tensor(buf340, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg265_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf342)
        del arg265_1
        del arg266_1
        buf343 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [linear_98], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg268_1, reinterpret_tensor(buf340, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg267_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf343)
        del arg267_1
        del arg268_1
        del buf340
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf344 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf341, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf342, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf343, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf345 = buf344[0]
        del buf344
        buf349 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf345, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg269_1, (1024, 1024), (1, 1024), 0), out=buf349)
        del arg269_1
        buf353 = reinterpret_tensor(buf345, (4, 512, 1024), (524288, 1024, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [attention_output_16, ln_output_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf336, buf349, arg270_1, arg271_1, arg272_1, buf353, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg271_1
        del arg272_1
        buf354 = reinterpret_tensor(buf334, (2048, 4096), (4096, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg273_1, (1024, 4096), (1, 1024), 0), out=buf354)
        del arg273_1
        buf355 = reinterpret_tensor(buf354, (4, 512, 4096), (2097152, 4096, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_99], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf355, arg274_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg274_1
        buf356 = reinterpret_tensor(buf353, (2048, 1024), (1024, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf355, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg275_1, (4096, 1024), (1, 4096), 0), out=buf356)
        del arg275_1
        buf357 = reinterpret_tensor(buf356, (4, 512, 1024), (524288, 1024, 1), 0); del buf356  # reuse
        buf361 = reinterpret_tensor(buf342, (4, 512, 1024), (524288, 1024, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [attention_output_16, layer_output_16, ln_outputs_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf357, buf336, buf349, arg270_1, arg276_1, arg277_1, arg278_1, buf361, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg270_1
        del arg276_1
        del arg277_1
        del arg278_1
        buf362 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg280_1, reinterpret_tensor(buf361, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg279_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf362)
        del arg279_1
        del arg280_1
        buf363 = reinterpret_tensor(buf336, (2048, 1024), (1024, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [linear_103], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg282_1, reinterpret_tensor(buf361, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg281_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf363)
        del arg281_1
        del arg282_1
        buf364 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [linear_104], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg284_1, reinterpret_tensor(buf361, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg283_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf364)
        del arg283_1
        del arg284_1
        del buf361
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf365 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf362, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf363, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf364, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf366 = buf365[0]
        del buf365
        buf370 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf366, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg285_1, (1024, 1024), (1, 1024), 0), out=buf370)
        del arg285_1
        buf374 = reinterpret_tensor(buf366, (4, 512, 1024), (524288, 1024, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [attention_output_17, ln_output_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf357, buf370, arg286_1, arg287_1, arg288_1, buf374, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg287_1
        del arg288_1
        buf375 = reinterpret_tensor(buf355, (2048, 4096), (4096, 1), 0); del buf355  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf374, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg289_1, (1024, 4096), (1, 1024), 0), out=buf375)
        del arg289_1
        buf376 = reinterpret_tensor(buf375, (4, 512, 4096), (2097152, 4096, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_105], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf376, arg290_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg290_1
        buf377 = reinterpret_tensor(buf374, (2048, 1024), (1024, 1), 0); del buf374  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf376, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg291_1, (4096, 1024), (1, 4096), 0), out=buf377)
        del arg291_1
        buf378 = reinterpret_tensor(buf377, (4, 512, 1024), (524288, 1024, 1), 0); del buf377  # reuse
        buf382 = reinterpret_tensor(buf363, (4, 512, 1024), (524288, 1024, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [attention_output_17, layer_output_17, ln_outputs_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf378, buf357, buf370, arg286_1, arg292_1, arg293_1, arg294_1, buf382, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg286_1
        del arg292_1
        del arg293_1
        del arg294_1
        buf383 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg296_1, reinterpret_tensor(buf382, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg295_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf383)
        del arg295_1
        del arg296_1
        buf384 = reinterpret_tensor(buf357, (2048, 1024), (1024, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [linear_109], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg298_1, reinterpret_tensor(buf382, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg297_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf384)
        del arg297_1
        del arg298_1
        buf385 = buf362; del buf362  # reuse
        # Topologically Sorted Source Nodes: [linear_110], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg300_1, reinterpret_tensor(buf382, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg299_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf385)
        del arg299_1
        del arg300_1
        del buf382
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf386 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf383, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf384, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf385, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf387 = buf386[0]
        del buf386
        buf391 = buf385; del buf385  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf387, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg301_1, (1024, 1024), (1, 1024), 0), out=buf391)
        del arg301_1
        buf395 = reinterpret_tensor(buf387, (4, 512, 1024), (524288, 1024, 1), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [attention_output_18, ln_output_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf378, buf391, arg302_1, arg303_1, arg304_1, buf395, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg303_1
        del arg304_1
        buf396 = reinterpret_tensor(buf376, (2048, 4096), (4096, 1), 0); del buf376  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf395, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg305_1, (1024, 4096), (1, 1024), 0), out=buf396)
        del arg305_1
        buf397 = reinterpret_tensor(buf396, (4, 512, 4096), (2097152, 4096, 1), 0); del buf396  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf397, arg306_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg306_1
        buf398 = reinterpret_tensor(buf395, (2048, 1024), (1024, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf397, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg307_1, (4096, 1024), (1, 4096), 0), out=buf398)
        del arg307_1
        buf399 = reinterpret_tensor(buf398, (4, 512, 1024), (524288, 1024, 1), 0); del buf398  # reuse
        buf403 = reinterpret_tensor(buf384, (4, 512, 1024), (524288, 1024, 1), 0); del buf384  # reuse
        # Topologically Sorted Source Nodes: [attention_output_18, layer_output_18, ln_outputs_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf399, buf378, buf391, arg302_1, arg308_1, arg309_1, arg310_1, buf403, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg302_1
        del arg308_1
        del arg309_1
        del arg310_1
        buf404 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg312_1, reinterpret_tensor(buf403, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg311_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf404)
        del arg311_1
        del arg312_1
        buf405 = reinterpret_tensor(buf378, (2048, 1024), (1024, 1), 0); del buf378  # reuse
        # Topologically Sorted Source Nodes: [linear_115], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg314_1, reinterpret_tensor(buf403, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg313_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf405)
        del arg313_1
        del arg314_1
        buf406 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [linear_116], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg316_1, reinterpret_tensor(buf403, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg315_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf406)
        del arg315_1
        del arg316_1
        del buf403
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf407 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf404, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf405, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf406, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf408 = buf407[0]
        del buf407
        buf412 = buf406; del buf406  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf408, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg317_1, (1024, 1024), (1, 1024), 0), out=buf412)
        del arg317_1
        buf416 = reinterpret_tensor(buf408, (4, 512, 1024), (524288, 1024, 1), 0); del buf408  # reuse
        # Topologically Sorted Source Nodes: [attention_output_19, ln_output_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf399, buf412, arg318_1, arg319_1, arg320_1, buf416, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg319_1
        del arg320_1
        buf417 = reinterpret_tensor(buf397, (2048, 4096), (4096, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf416, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg321_1, (1024, 4096), (1, 1024), 0), out=buf417)
        del arg321_1
        buf418 = reinterpret_tensor(buf417, (4, 512, 4096), (2097152, 4096, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_117], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf418, arg322_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg322_1
        buf419 = reinterpret_tensor(buf416, (2048, 1024), (1024, 1), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf418, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg323_1, (4096, 1024), (1, 4096), 0), out=buf419)
        del arg323_1
        buf420 = reinterpret_tensor(buf419, (4, 512, 1024), (524288, 1024, 1), 0); del buf419  # reuse
        buf424 = reinterpret_tensor(buf405, (4, 512, 1024), (524288, 1024, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [attention_output_19, layer_output_19, ln_outputs_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf420, buf399, buf412, arg318_1, arg324_1, arg325_1, arg326_1, buf424, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg318_1
        del arg324_1
        del arg325_1
        del arg326_1
        buf425 = buf412; del buf412  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg328_1, reinterpret_tensor(buf424, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg327_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf425)
        del arg327_1
        del arg328_1
        buf426 = reinterpret_tensor(buf399, (2048, 1024), (1024, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [linear_121], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg330_1, reinterpret_tensor(buf424, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg329_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf426)
        del arg329_1
        del arg330_1
        buf427 = buf404; del buf404  # reuse
        # Topologically Sorted Source Nodes: [linear_122], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg332_1, reinterpret_tensor(buf424, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg331_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf427)
        del arg331_1
        del arg332_1
        del buf424
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf428 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf425, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf426, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf427, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf429 = buf428[0]
        del buf428
        buf433 = buf427; del buf427  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf429, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg333_1, (1024, 1024), (1, 1024), 0), out=buf433)
        del arg333_1
        buf437 = reinterpret_tensor(buf429, (4, 512, 1024), (524288, 1024, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [attention_output_20, ln_output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf420, buf433, arg334_1, arg335_1, arg336_1, buf437, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg335_1
        del arg336_1
        buf438 = reinterpret_tensor(buf418, (2048, 4096), (4096, 1), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf437, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg337_1, (1024, 4096), (1, 1024), 0), out=buf438)
        del arg337_1
        buf439 = reinterpret_tensor(buf438, (4, 512, 4096), (2097152, 4096, 1), 0); del buf438  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_123], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf439, arg338_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg338_1
        buf440 = reinterpret_tensor(buf437, (2048, 1024), (1024, 1), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf439, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg339_1, (4096, 1024), (1, 4096), 0), out=buf440)
        del arg339_1
        buf441 = reinterpret_tensor(buf440, (4, 512, 1024), (524288, 1024, 1), 0); del buf440  # reuse
        buf445 = reinterpret_tensor(buf426, (4, 512, 1024), (524288, 1024, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [attention_output_20, layer_output_20, ln_outputs_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf441, buf420, buf433, arg334_1, arg340_1, arg341_1, arg342_1, buf445, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg334_1
        del arg340_1
        del arg341_1
        del arg342_1
        buf446 = buf433; del buf433  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg344_1, reinterpret_tensor(buf445, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg343_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf446)
        del arg343_1
        del arg344_1
        buf447 = reinterpret_tensor(buf420, (2048, 1024), (1024, 1), 0); del buf420  # reuse
        # Topologically Sorted Source Nodes: [linear_127], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg346_1, reinterpret_tensor(buf445, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg345_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf447)
        del arg345_1
        del arg346_1
        buf448 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [linear_128], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg348_1, reinterpret_tensor(buf445, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg347_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf448)
        del arg347_1
        del arg348_1
        del buf445
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf449 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf446, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf447, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf448, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf450 = buf449[0]
        del buf449
        buf454 = buf448; del buf448  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf450, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg349_1, (1024, 1024), (1, 1024), 0), out=buf454)
        del arg349_1
        buf458 = reinterpret_tensor(buf450, (4, 512, 1024), (524288, 1024, 1), 0); del buf450  # reuse
        # Topologically Sorted Source Nodes: [attention_output_21, ln_output_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf441, buf454, arg350_1, arg351_1, arg352_1, buf458, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg351_1
        del arg352_1
        buf459 = reinterpret_tensor(buf439, (2048, 4096), (4096, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf458, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg353_1, (1024, 4096), (1, 1024), 0), out=buf459)
        del arg353_1
        buf460 = reinterpret_tensor(buf459, (4, 512, 4096), (2097152, 4096, 1), 0); del buf459  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_129], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf460, arg354_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg354_1
        buf461 = reinterpret_tensor(buf458, (2048, 1024), (1024, 1), 0); del buf458  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf460, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg355_1, (4096, 1024), (1, 4096), 0), out=buf461)
        del arg355_1
        buf462 = reinterpret_tensor(buf461, (4, 512, 1024), (524288, 1024, 1), 0); del buf461  # reuse
        buf466 = reinterpret_tensor(buf447, (4, 512, 1024), (524288, 1024, 1), 0); del buf447  # reuse
        # Topologically Sorted Source Nodes: [attention_output_21, layer_output_21, ln_outputs_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf462, buf441, buf454, arg350_1, arg356_1, arg357_1, arg358_1, buf466, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg350_1
        del arg356_1
        del arg357_1
        del arg358_1
        buf467 = buf454; del buf454  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg360_1, reinterpret_tensor(buf466, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg359_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf467)
        del arg359_1
        del arg360_1
        buf468 = reinterpret_tensor(buf441, (2048, 1024), (1024, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [linear_133], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg362_1, reinterpret_tensor(buf466, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg361_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf468)
        del arg361_1
        del arg362_1
        buf469 = buf446; del buf446  # reuse
        # Topologically Sorted Source Nodes: [linear_134], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg364_1, reinterpret_tensor(buf466, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg363_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf469)
        del arg363_1
        del arg364_1
        del buf466
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf470 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf467, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf468, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf469, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        buf471 = buf470[0]
        del buf470
        buf475 = buf469; del buf469  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf471, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg365_1, (1024, 1024), (1, 1024), 0), out=buf475)
        del arg365_1
        buf479 = reinterpret_tensor(buf471, (4, 512, 1024), (524288, 1024, 1), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [attention_output_22, ln_output_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf462, buf475, arg366_1, arg367_1, arg368_1, buf479, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg367_1
        del arg368_1
        buf480 = reinterpret_tensor(buf460, (2048, 4096), (4096, 1), 0); del buf460  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf479, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg369_1, (1024, 4096), (1, 1024), 0), out=buf480)
        del arg369_1
        buf481 = reinterpret_tensor(buf480, (4, 512, 4096), (2097152, 4096, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_135], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf481, arg370_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg370_1
        buf482 = reinterpret_tensor(buf479, (2048, 1024), (1024, 1), 0); del buf479  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf481, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg371_1, (4096, 1024), (1, 4096), 0), out=buf482)
        del arg371_1
        buf483 = reinterpret_tensor(buf482, (4, 512, 1024), (524288, 1024, 1), 0); del buf482  # reuse
        buf487 = reinterpret_tensor(buf468, (4, 512, 1024), (524288, 1024, 1), 0); del buf468  # reuse
        # Topologically Sorted Source Nodes: [attention_output_22, layer_output_22, ln_outputs_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf483, buf462, buf475, arg366_1, arg372_1, arg373_1, arg374_1, buf487, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg366_1
        del arg372_1
        del arg373_1
        del arg374_1
        buf488 = buf475; del buf475  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg376_1, reinterpret_tensor(buf487, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg375_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf488)
        del arg375_1
        del arg376_1
        buf489 = reinterpret_tensor(buf462, (2048, 1024), (1024, 1), 0); del buf462  # reuse
        # Topologically Sorted Source Nodes: [linear_139], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg378_1, reinterpret_tensor(buf487, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg377_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf489)
        del arg377_1
        del arg378_1
        buf490 = buf467; del buf467  # reuse
        # Topologically Sorted Source Nodes: [linear_140], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg380_1, reinterpret_tensor(buf487, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg379_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf490)
        del arg379_1
        del arg380_1
        del buf487
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf491 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf488, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf489, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), reinterpret_tensor(buf490, (4, 16, 512, 64), (524288, 64, 1024, 1), 0), None, False, scale=0.125)
        del buf488
        buf492 = buf491[0]
        del buf491
        buf496 = buf490; del buf490  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf492, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg381_1, (1024, 1024), (1, 1024), 0), out=buf496)
        del arg381_1
        buf500 = reinterpret_tensor(buf492, (4, 512, 1024), (524288, 1024, 1), 0); del buf492  # reuse
        # Topologically Sorted Source Nodes: [attention_output_23, ln_output_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf483, buf496, arg382_1, arg383_1, arg384_1, buf500, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg383_1
        del arg384_1
        buf501 = reinterpret_tensor(buf481, (2048, 4096), (4096, 1), 0); del buf481  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf500, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg385_1, (1024, 4096), (1, 1024), 0), out=buf501)
        del arg385_1
        buf502 = reinterpret_tensor(buf501, (4, 512, 4096), (2097152, 4096, 1), 0); del buf501  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_141], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf502, arg386_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg386_1
        buf503 = reinterpret_tensor(buf500, (2048, 1024), (1024, 1), 0); del buf500  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf502, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg387_1, (4096, 1024), (1, 4096), 0), out=buf503)
        del arg387_1
        del buf502
        buf504 = reinterpret_tensor(buf503, (4, 512, 1024), (524288, 1024, 1), 0); del buf503  # reuse
        buf508 = reinterpret_tensor(buf489, (4, 512, 1024), (524288, 1024, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [attention_output_23, layer_output_23, hidden_states_144], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf504, buf483, buf496, arg382_1, arg388_1, arg389_1, arg390_1, buf508, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg382_1
        del arg388_1
        del arg389_1
        del arg390_1
        del buf483
        del buf496
        buf509 = reinterpret_tensor(buf504, (2048, 1024), (1024, 1), 0); del buf504  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf508, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg392_1, (1024, 1024), (1, 1024), 0), out=buf509)
        del arg392_1
        buf513 = buf508; del buf508  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_146, hidden_states_147], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_4.run(buf509, arg393_1, arg394_1, arg395_1, buf513, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg393_1
        del arg394_1
        del arg395_1
        del buf509
        buf514 = empty_strided_cuda((2048, 29056), (29056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_148], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg396_1, reinterpret_tensor(buf513, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg2_1, (1024, 29056), (1, 1024), 0), alpha=1, beta=1, out=buf514)
        del arg2_1
        del arg396_1
        del buf513
        buf515 = empty_strided_cuda((2044, 1), (1, 2048), torch.float32)
        buf516 = empty_strided_cuda((2044, 1), (1, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_5.run(buf514, buf515, buf516, 2044, 29056, grid=grid(2044), stream=stream0)
        buf517 = empty_strided_cuda((), (), torch.float32)
        buf519 = buf517; del buf517  # reuse
        # Topologically Sorted Source Nodes: [lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_6.run(buf519, arg0_1, buf514, buf515, buf516, 1, 2044, grid=grid(1), stream=stream0)
        del arg0_1
        del buf515
        del buf516
    return (buf519, reinterpret_tensor(buf514, (4, 512, 29056), (14876672, 29056, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((29056, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
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
    arg241_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
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
    arg289_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg392_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((29056, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MegatronBertForCausalLM', benchmark_compiled_module)
