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


# kernel path: /tmp/torchinductor_sahanp/72/c72hzaayutw6ovnfybfkbxb5xicpcqlf7xsgsa3wb3snj5c2eoyd.py
# Topologically Sorted Source Nodes: [inputs_embeds, token_type_embeddings, embeddings, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embeddings => add
#   embeddings_1 => add_1
#   embeddings_2 => add_2, add_3, mul, mul_1, rsqrt, sub, var_mean
#   inputs_embeds => embedding
#   position_embeddings => embedding_2
#   token_type_embeddings => embedding_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1, 3), kwargs = {})
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
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([RBLOCK], 32000, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32000), "index out of bounds: 0 <= tmp4 < 32000")
    tmp6 = tl.load(in_ptr1 + (r2 + (768*tmp4)), rmask, other=0.0)
    tmp8 = tl.full([RBLOCK], 4, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 4), "index out of bounds: 0 <= tmp11 < 4")
    tmp13 = tl.load(in_ptr3 + (r2 + (768*tmp11)), rmask, other=0.0)
    tmp14 = tmp6 + tmp13
    tmp16 = tl.full([RBLOCK], 512, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert((0 <= tmp19) & (tmp19 < 512), "index out of bounds: 0 <= tmp19 < 512")
    tmp21 = tl.load(in_ptr5 + (r2 + (768*tmp19)), rmask, other=0.0)
    tmp22 = tmp14 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tl.full([1], 768, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = tmp22 - tmp32
    tmp40 = 768.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-12
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp22, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp49, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vf/cvfpbitegfqrbfwwcknjoa3dvrs2kouh2tu54rtrz4l2bmapdjqd.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %add_tensor_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_25, %arg9_1), kwargs = {})
triton_poi_fused_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/al/calq5vo6r2le6axusu6u65xfpnzuhqzk6abntabm6hwglutwmf4e.py
# Topologically Sorted Source Nodes: [add_1, hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_1 => add_4
#   hidden_states => var_mean_1
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %select), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((2*r1) + (256*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
    tl.store(out_ptr1 + (x0), tmp5, None)
    tl.store(out_ptr2 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/et/cetu3trhvacuooe7lvbkmelddiwzyp35ux4razxlz3mj7gydjixl.py
# Topologically Sorted Source Nodes: [add_1, hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_1 => add_4
#   hidden_states => var_mean_1
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %select), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (6*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (6*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (6*x0)), rmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp3, 0)
    tmp8 = tl.where(rmask, tmp4, 0)
    tmp9 = tl.where(rmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, None)
    tl.store(out_ptr1 + (x0), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ow/cow25vqscstve5t3vtyztq7uwaigrrph4ametfqbbmscbptwaxu2.py
# Topologically Sorted Source Nodes: [add_1, hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_1 => add_4
#   hidden_states => add_5, add_6, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %select), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_3), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-12), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg10_1), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg11_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = (xindex // 768)
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (2*x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 768.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-12
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6r/c6r3bqskaoht75wxzcj7lnrxltiz5eb3w7xtad74b66f4d4f3geo.py
# Topologically Sorted Source Nodes: [mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_2], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_2 => add_7
#   add_3 => add_8
#   hidden_states_2 => mul_7
#   mul => mul_4
#   mul_1 => mul_5
#   mul_2 => mul_6
#   pow_1 => pow_1
#   tanh => tanh
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, 0.5), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_3, 3.0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.7978845608028654), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_6,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_8), kwargs = {})
triton_poi_fused_add_mul_pow_tanh_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/mk/cmk3nqoimdegpbpbaswakdrvzngdc4av5veyl62dijlgb3ymfgdj.py
# Topologically Sorted Source Nodes: [add_4, hidden_states_5], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_4 => add_9
#   hidden_states_5 => add_10, add_11, mul_8, mul_9, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %add_6), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_5), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-12), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %arg16_1), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg17_1), kwargs = {})
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/m4/cm4annyqfbephogipdwe2vgk64vmbxlyvvlog6ib6zsigsqirypl.py
# Topologically Sorted Source Nodes: [mul_48, pow_13, mul_49, add_49, mul_50, tanh_13, add_50, hidden_states_73, hidden_states_74], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_49 => add_100
#   add_50 => add_101
#   hidden_states_73 => mul_101
#   hidden_states_74 => add_102, add_103, mul_102, mul_103, rsqrt_25, sub_25, var_mean_25
#   mul_48 => mul_98
#   mul_49 => mul_99
#   mul_50 => mul_100
#   pow_13 => pow_13
#   tanh_13 => tanh_13
# Graph fragment:
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_51, 0.5), kwargs = {})
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_51, 3.0), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_13, 0.044715), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_51, %mul_99), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_100, 0.7978845608028654), kwargs = {})
#   %tanh_13 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_100,), kwargs = {})
#   %add_101 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh_13, 1.0), kwargs = {})
#   %mul_101 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_98, %add_101), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_101, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_101, %getitem_51), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-12), kwargs = {})
#   %rsqrt_25 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_102,), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt_25), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, %arg110_1), kwargs = {})
#   %add_103 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_103, %arg111_1), kwargs = {})
triton_per_fused_add_mul_native_layer_norm_pow_tanh_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_pow_tanh_7', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
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
    tmp39 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 768, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 768.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-12
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp42, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7w/c7wx3dg7xs5oxgvcejpx5dzlrrooug3i2lp7ahrriboczvlc4i74.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   masked_lm_loss => amax, exp, sub_26, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_54, [1], True), kwargs = {})
#   %sub_26 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_54, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_26,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_red_fused__log_softmax_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 32000
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
        tmp0 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/22/c22nlgcygeaquteon4dztdcu77yef7ewofqcn264sxyopxj26grs.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   masked_lm_loss => convert_element_type_12, div, full_default_1, ne_1, ne_2, neg, sum_2, sum_3, where_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_55, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_55, -100), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type_12), kwargs = {})
triton_red_fused_nll_loss_forward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_9', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 8192
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
        tmp5 = tl.full([XBLOCK, RBLOCK], 32000, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 32000)) | ~(rmask), "index out of bounds: 0 <= tmp8 < 32000")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (32000*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 512), (512, 1))
    assert_size_stride(arg1_1, (1, 512), (512, 1))
    assert_size_stride(arg2_1, (1, 512), (512, 1))
    assert_size_stride(arg3_1, (32000, 768), (768, 1))
    assert_size_stride(arg4_1, (4, 768), (768, 1))
    assert_size_stride(arg5_1, (512, 768), (768, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, 768), (768, 1))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (3072, 768), (768, 1))
    assert_size_stride(arg13_1, (3072, ), (1, ))
    assert_size_stride(arg14_1, (768, 3072), (3072, 1))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (3072, 768), (768, 1))
    assert_size_stride(arg21_1, (3072, ), (1, ))
    assert_size_stride(arg22_1, (768, 3072), (3072, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (3072, 768), (768, 1))
    assert_size_stride(arg29_1, (3072, ), (1, ))
    assert_size_stride(arg30_1, (768, 3072), (3072, 1))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (3072, 768), (768, 1))
    assert_size_stride(arg37_1, (3072, ), (1, ))
    assert_size_stride(arg38_1, (768, 3072), (3072, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (3072, 768), (768, 1))
    assert_size_stride(arg45_1, (3072, ), (1, ))
    assert_size_stride(arg46_1, (768, 3072), (3072, 1))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (3072, 768), (768, 1))
    assert_size_stride(arg53_1, (3072, ), (1, ))
    assert_size_stride(arg54_1, (768, 3072), (3072, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (3072, 768), (768, 1))
    assert_size_stride(arg61_1, (3072, ), (1, ))
    assert_size_stride(arg62_1, (768, 3072), (3072, 1))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (3072, 768), (768, 1))
    assert_size_stride(arg69_1, (3072, ), (1, ))
    assert_size_stride(arg70_1, (768, 3072), (3072, 1))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (3072, 768), (768, 1))
    assert_size_stride(arg77_1, (3072, ), (1, ))
    assert_size_stride(arg78_1, (768, 3072), (3072, 1))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (3072, 768), (768, 1))
    assert_size_stride(arg85_1, (3072, ), (1, ))
    assert_size_stride(arg86_1, (768, 3072), (3072, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (3072, 768), (768, 1))
    assert_size_stride(arg93_1, (3072, ), (1, ))
    assert_size_stride(arg94_1, (768, 3072), (3072, 1))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
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
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (32000, ), (1, ))
    assert_size_stride(arg113_1, (16, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 512, 768), (393216, 768, 1), torch.float32)
        buf4 = empty_strided_cuda((16, 512, 768), (393216, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, token_type_embeddings, embeddings, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg0_1, arg3_1, arg1_1, arg4_1, arg2_1, arg5_1, arg6_1, arg7_1, buf0, buf4, 8192, 768, grid=grid(8192), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        buf5 = reinterpret_tensor(buf0, (8192, 768), (768, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (8192, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0), out=buf5)
        del arg8_1
        buf6 = empty_strided_cuda((16, 512, 768), (393216, 768, 1), torch.complex64)
        buf7 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf7, arg9_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg9_1
        buf6.copy_(reinterpret_tensor(buf7, (16, 512, 768), (393216, 768, 1), 0))
        # Topologically Sorted Source Nodes: [fft_fftn], Original ATen: [aten._fft_c2c]
        buf9 = torch.ops.aten._fft_c2c.default(buf6, [1, 2], 0, True)
        del buf6
        buf10 = buf9
        del buf9
        # Topologically Sorted Source Nodes: [outputs], Original ATen: [aten.view_as_real]
        buf11 = torch.ops.aten.view_as_real.default(buf10)
        buf12 = buf11
        buf13 = empty_strided_cuda((16, 512, 1, 6), (3072, 6, 49152, 1), torch.float32)
        buf14 = empty_strided_cuda((16, 512, 1, 6), (3072, 6, 49152, 1), torch.float32)
        buf15 = empty_strided_cuda((16, 512, 1, 6), (3072, 6, 49152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_1, hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf7, buf12, buf13, buf14, buf15, 49152, 128, grid=grid(49152), stream=stream0)
        buf16 = empty_strided_cuda((16, 512, 1), (512, 1, 8192), torch.float32)
        buf17 = empty_strided_cuda((16, 512, 1), (512, 1, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [add_1, hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf13, buf14, buf15, buf16, buf17, 8192, 6, grid=grid(8192), stream=stream0)
        buf19 = reinterpret_tensor(buf7, (16, 512, 768), (393216, 768, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [add_1, hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf19, buf12, buf16, buf17, arg10_1, arg11_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg10_1
        del arg11_1
        del buf11
        del buf12
        buf20 = empty_strided_cuda((8192, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (8192, 768), (768, 1), 0), reinterpret_tensor(arg12_1, (768, 3072), (1, 768), 0), out=buf20)
        del arg12_1
        buf21 = reinterpret_tensor(buf20, (16, 512, 3072), (1572864, 3072, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_2], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf21, arg13_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg13_1
        buf22 = reinterpret_tensor(buf4, (8192, 768), (768, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg14_1, (3072, 768), (1, 3072), 0), out=buf22)
        del arg14_1
        buf26 = empty_strided_cuda((16, 512, 768), (393216, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_4, hidden_states_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf22, arg15_1, buf19, arg16_1, arg17_1, buf26, 8192, 768, grid=grid(8192), stream=stream0)
        del arg15_1
        del arg16_1
        del arg17_1
        buf27 = buf10; del buf10  # reuse
        buf27.copy_(buf26)
        # Topologically Sorted Source Nodes: [fft_fftn_1], Original ATen: [aten._fft_c2c]
        buf29 = torch.ops.aten._fft_c2c.default(buf27, [1, 2], 0, True)
        del buf27
        buf30 = buf29
        del buf29
        # Topologically Sorted Source Nodes: [outputs_1], Original ATen: [aten.view_as_real]
        buf31 = torch.ops.aten.view_as_real.default(buf30)
        buf32 = buf31
        buf33 = buf15; del buf15  # reuse
        buf34 = buf14; del buf14  # reuse
        buf35 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [add_5, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf26, buf32, buf33, buf34, buf35, 49152, 128, grid=grid(49152), stream=stream0)
        buf36 = buf17; del buf17  # reuse
        buf37 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [add_5, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf33, buf34, buf35, buf36, buf37, 8192, 6, grid=grid(8192), stream=stream0)
        buf39 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [add_5, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf39, buf32, buf36, buf37, arg18_1, arg19_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg18_1
        del arg19_1
        del buf31
        del buf32
        buf40 = reinterpret_tensor(buf21, (8192, 3072), (3072, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (8192, 768), (768, 1), 0), reinterpret_tensor(arg20_1, (768, 3072), (1, 768), 0), out=buf40)
        del arg20_1
        buf41 = reinterpret_tensor(buf40, (16, 512, 3072), (1572864, 3072, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_8], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf41, arg21_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg21_1
        buf42 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf41, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg22_1, (3072, 768), (1, 3072), 0), out=buf42)
        del arg22_1
        buf46 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [add_8, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf42, arg23_1, buf39, arg24_1, arg25_1, buf46, 8192, 768, grid=grid(8192), stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        buf47 = buf30; del buf30  # reuse
        buf47.copy_(buf46)
        # Topologically Sorted Source Nodes: [fft_fftn_2], Original ATen: [aten._fft_c2c]
        buf49 = torch.ops.aten._fft_c2c.default(buf47, [1, 2], 0, True)
        del buf47
        buf50 = buf49
        del buf49
        # Topologically Sorted Source Nodes: [outputs_2], Original ATen: [aten.view_as_real]
        buf51 = torch.ops.aten.view_as_real.default(buf50)
        buf52 = buf51
        buf53 = buf35; del buf35  # reuse
        buf54 = buf34; del buf34  # reuse
        buf55 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [add_9, hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf46, buf52, buf53, buf54, buf55, 49152, 128, grid=grid(49152), stream=stream0)
        buf56 = buf37; del buf37  # reuse
        buf57 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [add_9, hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf53, buf54, buf55, buf56, buf57, 8192, 6, grid=grid(8192), stream=stream0)
        buf59 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [add_9, hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf59, buf52, buf56, buf57, arg26_1, arg27_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg26_1
        del arg27_1
        del buf51
        del buf52
        buf60 = reinterpret_tensor(buf41, (8192, 3072), (3072, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (8192, 768), (768, 1), 0), reinterpret_tensor(arg28_1, (768, 3072), (1, 768), 0), out=buf60)
        del arg28_1
        buf61 = reinterpret_tensor(buf60, (16, 512, 3072), (1572864, 3072, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_14], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf61, arg29_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg29_1
        buf62 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg30_1, (3072, 768), (1, 3072), 0), out=buf62)
        del arg30_1
        buf66 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [add_12, hidden_states_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf62, arg31_1, buf59, arg32_1, arg33_1, buf66, 8192, 768, grid=grid(8192), stream=stream0)
        del arg31_1
        del arg32_1
        del arg33_1
        buf67 = buf50; del buf50  # reuse
        buf67.copy_(buf66)
        # Topologically Sorted Source Nodes: [fft_fftn_3], Original ATen: [aten._fft_c2c]
        buf69 = torch.ops.aten._fft_c2c.default(buf67, [1, 2], 0, True)
        del buf67
        buf70 = buf69
        del buf69
        # Topologically Sorted Source Nodes: [outputs_3], Original ATen: [aten.view_as_real]
        buf71 = torch.ops.aten.view_as_real.default(buf70)
        buf72 = buf71
        buf73 = buf55; del buf55  # reuse
        buf74 = buf54; del buf54  # reuse
        buf75 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [add_13, hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf66, buf72, buf73, buf74, buf75, 49152, 128, grid=grid(49152), stream=stream0)
        buf76 = buf57; del buf57  # reuse
        buf77 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [add_13, hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf73, buf74, buf75, buf76, buf77, 8192, 6, grid=grid(8192), stream=stream0)
        buf79 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [add_13, hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf79, buf72, buf76, buf77, arg34_1, arg35_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg34_1
        del arg35_1
        del buf71
        del buf72
        buf80 = reinterpret_tensor(buf61, (8192, 3072), (3072, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (8192, 768), (768, 1), 0), reinterpret_tensor(arg36_1, (768, 3072), (1, 768), 0), out=buf80)
        del arg36_1
        buf81 = reinterpret_tensor(buf80, (16, 512, 3072), (1572864, 3072, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_20], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf81, arg37_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg37_1
        buf82 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf81, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg38_1, (3072, 768), (1, 3072), 0), out=buf82)
        del arg38_1
        buf86 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [add_16, hidden_states_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf82, arg39_1, buf79, arg40_1, arg41_1, buf86, 8192, 768, grid=grid(8192), stream=stream0)
        del arg39_1
        del arg40_1
        del arg41_1
        buf87 = buf70; del buf70  # reuse
        buf87.copy_(buf86)
        # Topologically Sorted Source Nodes: [fft_fftn_4], Original ATen: [aten._fft_c2c]
        buf89 = torch.ops.aten._fft_c2c.default(buf87, [1, 2], 0, True)
        del buf87
        buf90 = buf89
        del buf89
        # Topologically Sorted Source Nodes: [outputs_4], Original ATen: [aten.view_as_real]
        buf91 = torch.ops.aten.view_as_real.default(buf90)
        buf92 = buf91
        buf93 = buf75; del buf75  # reuse
        buf94 = buf74; del buf74  # reuse
        buf95 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [add_17, hidden_states_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf86, buf92, buf93, buf94, buf95, 49152, 128, grid=grid(49152), stream=stream0)
        buf96 = buf77; del buf77  # reuse
        buf97 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [add_17, hidden_states_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf93, buf94, buf95, buf96, buf97, 8192, 6, grid=grid(8192), stream=stream0)
        buf99 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [add_17, hidden_states_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf99, buf92, buf96, buf97, arg42_1, arg43_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg42_1
        del arg43_1
        del buf91
        del buf92
        buf100 = reinterpret_tensor(buf81, (8192, 3072), (3072, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (8192, 768), (768, 1), 0), reinterpret_tensor(arg44_1, (768, 3072), (1, 768), 0), out=buf100)
        del arg44_1
        buf101 = reinterpret_tensor(buf100, (16, 512, 3072), (1572864, 3072, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_26], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf101, arg45_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg45_1
        buf102 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg46_1, (3072, 768), (1, 3072), 0), out=buf102)
        del arg46_1
        buf106 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [add_20, hidden_states_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf102, arg47_1, buf99, arg48_1, arg49_1, buf106, 8192, 768, grid=grid(8192), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        buf107 = buf90; del buf90  # reuse
        buf107.copy_(buf106)
        # Topologically Sorted Source Nodes: [fft_fftn_5], Original ATen: [aten._fft_c2c]
        buf109 = torch.ops.aten._fft_c2c.default(buf107, [1, 2], 0, True)
        del buf107
        buf110 = buf109
        del buf109
        # Topologically Sorted Source Nodes: [outputs_5], Original ATen: [aten.view_as_real]
        buf111 = torch.ops.aten.view_as_real.default(buf110)
        buf112 = buf111
        buf113 = buf95; del buf95  # reuse
        buf114 = buf94; del buf94  # reuse
        buf115 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [add_21, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf106, buf112, buf113, buf114, buf115, 49152, 128, grid=grid(49152), stream=stream0)
        buf116 = buf97; del buf97  # reuse
        buf117 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [add_21, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf113, buf114, buf115, buf116, buf117, 8192, 6, grid=grid(8192), stream=stream0)
        buf119 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [add_21, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf119, buf112, buf116, buf117, arg50_1, arg51_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg50_1
        del arg51_1
        del buf111
        del buf112
        buf120 = reinterpret_tensor(buf101, (8192, 3072), (3072, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (8192, 768), (768, 1), 0), reinterpret_tensor(arg52_1, (768, 3072), (1, 768), 0), out=buf120)
        del arg52_1
        buf121 = reinterpret_tensor(buf120, (16, 512, 3072), (1572864, 3072, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_32], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf121, arg53_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg53_1
        buf122 = reinterpret_tensor(buf99, (8192, 768), (768, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg54_1, (3072, 768), (1, 3072), 0), out=buf122)
        del arg54_1
        buf126 = reinterpret_tensor(buf102, (16, 512, 768), (393216, 768, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [add_24, hidden_states_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf122, arg55_1, buf119, arg56_1, arg57_1, buf126, 8192, 768, grid=grid(8192), stream=stream0)
        del arg55_1
        del arg56_1
        del arg57_1
        buf127 = buf110; del buf110  # reuse
        buf127.copy_(buf126)
        # Topologically Sorted Source Nodes: [fft_fftn_6], Original ATen: [aten._fft_c2c]
        buf129 = torch.ops.aten._fft_c2c.default(buf127, [1, 2], 0, True)
        del buf127
        buf130 = buf129
        del buf129
        # Topologically Sorted Source Nodes: [outputs_6], Original ATen: [aten.view_as_real]
        buf131 = torch.ops.aten.view_as_real.default(buf130)
        buf132 = buf131
        buf133 = buf115; del buf115  # reuse
        buf134 = buf114; del buf114  # reuse
        buf135 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [add_25, hidden_states_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf126, buf132, buf133, buf134, buf135, 49152, 128, grid=grid(49152), stream=stream0)
        buf136 = buf117; del buf117  # reuse
        buf137 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [add_25, hidden_states_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf133, buf134, buf135, buf136, buf137, 8192, 6, grid=grid(8192), stream=stream0)
        buf139 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [add_25, hidden_states_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf139, buf132, buf136, buf137, arg58_1, arg59_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg58_1
        del arg59_1
        del buf131
        del buf132
        buf140 = reinterpret_tensor(buf121, (8192, 3072), (3072, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf139, (8192, 768), (768, 1), 0), reinterpret_tensor(arg60_1, (768, 3072), (1, 768), 0), out=buf140)
        del arg60_1
        buf141 = reinterpret_tensor(buf140, (16, 512, 3072), (1572864, 3072, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_38], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf141, arg61_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg61_1
        buf142 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg62_1, (3072, 768), (1, 3072), 0), out=buf142)
        del arg62_1
        buf146 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [add_28, hidden_states_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf142, arg63_1, buf139, arg64_1, arg65_1, buf146, 8192, 768, grid=grid(8192), stream=stream0)
        del arg63_1
        del arg64_1
        del arg65_1
        buf147 = buf130; del buf130  # reuse
        buf147.copy_(buf146)
        # Topologically Sorted Source Nodes: [fft_fftn_7], Original ATen: [aten._fft_c2c]
        buf149 = torch.ops.aten._fft_c2c.default(buf147, [1, 2], 0, True)
        del buf147
        buf150 = buf149
        del buf149
        # Topologically Sorted Source Nodes: [outputs_7], Original ATen: [aten.view_as_real]
        buf151 = torch.ops.aten.view_as_real.default(buf150)
        buf152 = buf151
        buf153 = buf135; del buf135  # reuse
        buf154 = buf134; del buf134  # reuse
        buf155 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [add_29, hidden_states_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf146, buf152, buf153, buf154, buf155, 49152, 128, grid=grid(49152), stream=stream0)
        buf156 = buf137; del buf137  # reuse
        buf157 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [add_29, hidden_states_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf153, buf154, buf155, buf156, buf157, 8192, 6, grid=grid(8192), stream=stream0)
        buf159 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [add_29, hidden_states_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf159, buf152, buf156, buf157, arg66_1, arg67_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg66_1
        del arg67_1
        del buf151
        del buf152
        buf160 = reinterpret_tensor(buf141, (8192, 3072), (3072, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (8192, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 3072), (1, 768), 0), out=buf160)
        del arg68_1
        buf161 = reinterpret_tensor(buf160, (16, 512, 3072), (1572864, 3072, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_44], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf161, arg69_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg69_1
        buf162 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf161, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg70_1, (3072, 768), (1, 3072), 0), out=buf162)
        del arg70_1
        buf166 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [add_32, hidden_states_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf162, arg71_1, buf159, arg72_1, arg73_1, buf166, 8192, 768, grid=grid(8192), stream=stream0)
        del arg71_1
        del arg72_1
        del arg73_1
        buf167 = buf150; del buf150  # reuse
        buf167.copy_(buf166)
        # Topologically Sorted Source Nodes: [fft_fftn_8], Original ATen: [aten._fft_c2c]
        buf169 = torch.ops.aten._fft_c2c.default(buf167, [1, 2], 0, True)
        del buf167
        buf170 = buf169
        del buf169
        # Topologically Sorted Source Nodes: [outputs_8], Original ATen: [aten.view_as_real]
        buf171 = torch.ops.aten.view_as_real.default(buf170)
        buf172 = buf171
        buf173 = buf155; del buf155  # reuse
        buf174 = buf154; del buf154  # reuse
        buf175 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [add_33, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf166, buf172, buf173, buf174, buf175, 49152, 128, grid=grid(49152), stream=stream0)
        buf176 = buf157; del buf157  # reuse
        buf177 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [add_33, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf173, buf174, buf175, buf176, buf177, 8192, 6, grid=grid(8192), stream=stream0)
        buf179 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [add_33, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf179, buf172, buf176, buf177, arg74_1, arg75_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg74_1
        del arg75_1
        del buf171
        del buf172
        buf180 = reinterpret_tensor(buf161, (8192, 3072), (3072, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (8192, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 3072), (1, 768), 0), out=buf180)
        del arg76_1
        buf181 = reinterpret_tensor(buf180, (16, 512, 3072), (1572864, 3072, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_50], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf181, arg77_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg77_1
        buf182 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf181, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg78_1, (3072, 768), (1, 3072), 0), out=buf182)
        del arg78_1
        buf186 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [add_36, hidden_states_53], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf182, arg79_1, buf179, arg80_1, arg81_1, buf186, 8192, 768, grid=grid(8192), stream=stream0)
        del arg79_1
        del arg80_1
        del arg81_1
        buf187 = buf170; del buf170  # reuse
        buf187.copy_(buf186)
        # Topologically Sorted Source Nodes: [fft_fftn_9], Original ATen: [aten._fft_c2c]
        buf189 = torch.ops.aten._fft_c2c.default(buf187, [1, 2], 0, True)
        del buf187
        buf190 = buf189
        del buf189
        # Topologically Sorted Source Nodes: [outputs_9], Original ATen: [aten.view_as_real]
        buf191 = torch.ops.aten.view_as_real.default(buf190)
        buf192 = buf191
        buf193 = buf175; del buf175  # reuse
        buf194 = buf174; del buf174  # reuse
        buf195 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [add_37, hidden_states_54], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf186, buf192, buf193, buf194, buf195, 49152, 128, grid=grid(49152), stream=stream0)
        buf196 = buf177; del buf177  # reuse
        buf197 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [add_37, hidden_states_54], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf193, buf194, buf195, buf196, buf197, 8192, 6, grid=grid(8192), stream=stream0)
        buf199 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [add_37, hidden_states_54], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf199, buf192, buf196, buf197, arg82_1, arg83_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg82_1
        del arg83_1
        del buf191
        del buf192
        buf200 = reinterpret_tensor(buf181, (8192, 3072), (3072, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (8192, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 3072), (1, 768), 0), out=buf200)
        del arg84_1
        buf201 = reinterpret_tensor(buf200, (16, 512, 3072), (1572864, 3072, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_56], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf201, arg85_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg85_1
        buf202 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg86_1, (3072, 768), (1, 3072), 0), out=buf202)
        del arg86_1
        buf206 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [add_40, hidden_states_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf202, arg87_1, buf199, arg88_1, arg89_1, buf206, 8192, 768, grid=grid(8192), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        buf207 = buf190; del buf190  # reuse
        buf207.copy_(buf206)
        # Topologically Sorted Source Nodes: [fft_fftn_10], Original ATen: [aten._fft_c2c]
        buf209 = torch.ops.aten._fft_c2c.default(buf207, [1, 2], 0, True)
        del buf207
        buf210 = buf209
        del buf209
        # Topologically Sorted Source Nodes: [outputs_10], Original ATen: [aten.view_as_real]
        buf211 = torch.ops.aten.view_as_real.default(buf210)
        buf212 = buf211
        buf213 = buf195; del buf195  # reuse
        buf214 = buf194; del buf194  # reuse
        buf215 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [add_41, hidden_states_60], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf206, buf212, buf213, buf214, buf215, 49152, 128, grid=grid(49152), stream=stream0)
        buf216 = buf197; del buf197  # reuse
        buf217 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [add_41, hidden_states_60], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf213, buf214, buf215, buf216, buf217, 8192, 6, grid=grid(8192), stream=stream0)
        buf219 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [add_41, hidden_states_60], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf219, buf212, buf216, buf217, arg90_1, arg91_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg90_1
        del arg91_1
        del buf211
        del buf212
        buf220 = reinterpret_tensor(buf201, (8192, 3072), (3072, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (8192, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 3072), (1, 768), 0), out=buf220)
        del arg92_1
        buf221 = reinterpret_tensor(buf220, (16, 512, 3072), (1572864, 3072, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_62], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf221, arg93_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg93_1
        buf222 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg94_1, (3072, 768), (1, 3072), 0), out=buf222)
        del arg94_1
        buf226 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [add_44, hidden_states_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf222, arg95_1, buf219, arg96_1, arg97_1, buf226, 8192, 768, grid=grid(8192), stream=stream0)
        del arg95_1
        del arg96_1
        del arg97_1
        buf227 = buf210; del buf210  # reuse
        buf227.copy_(buf226)
        # Topologically Sorted Source Nodes: [fft_fftn_11], Original ATen: [aten._fft_c2c]
        buf229 = torch.ops.aten._fft_c2c.default(buf227, [1, 2], 0, True)
        del buf227
        buf230 = buf229
        del buf229
        # Topologically Sorted Source Nodes: [outputs_11], Original ATen: [aten.view_as_real]
        buf231 = torch.ops.aten.view_as_real.default(buf230)
        buf232 = buf231
        buf233 = buf215; del buf215  # reuse
        buf234 = buf214; del buf214  # reuse
        buf235 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [add_45, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_2.run(buf226, buf232, buf233, buf234, buf235, 49152, 128, grid=grid(49152), stream=stream0)
        buf236 = buf217; del buf217  # reuse
        buf237 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [add_45, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf233, buf234, buf235, buf236, buf237, 8192, 6, grid=grid(8192), stream=stream0)
        del buf233
        del buf234
        del buf235
        buf239 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [add_45, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_4.run(buf239, buf232, buf236, buf237, arg98_1, arg99_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg98_1
        del arg99_1
        del buf230
        del buf231
        del buf232
        buf240 = reinterpret_tensor(buf221, (8192, 3072), (3072, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (8192, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 3072), (1, 768), 0), out=buf240)
        del arg100_1
        buf241 = reinterpret_tensor(buf240, (16, 512, 3072), (1572864, 3072, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_68], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_5.run(buf241, arg101_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg101_1
        buf242 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf241, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg102_1, (3072, 768), (1, 3072), 0), out=buf242)
        del arg102_1
        del buf241
        buf246 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [add_48, hidden_states_71], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf242, arg103_1, buf239, arg104_1, arg105_1, buf246, 8192, 768, grid=grid(8192), stream=stream0)
        del arg103_1
        del arg104_1
        del arg105_1
        del buf239
        buf247 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf246, (8192, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 768), (1, 768), 0), out=buf247)
        del arg108_1
        buf251 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [mul_48, pow_13, mul_49, add_49, mul_50, tanh_13, add_50, hidden_states_73, hidden_states_74], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_pow_tanh_7.run(buf247, arg109_1, arg110_1, arg111_1, buf251, 8192, 768, grid=grid(8192), stream=stream0)
        del arg109_1
        del arg110_1
        del arg111_1
        del buf247
        buf252 = empty_strided_cuda((8192, 32000), (32000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg112_1, reinterpret_tensor(buf251, (8192, 768), (768, 1), 0), reinterpret_tensor(arg3_1, (768, 32000), (1, 768), 0), alpha=1, beta=1, out=buf252)
        del arg112_1
        del arg3_1
        del buf251
        buf253 = reinterpret_tensor(buf237, (8192, 1), (1, 8192), 0); del buf237  # reuse
        buf254 = reinterpret_tensor(buf236, (8192, 1), (1, 8192), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_8.run(buf252, buf253, buf254, 8192, 32000, grid=grid(8192), stream=stream0)
        buf255 = empty_strided_cuda((), (), torch.float32)
        buf257 = buf255; del buf255  # reuse
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_9.run(buf257, arg113_1, buf252, buf253, buf254, 1, 8192, grid=grid(1), stream=stream0)
        del arg113_1
        del buf253
        del buf254
    return (buf257, reinterpret_tensor(buf252, (16, 512, 32000), (16384000, 32000, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((32000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((4, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((32000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((16, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GoogleFnet', benchmark_compiled_module)
