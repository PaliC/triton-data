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


# kernel path: /tmp/torchinductor_sahanp/qp/cqp3b3bfomh5sayaa4zkkphmekr7jkwgnagimmoeiwnlcsl44x6s.py
# Topologically Sorted Source Nodes: [inputs_embeds, token_type_embeddings, embeddings, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embeddings => add
#   embeddings_1 => add_1
#   embeddings_2 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
#   inputs_embeds => embedding
#   position_embeddings => embedding_2
#   token_type_embeddings => embedding_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %arg0_1, 0), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %expand), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg201_1), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg6_1), kwargs = {})
triton_per_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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
    tmp1 = tl.full([XBLOCK, RBLOCK], 30522, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 30522), "index out of bounds: 0 <= tmp4 < 30522")
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


# kernel path: /tmp/torchinductor_sahanp/r5/cr5g7yj4l6dr6ikylsjocueiqebicdnnzve5krab4pr4zg5fyr7r.py
# Topologically Sorted Source Nodes: [add_2, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_2 => add_5
#   hidden_states_3 => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
# Graph fragment:
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_19, %view_1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_3), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-12), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg17_1), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg18_1), kwargs = {})
triton_per_fused_add_native_layer_norm_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_1', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 32768
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (256*x0)), None)
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 256, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp4 - tmp12
    tmp19 = 256.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-12
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/un/cun26y6dpjrctcq6mtfn5owbcwe6fxxjyqkjj5ivwbjuykbofe4z.py
# Topologically Sorted Source Nodes: [hidden_states_5], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_5 => add_8, erf, mul_5, mul_6, mul_7
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, 0.5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, 0.7071067811865476), kwargs = {})
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
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
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


# kernel path: /tmp/torchinductor_sahanp/ti/cti2ziubfzjflrwe4edfsjx2gucuzfmd62mj6ska6u5vyxpwfae4.py
# Topologically Sorted Source Nodes: [start_logits_1, start_loss], Original ATen: [aten.clone, aten._log_softmax]
# Source node to ATen node mapping:
#   start_logits_1 => clone_85
#   start_loss => amax_12, exp_12, sub_38, sum_13
# Graph fragment:
#   %clone_85 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_12 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_85, [1], True), kwargs = {})
#   %sub_38 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_85, %amax_12), kwargs = {})
#   %exp_12 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_38,), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_12, [1], True), kwargs = {})
triton_red_fused__log_softmax_clone_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_clone_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_sahanp/gg/cgguk5t6sua37litvq6dvjvhzzd6tlvymzjvyhnjhfagsgetf6ys.py
# Topologically Sorted Source Nodes: [end_logits_1, end_loss], Original ATen: [aten.clone, aten._log_softmax]
# Source node to ATen node mapping:
#   end_logits_1 => clone_86
#   end_loss => amax_13, exp_13, sub_40, sum_16
# Graph fragment:
#   %clone_86 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_1,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_13 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_86, [1], True), kwargs = {})
#   %sub_40 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_86, %amax_13), kwargs = {})
#   %exp_13 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_40,), kwargs = {})
#   %sum_16 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_13, [1], True), kwargs = {})
triton_red_fused__log_softmax_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_clone_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_sahanp/wv/cwvax4e62mpe2tyussifrkdvktyq4ljfsck2pllj5uo3owb2d6f6.py
# Topologically Sorted Source Nodes: [start_positions, start_loss, end_positions, end_loss, add_37, total_loss], Original ATen: [aten.clamp, aten.nll_loss_forward, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_37 => add_100
#   end_loss => convert_element_type_1, div_25, full_default_4, ne_4, ne_5, neg_1, sum_17, sum_18, where_3
#   end_positions => clamp_max_1, clamp_min_1
#   start_loss => convert_element_type, div_24, full_default_2, ne_1, ne_2, neg, sum_14, sum_15, where_1
#   start_positions => clamp_max, clamp_min
#   total_loss => div_26
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg204_1, 0), kwargs = {})
#   %clamp_max : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 512), kwargs = {})
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max, 512), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_2,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_2), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max, 512), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_14, torch.float32), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_15, %convert_element_type), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg205_1, 0), kwargs = {})
#   %clamp_max_1 : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 512), kwargs = {})
#   %ne_4 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max_1, 512), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_3,), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %neg_1, %full_default_4), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_3,), kwargs = {})
#   %ne_5 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max_1, 512), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_5,), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_17, torch.float32), kwargs = {})
#   %div_25 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_18, %convert_element_type_1), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_24, %div_25), kwargs = {})
#   %div_26 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_100, 2), kwargs = {})
triton_per_fused_add_clamp_div_nll_loss_forward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {9: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_div_nll_loss_forward_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 512), (512, 1))
    assert_size_stride(arg1_1, (1, 512), (512, 1))
    assert_size_stride(arg2_1, (30522, 128), (128, 1))
    assert_size_stride(arg3_1, (512, 128), (128, 1))
    assert_size_stride(arg4_1, (2, 128), (128, 1))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (256, 128), (128, 1))
    assert_size_stride(arg8_1, (256, ), (1, ))
    assert_size_stride(arg9_1, (256, 256), (256, 1))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, 256), (256, 1))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, 256), (256, 1))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, 256), (256, 1))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (1024, 256), (256, 1))
    assert_size_stride(arg20_1, (1024, ), (1, ))
    assert_size_stride(arg21_1, (256, 1024), (1024, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, 256), (256, 1))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (256, 256), (256, 1))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, 256), (256, 1))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (256, 256), (256, 1))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (1024, 256), (256, 1))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (256, 1024), (1024, 1))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, 256), (256, 1))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, 256), (256, 1))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, 256), (256, 1))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, 256), (256, 1))
    assert_size_stride(arg48_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (1024, 256), (256, 1))
    assert_size_stride(arg52_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (256, 1024), (1024, 1))
    assert_size_stride(arg54_1, (256, ), (1, ))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (256, 256), (256, 1))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (256, 256), (256, 1))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (256, 256), (256, 1))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (256, 256), (256, 1))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (256, ), (1, ))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg67_1, (1024, 256), (256, 1))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (256, 1024), (1024, 1))
    assert_size_stride(arg70_1, (256, ), (1, ))
    assert_size_stride(arg71_1, (256, ), (1, ))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, 256), (256, 1))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, 256), (256, 1))
    assert_size_stride(arg76_1, (256, ), (1, ))
    assert_size_stride(arg77_1, (256, 256), (256, 1))
    assert_size_stride(arg78_1, (256, ), (1, ))
    assert_size_stride(arg79_1, (256, 256), (256, 1))
    assert_size_stride(arg80_1, (256, ), (1, ))
    assert_size_stride(arg81_1, (256, ), (1, ))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (1024, 256), (256, 1))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (256, 1024), (1024, 1))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, 256), (256, 1))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (256, 256), (256, 1))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (256, 256), (256, 1))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, 256), (256, 1))
    assert_size_stride(arg96_1, (256, ), (1, ))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (1024, 256), (256, 1))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (256, 1024), (1024, 1))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (256, ), (1, ))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, 256), (256, 1))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (256, 256), (256, 1))
    assert_size_stride(arg108_1, (256, ), (1, ))
    assert_size_stride(arg109_1, (256, 256), (256, 1))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, 256), (256, 1))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (256, ), (1, ))
    assert_size_stride(arg114_1, (256, ), (1, ))
    assert_size_stride(arg115_1, (1024, 256), (256, 1))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (256, 1024), (1024, 1))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (256, ), (1, ))
    assert_size_stride(arg120_1, (256, ), (1, ))
    assert_size_stride(arg121_1, (256, 256), (256, 1))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, 256), (256, 1))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (256, 256), (256, 1))
    assert_size_stride(arg126_1, (256, ), (1, ))
    assert_size_stride(arg127_1, (256, 256), (256, 1))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (256, ), (1, ))
    assert_size_stride(arg131_1, (1024, 256), (256, 1))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (256, 1024), (1024, 1))
    assert_size_stride(arg134_1, (256, ), (1, ))
    assert_size_stride(arg135_1, (256, ), (1, ))
    assert_size_stride(arg136_1, (256, ), (1, ))
    assert_size_stride(arg137_1, (256, 256), (256, 1))
    assert_size_stride(arg138_1, (256, ), (1, ))
    assert_size_stride(arg139_1, (256, 256), (256, 1))
    assert_size_stride(arg140_1, (256, ), (1, ))
    assert_size_stride(arg141_1, (256, 256), (256, 1))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, 256), (256, 1))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, ), (1, ))
    assert_size_stride(arg147_1, (1024, 256), (256, 1))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (256, 1024), (1024, 1))
    assert_size_stride(arg150_1, (256, ), (1, ))
    assert_size_stride(arg151_1, (256, ), (1, ))
    assert_size_stride(arg152_1, (256, ), (1, ))
    assert_size_stride(arg153_1, (256, 256), (256, 1))
    assert_size_stride(arg154_1, (256, ), (1, ))
    assert_size_stride(arg155_1, (256, 256), (256, 1))
    assert_size_stride(arg156_1, (256, ), (1, ))
    assert_size_stride(arg157_1, (256, 256), (256, 1))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (256, 256), (256, 1))
    assert_size_stride(arg160_1, (256, ), (1, ))
    assert_size_stride(arg161_1, (256, ), (1, ))
    assert_size_stride(arg162_1, (256, ), (1, ))
    assert_size_stride(arg163_1, (1024, 256), (256, 1))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (256, 1024), (1024, 1))
    assert_size_stride(arg166_1, (256, ), (1, ))
    assert_size_stride(arg167_1, (256, ), (1, ))
    assert_size_stride(arg168_1, (256, ), (1, ))
    assert_size_stride(arg169_1, (256, 256), (256, 1))
    assert_size_stride(arg170_1, (256, ), (1, ))
    assert_size_stride(arg171_1, (256, 256), (256, 1))
    assert_size_stride(arg172_1, (256, ), (1, ))
    assert_size_stride(arg173_1, (256, 256), (256, 1))
    assert_size_stride(arg174_1, (256, ), (1, ))
    assert_size_stride(arg175_1, (256, 256), (256, 1))
    assert_size_stride(arg176_1, (256, ), (1, ))
    assert_size_stride(arg177_1, (256, ), (1, ))
    assert_size_stride(arg178_1, (256, ), (1, ))
    assert_size_stride(arg179_1, (1024, 256), (256, 1))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (256, 1024), (1024, 1))
    assert_size_stride(arg182_1, (256, ), (1, ))
    assert_size_stride(arg183_1, (256, ), (1, ))
    assert_size_stride(arg184_1, (256, ), (1, ))
    assert_size_stride(arg185_1, (256, 256), (256, 1))
    assert_size_stride(arg186_1, (256, ), (1, ))
    assert_size_stride(arg187_1, (256, 256), (256, 1))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (256, 256), (256, 1))
    assert_size_stride(arg190_1, (256, ), (1, ))
    assert_size_stride(arg191_1, (256, 256), (256, 1))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (256, ), (1, ))
    assert_size_stride(arg194_1, (256, ), (1, ))
    assert_size_stride(arg195_1, (1024, 256), (256, 1))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (256, 1024), (1024, 1))
    assert_size_stride(arg198_1, (256, ), (1, ))
    assert_size_stride(arg199_1, (256, ), (1, ))
    assert_size_stride(arg200_1, (256, ), (1, ))
    assert_size_stride(arg201_1, (1, 512), (512, 1))
    assert_size_stride(arg202_1, (2, 256), (256, 1))
    assert_size_stride(arg203_1, (2, ), (1, ))
    assert_size_stride(arg204_1, (64, ), (1, ))
    assert_size_stride(arg205_1, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 512, 128), (65536, 128, 1), torch.float32)
        buf4 = empty_strided_cuda((64, 512, 128), (65536, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, token_type_embeddings, embeddings, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg0_1, arg2_1, arg1_1, arg4_1, arg201_1, arg3_1, arg5_1, arg6_1, buf0, buf4, 32768, 128, grid=grid(32768), stream=stream0)
        del arg0_1
        del arg1_1
        del arg201_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del buf0
        buf5 = empty_strided_cuda((32768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf4, (32768, 128), (128, 1), 0), reinterpret_tensor(arg7_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del arg7_1
        del arg8_1
        del buf4
        buf6 = empty_strided_cuda((32768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, buf5, reinterpret_tensor(arg9_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf6)
        del arg10_1
        del arg9_1
        buf7 = empty_strided_cuda((32768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, buf5, reinterpret_tensor(arg11_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf7)
        del arg11_1
        del arg12_1
        buf8 = empty_strided_cuda((32768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg14_1, buf5, reinterpret_tensor(arg13_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf8)
        del arg13_1
        del arg14_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf6, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf7, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf8, (64, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf6
        buf10 = buf9[0]
        del buf9
        buf14 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf10, (32768, 256), (256, 1), 0), reinterpret_tensor(arg15_1, (256, 256), (1, 256), 0), out=buf14)
        del arg15_1
        buf18 = reinterpret_tensor(buf10, (64, 512, 256), (131072, 256, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [add_2, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf14, arg16_1, buf5, arg17_1, arg18_1, buf18, 32768, 256, grid=grid(32768), stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        buf19 = empty_strided_cuda((32768, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (32768, 256), (256, 1), 0), reinterpret_tensor(arg19_1, (256, 1024), (1, 256), 0), out=buf19)
        del arg19_1
        buf20 = reinterpret_tensor(buf19, (64, 512, 1024), (524288, 1024, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf20, arg20_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg20_1
        buf21 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (32768, 1024), (1024, 1), 0), reinterpret_tensor(arg21_1, (1024, 256), (1, 1024), 0), out=buf21)
        del arg21_1
        buf25 = reinterpret_tensor(buf14, (64, 512, 256), (131072, 256, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [add_3, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf21, arg22_1, buf18, arg23_1, arg24_1, buf25, 32768, 256, grid=grid(32768), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        buf26 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg26_1, reinterpret_tensor(buf25, (32768, 256), (256, 1), 0), reinterpret_tensor(arg25_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf26)
        del arg25_1
        del arg26_1
        buf27 = reinterpret_tensor(buf18, (32768, 256), (256, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg28_1, reinterpret_tensor(buf25, (32768, 256), (256, 1), 0), reinterpret_tensor(arg27_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf27)
        del arg27_1
        del arg28_1
        buf28 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg30_1, reinterpret_tensor(buf25, (32768, 256), (256, 1), 0), reinterpret_tensor(arg29_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf28)
        del arg29_1
        del arg30_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf29 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf26, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf27, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf28, (64, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf26
        buf30 = buf29[0]
        del buf29
        buf34 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (32768, 256), (256, 1), 0), reinterpret_tensor(arg31_1, (256, 256), (1, 256), 0), out=buf34)
        del arg31_1
        buf38 = reinterpret_tensor(buf30, (64, 512, 256), (131072, 256, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [add_5, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf34, arg32_1, buf25, arg33_1, arg34_1, buf38, 32768, 256, grid=grid(32768), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        buf39 = reinterpret_tensor(buf20, (32768, 1024), (1024, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (32768, 256), (256, 1), 0), reinterpret_tensor(arg35_1, (256, 1024), (1, 256), 0), out=buf39)
        del arg35_1
        buf40 = reinterpret_tensor(buf39, (64, 512, 1024), (524288, 1024, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_13], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf40, arg36_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg36_1
        buf41 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (32768, 1024), (1024, 1), 0), reinterpret_tensor(arg37_1, (1024, 256), (1, 1024), 0), out=buf41)
        del arg37_1
        buf45 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [add_6, hidden_states_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf41, arg38_1, buf38, arg39_1, arg40_1, buf45, 32768, 256, grid=grid(32768), stream=stream0)
        del arg38_1
        del arg39_1
        del arg40_1
        buf46 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg42_1, reinterpret_tensor(buf45, (32768, 256), (256, 1), 0), reinterpret_tensor(arg41_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf46)
        del arg41_1
        del arg42_1
        buf47 = reinterpret_tensor(buf38, (32768, 256), (256, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg44_1, reinterpret_tensor(buf45, (32768, 256), (256, 1), 0), reinterpret_tensor(arg43_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf47)
        del arg43_1
        del arg44_1
        buf48 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg46_1, reinterpret_tensor(buf45, (32768, 256), (256, 1), 0), reinterpret_tensor(arg45_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf48)
        del arg45_1
        del arg46_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf49 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf46, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf47, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf48, (64, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf46
        buf50 = buf49[0]
        del buf49
        buf54 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf50, (32768, 256), (256, 1), 0), reinterpret_tensor(arg47_1, (256, 256), (1, 256), 0), out=buf54)
        del arg47_1
        buf58 = reinterpret_tensor(buf50, (64, 512, 256), (131072, 256, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [add_8, hidden_states_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf54, arg48_1, buf45, arg49_1, arg50_1, buf58, 32768, 256, grid=grid(32768), stream=stream0)
        del arg48_1
        del arg49_1
        del arg50_1
        buf59 = reinterpret_tensor(buf40, (32768, 1024), (1024, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (32768, 256), (256, 1), 0), reinterpret_tensor(arg51_1, (256, 1024), (1, 256), 0), out=buf59)
        del arg51_1
        buf60 = reinterpret_tensor(buf59, (64, 512, 1024), (524288, 1024, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf60, arg52_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg52_1
        buf61 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (32768, 1024), (1024, 1), 0), reinterpret_tensor(arg53_1, (1024, 256), (1, 1024), 0), out=buf61)
        del arg53_1
        buf65 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [add_9, hidden_states_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf61, arg54_1, buf58, arg55_1, arg56_1, buf65, 32768, 256, grid=grid(32768), stream=stream0)
        del arg54_1
        del arg55_1
        del arg56_1
        buf66 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg58_1, reinterpret_tensor(buf65, (32768, 256), (256, 1), 0), reinterpret_tensor(arg57_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf66)
        del arg57_1
        del arg58_1
        buf67 = reinterpret_tensor(buf58, (32768, 256), (256, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg60_1, reinterpret_tensor(buf65, (32768, 256), (256, 1), 0), reinterpret_tensor(arg59_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf67)
        del arg59_1
        del arg60_1
        buf68 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg62_1, reinterpret_tensor(buf65, (32768, 256), (256, 1), 0), reinterpret_tensor(arg61_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf68)
        del arg61_1
        del arg62_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf69 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf66, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf67, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf68, (64, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf66
        buf70 = buf69[0]
        del buf69
        buf74 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (32768, 256), (256, 1), 0), reinterpret_tensor(arg63_1, (256, 256), (1, 256), 0), out=buf74)
        del arg63_1
        buf78 = reinterpret_tensor(buf70, (64, 512, 256), (131072, 256, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [add_11, hidden_states_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf74, arg64_1, buf65, arg65_1, arg66_1, buf78, 32768, 256, grid=grid(32768), stream=stream0)
        del arg64_1
        del arg65_1
        del arg66_1
        buf79 = reinterpret_tensor(buf60, (32768, 1024), (1024, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (32768, 256), (256, 1), 0), reinterpret_tensor(arg67_1, (256, 1024), (1, 256), 0), out=buf79)
        del arg67_1
        buf80 = reinterpret_tensor(buf79, (64, 512, 1024), (524288, 1024, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf80, arg68_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg68_1
        buf81 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (32768, 1024), (1024, 1), 0), reinterpret_tensor(arg69_1, (1024, 256), (1, 1024), 0), out=buf81)
        del arg69_1
        buf85 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [add_12, hidden_states_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf81, arg70_1, buf78, arg71_1, arg72_1, buf85, 32768, 256, grid=grid(32768), stream=stream0)
        del arg70_1
        del arg71_1
        del arg72_1
        buf86 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg74_1, reinterpret_tensor(buf85, (32768, 256), (256, 1), 0), reinterpret_tensor(arg73_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf86)
        del arg73_1
        del arg74_1
        buf87 = reinterpret_tensor(buf78, (32768, 256), (256, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg76_1, reinterpret_tensor(buf85, (32768, 256), (256, 1), 0), reinterpret_tensor(arg75_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf87)
        del arg75_1
        del arg76_1
        buf88 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg78_1, reinterpret_tensor(buf85, (32768, 256), (256, 1), 0), reinterpret_tensor(arg77_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf88)
        del arg77_1
        del arg78_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf89 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf86, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf87, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf88, (64, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf86
        buf90 = buf89[0]
        del buf89
        buf94 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (32768, 256), (256, 1), 0), reinterpret_tensor(arg79_1, (256, 256), (1, 256), 0), out=buf94)
        del arg79_1
        buf98 = reinterpret_tensor(buf90, (64, 512, 256), (131072, 256, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [add_14, hidden_states_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf94, arg80_1, buf85, arg81_1, arg82_1, buf98, 32768, 256, grid=grid(32768), stream=stream0)
        del arg80_1
        del arg81_1
        del arg82_1
        buf99 = reinterpret_tensor(buf80, (32768, 1024), (1024, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (32768, 256), (256, 1), 0), reinterpret_tensor(arg83_1, (256, 1024), (1, 256), 0), out=buf99)
        del arg83_1
        buf100 = reinterpret_tensor(buf99, (64, 512, 1024), (524288, 1024, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf100, arg84_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg84_1
        buf101 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (32768, 1024), (1024, 1), 0), reinterpret_tensor(arg85_1, (1024, 256), (1, 1024), 0), out=buf101)
        del arg85_1
        buf105 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [add_15, hidden_states_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf101, arg86_1, buf98, arg87_1, arg88_1, buf105, 32768, 256, grid=grid(32768), stream=stream0)
        del arg86_1
        del arg87_1
        del arg88_1
        buf106 = reinterpret_tensor(buf98, (32768, 256), (256, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg90_1, reinterpret_tensor(buf105, (32768, 256), (256, 1), 0), reinterpret_tensor(arg89_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf106)
        del arg89_1
        del arg90_1
        buf107 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg92_1, reinterpret_tensor(buf105, (32768, 256), (256, 1), 0), reinterpret_tensor(arg91_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf107)
        del arg91_1
        del arg92_1
        buf108 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg94_1, reinterpret_tensor(buf105, (32768, 256), (256, 1), 0), reinterpret_tensor(arg93_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf108)
        del arg93_1
        del arg94_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf109 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf106, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf107, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf108, (64, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf106
        buf110 = buf109[0]
        del buf109
        buf114 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (32768, 256), (256, 1), 0), reinterpret_tensor(arg95_1, (256, 256), (1, 256), 0), out=buf114)
        del arg95_1
        buf118 = reinterpret_tensor(buf110, (64, 512, 256), (131072, 256, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [add_17, hidden_states_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf114, arg96_1, buf105, arg97_1, arg98_1, buf118, 32768, 256, grid=grid(32768), stream=stream0)
        del arg96_1
        del arg97_1
        del arg98_1
        buf119 = reinterpret_tensor(buf100, (32768, 1024), (1024, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (32768, 256), (256, 1), 0), reinterpret_tensor(arg99_1, (256, 1024), (1, 256), 0), out=buf119)
        del arg99_1
        buf120 = reinterpret_tensor(buf119, (64, 512, 1024), (524288, 1024, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf120, arg100_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg100_1
        buf121 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (32768, 1024), (1024, 1), 0), reinterpret_tensor(arg101_1, (1024, 256), (1, 1024), 0), out=buf121)
        del arg101_1
        buf125 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [add_18, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf121, arg102_1, buf118, arg103_1, arg104_1, buf125, 32768, 256, grid=grid(32768), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        buf126 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg106_1, reinterpret_tensor(buf125, (32768, 256), (256, 1), 0), reinterpret_tensor(arg105_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf126)
        del arg105_1
        del arg106_1
        buf127 = reinterpret_tensor(buf118, (32768, 256), (256, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg108_1, reinterpret_tensor(buf125, (32768, 256), (256, 1), 0), reinterpret_tensor(arg107_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf127)
        del arg107_1
        del arg108_1
        buf128 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg110_1, reinterpret_tensor(buf125, (32768, 256), (256, 1), 0), reinterpret_tensor(arg109_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf128)
        del arg109_1
        del arg110_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf129 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf126, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf127, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf128, (64, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf126
        buf130 = buf129[0]
        del buf129
        buf134 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (32768, 256), (256, 1), 0), reinterpret_tensor(arg111_1, (256, 256), (1, 256), 0), out=buf134)
        del arg111_1
        buf138 = reinterpret_tensor(buf130, (64, 512, 256), (131072, 256, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [add_20, hidden_states_51], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf134, arg112_1, buf125, arg113_1, arg114_1, buf138, 32768, 256, grid=grid(32768), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        buf139 = reinterpret_tensor(buf120, (32768, 1024), (1024, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (32768, 256), (256, 1), 0), reinterpret_tensor(arg115_1, (256, 1024), (1, 256), 0), out=buf139)
        del arg115_1
        buf140 = reinterpret_tensor(buf139, (64, 512, 1024), (524288, 1024, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_53], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf140, arg116_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg116_1
        buf141 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (32768, 1024), (1024, 1), 0), reinterpret_tensor(arg117_1, (1024, 256), (1, 1024), 0), out=buf141)
        del arg117_1
        buf145 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [add_21, hidden_states_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf141, arg118_1, buf138, arg119_1, arg120_1, buf145, 32768, 256, grid=grid(32768), stream=stream0)
        del arg118_1
        del arg119_1
        del arg120_1
        buf146 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg122_1, reinterpret_tensor(buf145, (32768, 256), (256, 1), 0), reinterpret_tensor(arg121_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf146)
        del arg121_1
        del arg122_1
        buf147 = reinterpret_tensor(buf138, (32768, 256), (256, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg124_1, reinterpret_tensor(buf145, (32768, 256), (256, 1), 0), reinterpret_tensor(arg123_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf147)
        del arg123_1
        del arg124_1
        buf148 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg126_1, reinterpret_tensor(buf145, (32768, 256), (256, 1), 0), reinterpret_tensor(arg125_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf148)
        del arg125_1
        del arg126_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf149 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf146, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf147, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf148, (64, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf146
        buf150 = buf149[0]
        del buf149
        buf154 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf150, (32768, 256), (256, 1), 0), reinterpret_tensor(arg127_1, (256, 256), (1, 256), 0), out=buf154)
        del arg127_1
        buf158 = reinterpret_tensor(buf150, (64, 512, 256), (131072, 256, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [add_23, hidden_states_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf154, arg128_1, buf145, arg129_1, arg130_1, buf158, 32768, 256, grid=grid(32768), stream=stream0)
        del arg128_1
        del arg129_1
        del arg130_1
        buf159 = reinterpret_tensor(buf140, (32768, 1024), (1024, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (32768, 256), (256, 1), 0), reinterpret_tensor(arg131_1, (256, 1024), (1, 256), 0), out=buf159)
        del arg131_1
        buf160 = reinterpret_tensor(buf159, (64, 512, 1024), (524288, 1024, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf160, arg132_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg132_1
        buf161 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (32768, 1024), (1024, 1), 0), reinterpret_tensor(arg133_1, (1024, 256), (1, 1024), 0), out=buf161)
        del arg133_1
        buf165 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [add_24, hidden_states_64], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf161, arg134_1, buf158, arg135_1, arg136_1, buf165, 32768, 256, grid=grid(32768), stream=stream0)
        del arg134_1
        del arg135_1
        del arg136_1
        buf166 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg138_1, reinterpret_tensor(buf165, (32768, 256), (256, 1), 0), reinterpret_tensor(arg137_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf166)
        del arg137_1
        del arg138_1
        buf167 = reinterpret_tensor(buf158, (32768, 256), (256, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg140_1, reinterpret_tensor(buf165, (32768, 256), (256, 1), 0), reinterpret_tensor(arg139_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf167)
        del arg139_1
        del arg140_1
        buf168 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg142_1, reinterpret_tensor(buf165, (32768, 256), (256, 1), 0), reinterpret_tensor(arg141_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf168)
        del arg141_1
        del arg142_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf169 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf166, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf167, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf168, (64, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf166
        buf170 = buf169[0]
        del buf169
        buf174 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (32768, 256), (256, 1), 0), reinterpret_tensor(arg143_1, (256, 256), (1, 256), 0), out=buf174)
        del arg143_1
        buf178 = reinterpret_tensor(buf170, (64, 512, 256), (131072, 256, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [add_26, hidden_states_67], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf174, arg144_1, buf165, arg145_1, arg146_1, buf178, 32768, 256, grid=grid(32768), stream=stream0)
        del arg144_1
        del arg145_1
        del arg146_1
        buf179 = reinterpret_tensor(buf160, (32768, 1024), (1024, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (32768, 256), (256, 1), 0), reinterpret_tensor(arg147_1, (256, 1024), (1, 256), 0), out=buf179)
        del arg147_1
        buf180 = reinterpret_tensor(buf179, (64, 512, 1024), (524288, 1024, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf180, arg148_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg148_1
        buf181 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (32768, 1024), (1024, 1), 0), reinterpret_tensor(arg149_1, (1024, 256), (1, 1024), 0), out=buf181)
        del arg149_1
        buf185 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [add_27, hidden_states_72], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf181, arg150_1, buf178, arg151_1, arg152_1, buf185, 32768, 256, grid=grid(32768), stream=stream0)
        del arg150_1
        del arg151_1
        del arg152_1
        buf186 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg154_1, reinterpret_tensor(buf185, (32768, 256), (256, 1), 0), reinterpret_tensor(arg153_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf186)
        del arg153_1
        del arg154_1
        buf187 = reinterpret_tensor(buf178, (32768, 256), (256, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg156_1, reinterpret_tensor(buf185, (32768, 256), (256, 1), 0), reinterpret_tensor(arg155_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf187)
        del arg155_1
        del arg156_1
        buf188 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg158_1, reinterpret_tensor(buf185, (32768, 256), (256, 1), 0), reinterpret_tensor(arg157_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf188)
        del arg157_1
        del arg158_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf189 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf186, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf187, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf188, (64, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf186
        buf190 = buf189[0]
        del buf189
        buf194 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (32768, 256), (256, 1), 0), reinterpret_tensor(arg159_1, (256, 256), (1, 256), 0), out=buf194)
        del arg159_1
        buf198 = reinterpret_tensor(buf190, (64, 512, 256), (131072, 256, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [add_29, hidden_states_75], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf194, arg160_1, buf185, arg161_1, arg162_1, buf198, 32768, 256, grid=grid(32768), stream=stream0)
        del arg160_1
        del arg161_1
        del arg162_1
        buf199 = reinterpret_tensor(buf180, (32768, 1024), (1024, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (32768, 256), (256, 1), 0), reinterpret_tensor(arg163_1, (256, 1024), (1, 256), 0), out=buf199)
        del arg163_1
        buf200 = reinterpret_tensor(buf199, (64, 512, 1024), (524288, 1024, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf200, arg164_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg164_1
        buf201 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (32768, 1024), (1024, 1), 0), reinterpret_tensor(arg165_1, (1024, 256), (1, 1024), 0), out=buf201)
        del arg165_1
        buf205 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [add_30, hidden_states_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf201, arg166_1, buf198, arg167_1, arg168_1, buf205, 32768, 256, grid=grid(32768), stream=stream0)
        del arg166_1
        del arg167_1
        del arg168_1
        buf206 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg170_1, reinterpret_tensor(buf205, (32768, 256), (256, 1), 0), reinterpret_tensor(arg169_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf206)
        del arg169_1
        del arg170_1
        buf207 = reinterpret_tensor(buf198, (32768, 256), (256, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg172_1, reinterpret_tensor(buf205, (32768, 256), (256, 1), 0), reinterpret_tensor(arg171_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf207)
        del arg171_1
        del arg172_1
        buf208 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg174_1, reinterpret_tensor(buf205, (32768, 256), (256, 1), 0), reinterpret_tensor(arg173_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf208)
        del arg173_1
        del arg174_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf209 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf206, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf207, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf208, (64, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf206
        buf210 = buf209[0]
        del buf209
        buf214 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (32768, 256), (256, 1), 0), reinterpret_tensor(arg175_1, (256, 256), (1, 256), 0), out=buf214)
        del arg175_1
        buf218 = reinterpret_tensor(buf210, (64, 512, 256), (131072, 256, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [add_32, hidden_states_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf214, arg176_1, buf205, arg177_1, arg178_1, buf218, 32768, 256, grid=grid(32768), stream=stream0)
        del arg176_1
        del arg177_1
        del arg178_1
        buf219 = reinterpret_tensor(buf200, (32768, 1024), (1024, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (32768, 256), (256, 1), 0), reinterpret_tensor(arg179_1, (256, 1024), (1, 256), 0), out=buf219)
        del arg179_1
        buf220 = reinterpret_tensor(buf219, (64, 512, 1024), (524288, 1024, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf220, arg180_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg180_1
        buf221 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (32768, 1024), (1024, 1), 0), reinterpret_tensor(arg181_1, (1024, 256), (1, 1024), 0), out=buf221)
        del arg181_1
        buf225 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [add_33, hidden_states_88], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf221, arg182_1, buf218, arg183_1, arg184_1, buf225, 32768, 256, grid=grid(32768), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        buf226 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg186_1, reinterpret_tensor(buf225, (32768, 256), (256, 1), 0), reinterpret_tensor(arg185_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf226)
        del arg185_1
        del arg186_1
        buf227 = reinterpret_tensor(buf218, (32768, 256), (256, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg188_1, reinterpret_tensor(buf225, (32768, 256), (256, 1), 0), reinterpret_tensor(arg187_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf227)
        del arg187_1
        del arg188_1
        buf228 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [linear_69], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg190_1, reinterpret_tensor(buf225, (32768, 256), (256, 1), 0), reinterpret_tensor(arg189_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf228)
        del arg189_1
        del arg190_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf229 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf226, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf227, (64, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf228, (64, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf226
        del buf227
        buf230 = buf229[0]
        del buf229
        buf234 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (32768, 256), (256, 1), 0), reinterpret_tensor(arg191_1, (256, 256), (1, 256), 0), out=buf234)
        del arg191_1
        buf238 = reinterpret_tensor(buf230, (64, 512, 256), (131072, 256, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [add_35, hidden_states_91], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf234, arg192_1, buf225, arg193_1, arg194_1, buf238, 32768, 256, grid=grid(32768), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        buf239 = reinterpret_tensor(buf220, (32768, 1024), (1024, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (32768, 256), (256, 1), 0), reinterpret_tensor(arg195_1, (256, 1024), (1, 256), 0), out=buf239)
        del arg195_1
        buf240 = reinterpret_tensor(buf239, (64, 512, 1024), (524288, 1024, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf240, arg196_1, 33554432, grid=grid(33554432), stream=stream0)
        del arg196_1
        buf241 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (32768, 1024), (1024, 1), 0), reinterpret_tensor(arg197_1, (1024, 256), (1, 1024), 0), out=buf241)
        del arg197_1
        del buf240
        buf245 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [add_36, hidden_states_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf241, arg198_1, buf238, arg199_1, arg200_1, buf245, 32768, 256, grid=grid(32768), stream=stream0)
        del arg198_1
        del arg199_1
        del arg200_1
        del buf238
        del buf241
        buf246 = empty_strided_cuda((32768, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (32768, 256), (256, 1), 0), reinterpret_tensor(arg202_1, (256, 2), (1, 256), 0), out=buf246)
        del arg202_1
        del buf245
        buf247 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        buf248 = empty_strided_cuda((64, 1), (1, 64), torch.float32)
        buf249 = empty_strided_cuda((64, 1), (1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [start_logits_1, start_loss], Original ATen: [aten.clone, aten._log_softmax]
        triton_red_fused__log_softmax_clone_3.run(buf246, arg203_1, buf247, buf248, buf249, 64, 512, grid=grid(64), stream=stream0)
        buf252 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        buf253 = empty_strided_cuda((64, 1), (1, 64), torch.float32)
        buf254 = empty_strided_cuda((64, 1), (1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [end_logits_1, end_loss], Original ATen: [aten.clone, aten._log_softmax]
        triton_red_fused__log_softmax_clone_4.run(buf246, arg203_1, buf252, buf253, buf254, 64, 512, grid=grid(64), stream=stream0)
        del arg203_1
        del buf246
        buf250 = empty_strided_cuda((), (), torch.float32)
        buf257 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [start_positions, start_loss, end_positions, end_loss, add_37, total_loss], Original ATen: [aten.clamp, aten.nll_loss_forward, aten.add, aten.div]
        triton_per_fused_add_clamp_div_nll_loss_forward_5.run(buf257, arg204_1, buf247, buf248, buf249, arg205_1, buf252, buf253, buf254, 1, 64, grid=grid(1), stream=stream0)
        del arg204_1
        del arg205_1
        del buf248
        del buf249
        del buf253
        del buf254
    return (buf257, buf247, buf252, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((30522, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg202_1 = rand_strided((2, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg205_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ElectraForQuestionAnswering', benchmark_compiled_module)
