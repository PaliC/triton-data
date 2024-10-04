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


# kernel path: /tmp/torchinductor_sahanp/bt/cbtjvtlbz6hnt3dbvzbkq3cuccmmnx3lg5wm2inwqpspw5zh3eiz.py
# Topologically Sorted Source Nodes: [inputs_embeds_1, pow_14, variance_13, add_28, rsqrt_13, hidden_states_65, normed_hidden_states_6], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_28 => add_35
#   hidden_states_65 => mul_32
#   inputs_embeds_1 => embedding_2
#   normed_hidden_states_6 => mul_33
#   pow_14 => pow_14
#   rsqrt_13 => rsqrt_13
#   variance_13 => mean_13
# Graph fragment:
#   %embedding_2 : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view_145), kwargs = {})
#   %pow_14 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%embedding_2, 2), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_14, [-1], True), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_13, 1e-06), kwargs = {})
#   %rsqrt_13 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding_2, %rsqrt_13), kwargs = {})
#   %mul_33 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg58_1, %mul_32), kwargs = {})
triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tl.full([XBLOCK, RBLOCK], 32128, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert((0 <= tmp4) & (tmp4 < 32128), "index out of bounds: 0 <= tmp4 < 32128")
        tmp6 = tl.load(in_ptr1 + (r1 + (512*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp6 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full([XBLOCK, RBLOCK], 32128, tl.int32)
        tmp13 = tmp0 + tmp12
        tmp14 = tmp0 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp0)
        tl.device_assert((0 <= tmp15) & (tmp15 < 32128), "index out of bounds: 0 <= tmp15 < 32128")
        tmp17 = tl.load(in_ptr1 + (r1 + (512*tmp15)), rmask, eviction_policy='evict_first', other=0.0)
        tmp18 = 512.0
        tmp19 = tmp9 / tmp18
        tmp20 = 1e-06
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp24 = tmp11 * tmp23
        tl.store(out_ptr1 + (r1 + (512*x0)), tmp24, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ze/czebwambxotwfzrdknrb4d7rglxopbv73pump2ynjt7vxde63sch.py
# Topologically Sorted Source Nodes: [scores_12], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   scores_12 => clone_51
# Graph fragment:
#   %clone_51 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_24,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536) % 8
    x3 = (xindex // 524288)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (512*x1) + (524288*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qj/cqjvwqllkjcbr2yyhgfy2aox2nqtc3svivg7tkcxhicpzoqcsmbu.py
# Topologically Sorted Source Nodes: [scores_12], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   scores_12 => clone_52
# Graph fragment:
#   %clone_52 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_25,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4t/c4tibahtqqbgperwfse3s67ke6dguqs426vuiynl7qnjhaqza245.py
# Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_13, softmax_6], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   extended_attention_mask_5 => mul_30
#   position_bias_1 => add_38
#   scores_13 => add_39
#   softmax_6 => amax_6, div_10, exp_6, sub_11, sum_7
#   sub_2 => sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %unsqueeze_9), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, -3.4028234663852886e+38), kwargs = {})
#   %add_38 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_16, %mul_30), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_157, %add_38), kwargs = {})
#   %amax_6 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_159, [-1], True), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_159, %amax_6), kwargs = {})
#   %exp_6 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_11,), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_6, [-1], True), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_6, %sum_7), kwargs = {})
triton_per_fused__softmax_add_mul_rsub_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_rsub_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r3 = rindex
    x4 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 8
    tmp0 = tl.load(in_out_ptr0 + (r3 + (1024*x4)), None)
    tmp1 = (-1)*((0) * ((0) <= (r3 + ((-1)*x0))) + (r3 + ((-1)*x0)) * ((r3 + ((-1)*x0)) < (0)))
    tmp2 = tl.full([1], 16, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tmp1.to(tl.float32)
    tmp5 = 0.0625
    tmp6 = tmp4 * tmp5
    tmp7 = tl_math.log(tmp6)
    tmp8 = 0.48089834696298783
    tmp9 = tmp7 * tmp8
    tmp10 = 16.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.int64)
    tmp13 = tmp12 + tmp2
    tmp14 = tl.full([1], 31, tl.int64)
    tmp15 = triton_helpers.minimum(tmp13, tmp14)
    tmp16 = tl.where(tmp3, tmp1, tmp15)
    tmp17 = tl.full([1], 0, tl.int64)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([RBLOCK], 32, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 32), "index out of bounds: 0 <= tmp22 < 32")
    tmp24 = tl.load(in_ptr0 + (x1 + (8*tmp22)), None, eviction_policy='evict_last')
    tmp25 = r3
    tmp26 = x0
    tmp27 = tmp25 <= tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = 1.0
    tmp30 = tmp29 - tmp28
    tmp31 = -3.4028234663852886e+38
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 + tmp32
    tmp34 = tmp0 + tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp35, 0))
    tmp38 = tmp34 - tmp37
    tmp39 = tl_math.exp(tmp38)
    tmp40 = tl.broadcast_to(tmp39, [RBLOCK])
    tmp42 = triton_helpers.promote_to_tensor(tl.sum(tmp40, 0))
    tmp43 = tmp39 / tmp42
    tl.store(out_ptr2 + (r3 + (1024*x4)), tmp43, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3e/c3egjdcz2fl6iughxup6bgzrym7bvts2fu26fg4jxysu4slrlndz.py
# Topologically Sorted Source Nodes: [contiguous_6], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_6 => clone_55
# Graph fragment:
#   %clone_55 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_75,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 8
    x2 = (xindex // 512) % 1024
    x3 = (xindex // 524288)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (65536*x1) + (524288*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cc/cccjzmf24tjbvt35wkgmjbbij5gbnf7gcugftzte7zvvxfpda3rr.py
# Topologically Sorted Source Nodes: [inputs_embeds_1, hidden_states_68, pow_15, variance_14, add_33, rsqrt_14, hidden_states_69, normed_hidden_states_7, inputs_embeds, pow_1, variance, add, rsqrt, hidden_states_1, normed_hidden_states], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_33 => add_41
#   hidden_states_1 => mul_1
#   hidden_states_68 => add_40
#   hidden_states_69 => mul_35
#   inputs_embeds => embedding
#   inputs_embeds_1 => embedding_2
#   normed_hidden_states => mul_2
#   normed_hidden_states_7 => mul_36
#   pow_1 => pow_1
#   pow_15 => pow_15
#   rsqrt => rsqrt
#   rsqrt_14 => rsqrt_14
#   variance => mean
#   variance_14 => mean_14
# Graph fragment:
#   %embedding_2 : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view_145), kwargs = {})
#   %add_40 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding_2, %view_165), kwargs = {})
#   %pow_15 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_40, 2), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_15, [-1], True), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_14, 1e-06), kwargs = {})
#   %rsqrt_14 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_41,), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_40, %rsqrt_14), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg63_1, %mul_35), kwargs = {})
#   %embedding : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%embedding, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg7_1, %mul_1), kwargs = {})
triton_red_fused_add_embedding_mean_mul_pow_rsqrt_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 32128, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert((0 <= tmp4) & (tmp4 < 32128), "index out of bounds: 0 <= tmp4 < 32128")
        tmp6 = tl.load(in_ptr1 + (r1 + (512*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
        tmp13 = tmp6 * tmp6
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp17 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.full([XBLOCK, RBLOCK], 32128, tl.int32)
        tmp19 = tmp0 + tmp18
        tmp20 = tmp0 < 0
        tmp21 = tl.where(tmp20, tmp19, tmp0)
        tl.device_assert((0 <= tmp21) & (tmp21 < 32128), "index out of bounds: 0 <= tmp21 < 32128")
        tmp23 = tl.load(in_ptr1 + (r1 + (512*tmp21)), rmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tmp23 + tmp24
        tmp26 = 512.0
        tmp27 = tmp11 / tmp26
        tmp28 = 1e-06
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp32 = tmp17 * tmp31
        tmp34 = tmp15 / tmp26
        tmp35 = tmp34 + tmp28
        tmp36 = libdevice.rsqrt(tmp35)
        tmp37 = tmp23 * tmp36
        tmp38 = tmp33 * tmp37
        tl.store(out_ptr2 + (r1 + (512*x0)), tmp32, rmask)
        tl.store(out_ptr3 + (r1 + (512*x0)), tmp38, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zb/czbrxvhiqvaotd5psu55arkl4l5ek7i5we5i4e67jiv5ql3qifmk.py
# Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_1, softmax], Original ATen: [aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   extended_attention_mask_2 => full_default
#   position_bias => add_4
#   scores_1 => add_5
#   softmax => amax, div_2, exp, sub_2, sum_1
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 1, 1, 1024], -0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_4 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_4, %full_default), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_12, %add_4), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_14, [-1], True), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_14, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_per_fused__softmax_add_mul_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r3 = rindex
    x4 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 8
    tmp0 = tl.load(in_out_ptr0 + (r3 + (1024*x4)), None)
    tmp1 = r3 + ((-1)*x0)
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 > tmp2
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.full([1], 16, tl.int64)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6 + tmp2
    tmp8 = tl_math.abs(r3 + ((-1)*x0))
    tmp9 = tl.full([1], 8, tl.int64)
    tmp10 = tmp8 < tmp9
    tmp11 = tmp8.to(tl.float32)
    tmp12 = 0.125
    tmp13 = tmp11 * tmp12
    tmp14 = tl_math.log(tmp13)
    tmp15 = 0.36067376022224085
    tmp16 = tmp14 * tmp15
    tmp17 = 8.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.int64)
    tmp20 = tmp19 + tmp9
    tmp21 = tl.full([1], 15, tl.int64)
    tmp22 = triton_helpers.minimum(tmp20, tmp21)
    tmp23 = tl.where(tmp10, tmp8, tmp22)
    tmp24 = tmp7 + tmp23
    tmp25 = tl.full([RBLOCK], 32, tl.int32)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp24 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp24)
    tl.device_assert((0 <= tmp28) & (tmp28 < 32), "index out of bounds: 0 <= tmp28 < 32")
    tmp30 = tl.load(in_ptr0 + (x1 + (8*tmp28)), None, eviction_policy='evict_last')
    tmp31 = -0.0
    tmp32 = tmp30 + tmp31
    tmp33 = tmp0 + tmp32
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp34, 0))
    tmp37 = tmp33 - tmp36
    tmp38 = tl_math.exp(tmp37)
    tmp39 = tl.broadcast_to(tmp38, [RBLOCK])
    tmp41 = triton_helpers.promote_to_tensor(tl.sum(tmp39, 0))
    tmp42 = tmp38 / tmp41
    tl.store(out_ptr2 + (r3 + (1024*x4)), tmp42, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/44/c44fmilqllnquzrfglagkp5bgnoyrmxdpqorddrzpjk2do6gtiph.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_4, pow_2, variance_1, add_5, rsqrt_1, hidden_states_5, forwarded_states], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_5 => add_7
#   forwarded_states => mul_6
#   hidden_states_4 => add_6
#   hidden_states_5 => mul_5
#   inputs_embeds => embedding
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
#   variance_1 => mean_1
# Graph fragment:
#   %embedding : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_20), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_6, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %rsqrt_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg10_1, %mul_5), kwargs = {})
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), None)
    tmp13 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.full([RBLOCK], 32128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32128), "index out of bounds: 0 <= tmp4 < 32128")
    tmp6 = tl.load(in_ptr1 + (r1 + (512*tmp4)), None)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp14 = 512.0
    tmp15 = tmp12 / tmp14
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp8 * tmp18
    tmp20 = tmp13 * tmp19
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5s/c5szzlruz2r267e4f6l4dqc7mbsuwnbmrap4b26wusddh5llir5n.py
# Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   hidden_states_7 => relu
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_22,), kwargs = {})
triton_poi_fused_relu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/q6/cq63zv26ztwo27ayb3ophcx4ilyhuhm74p7fo63malsgduby2o2m.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_4, hidden_states_10, pow_3, variance_2, add_7, rsqrt_2, hidden_states_11, normed_hidden_states_1], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_7 => add_9
#   hidden_states_10 => add_8
#   hidden_states_11 => mul_7
#   hidden_states_4 => add_6
#   inputs_embeds => embedding
#   normed_hidden_states_1 => mul_8
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
#   variance_2 => mean_2
# Graph fragment:
#   %embedding : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_20), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_24), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_8, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %rsqrt_2), kwargs = {})
#   %mul_8 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg15_1, %mul_7), kwargs = {})
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_9', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), None)
    tmp9 = tl.load(in_ptr3 + (r1 + (512*x0)), None)
    tmp15 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.full([RBLOCK], 32128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32128), "index out of bounds: 0 <= tmp4 < 32128")
    tmp6 = tl.load(in_ptr1 + (r1 + (512*tmp4)), None)
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp16 = 512.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp10 * tmp20
    tmp22 = tmp15 * tmp21
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nw/cnwpyo42qbnt5mnzq5h4o4plczjnzjljzp57trs7nvrjec36lkgj.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_4, hidden_states_10, hidden_states_14, pow_4, variance_3, add_9, rsqrt_3, hidden_states_15, forwarded_states_1], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_9 => add_12
#   forwarded_states_1 => mul_10
#   hidden_states_10 => add_8
#   hidden_states_14 => add_11
#   hidden_states_15 => mul_9
#   hidden_states_4 => add_6
#   inputs_embeds => embedding
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
#   variance_3 => mean_3
# Graph fragment:
#   %embedding : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_20), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_24), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_44), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_11, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %rsqrt_3), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg18_1, %mul_9), kwargs = {})
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (r1 + (512*x0)), None)
    tmp9 = tl.load(in_ptr2 + (r1 + (512*x0)), None)
    tmp11 = tl.load(in_ptr3 + (r1 + (512*x0)), None)
    tmp17 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.full([RBLOCK], 32128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32128), "index out of bounds: 0 <= tmp4 < 32128")
    tmp6 = tl.load(in_ptr1 + (r1 + (512*tmp4)), None)
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp18 = 512.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp12 * tmp22
    tmp24 = tmp17 * tmp23
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp12, None)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qy/cqymdijviyo42sljjv4dtifks2qv4was77jghtalxuq65wjwbeqk.py
# Topologically Sorted Source Nodes: [hidden_states_20, pow_5, variance_4, add_11, rsqrt_4, hidden_states_21, normed_hidden_states_2], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_11 => add_14
#   hidden_states_20 => add_13
#   hidden_states_21 => mul_11
#   normed_hidden_states_2 => mul_12
#   pow_5 => pow_5
#   rsqrt_4 => rsqrt_4
#   variance_4 => mean_4
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_48), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_13, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %rsqrt_4), kwargs = {})
#   %mul_12 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg23_1, %mul_11), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), None)
    tmp7 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp8 = 512.0
    tmp9 = tmp6 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp2 * tmp12
    tmp14 = tmp7 * tmp13
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2m/c2mcn7awhzqwckrhuj2hicmpflxwnzuqyyun25zyb2dxtgnm7q4a.py
# Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_24, pow_6, variance_5, add_13, rsqrt_5, hidden_states_25, forwarded_states_2], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_13 => add_17
#   forwarded_states_2 => mul_14
#   hidden_states_20 => add_13
#   hidden_states_24 => add_16
#   hidden_states_25 => mul_13
#   pow_6 => pow_6
#   rsqrt_5 => rsqrt_5
#   variance_5 => mean_5
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_48), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_68), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_16, 2), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_6, [-1], True), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, 1e-06), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %rsqrt_5), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg26_1, %mul_13), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), None)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), None)
    tmp9 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = 512.0
    tmp11 = tmp8 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp4 * tmp14
    tmp16 = tmp9 * tmp15
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6i/c6ijbhd6e3sq3svrlm2ngzqpst6aexztnp5iad4ccvg5p7ftbg4g.py
# Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_24, hidden_states_30, pow_7, variance_6, add_15, rsqrt_6, hidden_states_31, normed_hidden_states_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_15 => add_19
#   hidden_states_20 => add_13
#   hidden_states_24 => add_16
#   hidden_states_30 => add_18
#   hidden_states_31 => mul_15
#   normed_hidden_states_3 => mul_16
#   pow_7 => pow_7
#   rsqrt_6 => rsqrt_6
#   variance_6 => mean_6
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_48), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_68), kwargs = {})
#   %add_18 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_72), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_18, 2), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [-1], True), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_6, 1e-06), kwargs = {})
#   %rsqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %rsqrt_6), kwargs = {})
#   %mul_16 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg31_1, %mul_15), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_13', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), None)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), None)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), None)
    tmp11 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp12 = 512.0
    tmp13 = tmp10 / tmp12
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp6 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp18, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ip/cip4skdzzox6sm5uzyahk35iwvibhmcfhv4ifuazp4nrpmh32xob.py
# Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_24, hidden_states_30, hidden_states_34, pow_8, variance_7, add_17, rsqrt_7, hidden_states_35, forwarded_states_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_17 => add_22
#   forwarded_states_3 => mul_18
#   hidden_states_20 => add_13
#   hidden_states_24 => add_16
#   hidden_states_30 => add_18
#   hidden_states_34 => add_21
#   hidden_states_35 => mul_17
#   pow_8 => pow_8
#   rsqrt_7 => rsqrt_7
#   variance_7 => mean_7
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_48), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_68), kwargs = {})
#   %add_18 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_72), kwargs = {})
#   %add_21 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %view_92), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_21, 2), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [-1], True), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_7, 1e-06), kwargs = {})
#   %rsqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %rsqrt_7), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg34_1, %mul_17), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_14', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), None)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), None)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), None)
    tmp13 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp14 = 512.0
    tmp15 = tmp12 / tmp14
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp8 * tmp18
    tmp20 = tmp13 * tmp19
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, None)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vn/cvnewziagl6d2xvxr4au7rxnrf3h2mo357e3ae3wawbpr22ibfmi.py
# Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_44, hidden_states_50, hidden_states_54, pow_12, variance_11, add_25, rsqrt_11, hidden_states_55, forwarded_states_5], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_25 => add_32
#   forwarded_states_5 => mul_26
#   hidden_states_40 => add_23
#   hidden_states_44 => add_26
#   hidden_states_50 => add_28
#   hidden_states_54 => add_31
#   hidden_states_55 => mul_25
#   pow_12 => pow_12
#   rsqrt_11 => rsqrt_11
#   variance_11 => mean_11
# Graph fragment:
#   %add_23 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_96), kwargs = {})
#   %add_26 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %view_116), kwargs = {})
#   %add_28 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %view_120), kwargs = {})
#   %add_31 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %view_140), kwargs = {})
#   %pow_12 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_31, 2), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_12, [-1], True), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_11, 1e-06), kwargs = {})
#   %rsqrt_11 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_32,), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_31, %rsqrt_11), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg50_1, %mul_25), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_15', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_out_ptr0 + (r1 + (512*x0)), None)
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), None)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), None)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), None)
    tmp13 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp14 = 512.0
    tmp15 = tmp12 / tmp14
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp8 * tmp18
    tmp20 = tmp13 * tmp19
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, None)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/w3/cw3rmtptd2kut4s43ic35xzjxz7lec7z2ee5ebb2nicoku7czd2c.py
# Topologically Sorted Source Nodes: [softmax_7], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   softmax_7 => amax_7, div_11, exp_7, sub_12, sum_8
# Graph fragment:
#   %amax_7 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_179, [-1], True), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_179, %amax_7), kwargs = {})
#   %exp_7 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_12,), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_7, [-1], True), kwargs = {})
#   %div_11 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_7, %sum_8), kwargs = {})
triton_per_fused__softmax_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_16', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 32768
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
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp1, 0))
    tmp4 = tmp0 - tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tmp5 / tmp8
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uo/cuo4pbljd6kd4itsgzbzoafotg55rtaavqo7xtkqmnvajrgt32bz.py
# Topologically Sorted Source Nodes: [hidden_states_133, layer_output_5, hidden_states_142, pow_32, variance_31, add_68, rsqrt_31, hidden_states_143, hidden_states_144, sequence_output], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_68 => add_87
#   hidden_states_133 => add_81
#   hidden_states_142 => add_86
#   hidden_states_143 => mul_69
#   hidden_states_144 => mul_70
#   layer_output_5 => add_84
#   pow_32 => pow_32
#   rsqrt_31 => rsqrt_31
#   sequence_output => mul_71
#   variance_31 => mean_31
# Graph fragment:
#   %add_81 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_78, %view_385), kwargs = {})
#   %add_84 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_81, %view_405), kwargs = {})
#   %add_86 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_84, %view_409), kwargs = {})
#   %pow_32 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_86, 2), kwargs = {})
#   %mean_31 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_32, [-1], True), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_31, 1e-06), kwargs = {})
#   %rsqrt_31 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_87,), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_86, %rsqrt_31), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg132_1, %mul_69), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, 0.04419417382415922), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_17', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), None)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), None)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), None)
    tmp11 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp12 = 512.0
    tmp13 = tmp10 / tmp12
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp6 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = 0.04419417382415922
    tmp20 = tmp18 * tmp19
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eb/cebfjafv6ubtfpfrjm2nbadwqcg3wmnlhdrnsnh77ufye5fslof7.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   loss => amax_18, exp_18, sub_23, sum_19
# Graph fragment:
#   %amax_18 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_412, [1], True), kwargs = {})
#   %sub_23 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_412, %amax_18), kwargs = {})
#   %exp_18 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_23,), kwargs = {})
#   %sum_19 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_18, [1], True), kwargs = {})
triton_red_fused__log_softmax_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_18', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 32128
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
        tmp0 = tl.load(in_ptr0 + (r1 + (32128*x0)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (32128*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hm/chmd7436j7n7rlkzeyx2y7dfu7vvvjakgriufftcdgcrg6waebit.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type_7, div_22, full_default_7, ne_1, ne_2, neg_1, sum_20, sum_21, where_3
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_413, -100), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg_1, %full_default_7), kwargs = {})
#   %sum_21 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_3,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_413, -100), kwargs = {})
#   %sum_20 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_20, torch.float32), kwargs = {})
#   %div_22 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_21, %convert_element_type_7), kwargs = {})
triton_red_fused_nll_loss_forward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {5: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_19', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
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
        tmp5 = tl.full([XBLOCK, RBLOCK], 32128, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 32128)) | ~(rmask), "index out of bounds: 0 <= tmp8 < 32128")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (32128*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 1024), (1024, 1))
    assert_size_stride(arg1_1, (32128, 512), (512, 1))
    assert_size_stride(arg2_1, (512, 512), (512, 1))
    assert_size_stride(arg3_1, (512, 512), (512, 1))
    assert_size_stride(arg4_1, (512, 512), (512, 1))
    assert_size_stride(arg5_1, (512, 512), (512, 1))
    assert_size_stride(arg6_1, (32, 8), (8, 1))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (2048, 512), (512, 1))
    assert_size_stride(arg9_1, (512, 2048), (2048, 1))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (512, 512), (512, 1))
    assert_size_stride(arg12_1, (512, 512), (512, 1))
    assert_size_stride(arg13_1, (512, 512), (512, 1))
    assert_size_stride(arg14_1, (512, 512), (512, 1))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (2048, 512), (512, 1))
    assert_size_stride(arg17_1, (512, 2048), (2048, 1))
    assert_size_stride(arg18_1, (512, ), (1, ))
    assert_size_stride(arg19_1, (512, 512), (512, 1))
    assert_size_stride(arg20_1, (512, 512), (512, 1))
    assert_size_stride(arg21_1, (512, 512), (512, 1))
    assert_size_stride(arg22_1, (512, 512), (512, 1))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (2048, 512), (512, 1))
    assert_size_stride(arg25_1, (512, 2048), (2048, 1))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, 512), (512, 1))
    assert_size_stride(arg28_1, (512, 512), (512, 1))
    assert_size_stride(arg29_1, (512, 512), (512, 1))
    assert_size_stride(arg30_1, (512, 512), (512, 1))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (2048, 512), (512, 1))
    assert_size_stride(arg33_1, (512, 2048), (2048, 1))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, 512), (512, 1))
    assert_size_stride(arg36_1, (512, 512), (512, 1))
    assert_size_stride(arg37_1, (512, 512), (512, 1))
    assert_size_stride(arg38_1, (512, 512), (512, 1))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (2048, 512), (512, 1))
    assert_size_stride(arg41_1, (512, 2048), (2048, 1))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (512, 512), (512, 1))
    assert_size_stride(arg44_1, (512, 512), (512, 1))
    assert_size_stride(arg45_1, (512, 512), (512, 1))
    assert_size_stride(arg46_1, (512, 512), (512, 1))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (2048, 512), (512, 1))
    assert_size_stride(arg49_1, (512, 2048), (2048, 1))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (4, 1024), (1024, 1))
    assert_size_stride(arg53_1, (512, 512), (512, 1))
    assert_size_stride(arg54_1, (512, 512), (512, 1))
    assert_size_stride(arg55_1, (512, 512), (512, 1))
    assert_size_stride(arg56_1, (512, 512), (512, 1))
    assert_size_stride(arg57_1, (32, 8), (8, 1))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (512, 512), (512, 1))
    assert_size_stride(arg60_1, (512, 512), (512, 1))
    assert_size_stride(arg61_1, (512, 512), (512, 1))
    assert_size_stride(arg62_1, (512, 512), (512, 1))
    assert_size_stride(arg63_1, (512, ), (1, ))
    assert_size_stride(arg64_1, (2048, 512), (512, 1))
    assert_size_stride(arg65_1, (512, 2048), (2048, 1))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (512, 512), (512, 1))
    assert_size_stride(arg68_1, (512, 512), (512, 1))
    assert_size_stride(arg69_1, (512, 512), (512, 1))
    assert_size_stride(arg70_1, (512, 512), (512, 1))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (512, 512), (512, 1))
    assert_size_stride(arg73_1, (512, 512), (512, 1))
    assert_size_stride(arg74_1, (512, 512), (512, 1))
    assert_size_stride(arg75_1, (512, 512), (512, 1))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (2048, 512), (512, 1))
    assert_size_stride(arg78_1, (512, 2048), (2048, 1))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (512, 512), (512, 1))
    assert_size_stride(arg81_1, (512, 512), (512, 1))
    assert_size_stride(arg82_1, (512, 512), (512, 1))
    assert_size_stride(arg83_1, (512, 512), (512, 1))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, 512), (512, 1))
    assert_size_stride(arg86_1, (512, 512), (512, 1))
    assert_size_stride(arg87_1, (512, 512), (512, 1))
    assert_size_stride(arg88_1, (512, 512), (512, 1))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (2048, 512), (512, 1))
    assert_size_stride(arg91_1, (512, 2048), (2048, 1))
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, 512), (512, 1))
    assert_size_stride(arg94_1, (512, 512), (512, 1))
    assert_size_stride(arg95_1, (512, 512), (512, 1))
    assert_size_stride(arg96_1, (512, 512), (512, 1))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (512, 512), (512, 1))
    assert_size_stride(arg99_1, (512, 512), (512, 1))
    assert_size_stride(arg100_1, (512, 512), (512, 1))
    assert_size_stride(arg101_1, (512, 512), (512, 1))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (2048, 512), (512, 1))
    assert_size_stride(arg104_1, (512, 2048), (2048, 1))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (512, 512), (512, 1))
    assert_size_stride(arg107_1, (512, 512), (512, 1))
    assert_size_stride(arg108_1, (512, 512), (512, 1))
    assert_size_stride(arg109_1, (512, 512), (512, 1))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (512, 512), (512, 1))
    assert_size_stride(arg112_1, (512, 512), (512, 1))
    assert_size_stride(arg113_1, (512, 512), (512, 1))
    assert_size_stride(arg114_1, (512, 512), (512, 1))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (2048, 512), (512, 1))
    assert_size_stride(arg117_1, (512, 2048), (2048, 1))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, 512), (512, 1))
    assert_size_stride(arg120_1, (512, 512), (512, 1))
    assert_size_stride(arg121_1, (512, 512), (512, 1))
    assert_size_stride(arg122_1, (512, 512), (512, 1))
    assert_size_stride(arg123_1, (512, ), (1, ))
    assert_size_stride(arg124_1, (512, 512), (512, 1))
    assert_size_stride(arg125_1, (512, 512), (512, 1))
    assert_size_stride(arg126_1, (512, 512), (512, 1))
    assert_size_stride(arg127_1, (512, 512), (512, 1))
    assert_size_stride(arg128_1, (512, ), (1, ))
    assert_size_stride(arg129_1, (2048, 512), (512, 1))
    assert_size_stride(arg130_1, (512, 2048), (2048, 1))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (512, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((4, 1024, 512), (524288, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds_1, pow_14, variance_13, add_28, rsqrt_13, hidden_states_65, normed_hidden_states_6], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0.run(arg0_1, arg1_1, arg58_1, buf1, 4096, 512, grid=grid(4096), stream=stream0)
        del arg58_1
        buf2 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (4096, 512), (512, 1), 0), reinterpret_tensor(arg53_1, (512, 512), (1, 512), 0), out=buf2)
        del arg53_1
        buf3 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (4096, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 512), (1, 512), 0), out=buf3)
        del arg54_1
        buf4 = empty_strided_cuda((4, 8, 1024, 64), (524288, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf2, buf4, 2097152, grid=grid(2097152), stream=stream0)
        buf5 = reinterpret_tensor(buf2, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf3, buf5, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf6 = empty_strided_cuda((32, 1024, 1024), (1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf5, (32, 64, 1024), (65536, 1024, 1), 0), out=buf6)
        buf7 = reinterpret_tensor(buf6, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf6  # reuse
        buf11 = empty_strided_cuda((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_13, softmax_6], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf7, arg57_1, buf11, 32768, 1024, grid=grid(32768), stream=stream0)
        buf10 = reinterpret_tensor(buf5, (4096, 512), (512, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (4096, 512), (512, 1), 0), reinterpret_tensor(arg55_1, (512, 512), (1, 512), 0), out=buf10)
        del arg55_1
        buf12 = reinterpret_tensor(buf1, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf10, buf12, 2097152, grid=grid(2097152), stream=stream0)
        buf13 = reinterpret_tensor(buf4, (32, 1024, 64), (65536, 64, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf12, (32, 1024, 64), (65536, 64, 1), 0), out=buf13)
        buf14 = reinterpret_tensor(buf12, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [contiguous_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf13, buf14, 2097152, grid=grid(2097152), stream=stream0)
        buf15 = reinterpret_tensor(buf13, (4096, 512), (512, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [attn_output_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (4096, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 512), (1, 512), 0), out=buf15)
        del arg56_1
        buf17 = reinterpret_tensor(buf14, (4, 1024, 512), (524288, 512, 1), 0); del buf14  # reuse
        buf20 = empty_strided_cuda((4, 1024, 512), (524288, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds_1, hidden_states_68, pow_15, variance_14, add_33, rsqrt_14, hidden_states_69, normed_hidden_states_7, inputs_embeds, pow_1, variance, add, rsqrt, hidden_states_1, normed_hidden_states], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_5.run(arg0_1, arg1_1, buf15, arg63_1, arg7_1, buf17, buf20, 4096, 512, grid=grid(4096), stream=stream0)
        del arg63_1
        del arg7_1
        buf18 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (4096, 512), (512, 1), 0), reinterpret_tensor(arg59_1, (512, 512), (1, 512), 0), out=buf18)
        del arg59_1
        buf21 = reinterpret_tensor(buf17, (4096, 512), (512, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (4096, 512), (512, 1), 0), reinterpret_tensor(arg2_1, (512, 512), (1, 512), 0), out=buf21)
        del arg2_1
        buf22 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (4096, 512), (512, 1), 0), reinterpret_tensor(arg3_1, (512, 512), (1, 512), 0), out=buf22)
        del arg3_1
        buf23 = empty_strided_cuda((4, 8, 1024, 64), (524288, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf21, buf23, 2097152, grid=grid(2097152), stream=stream0)
        buf24 = reinterpret_tensor(buf21, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf22, buf24, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf25 = reinterpret_tensor(buf11, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf23, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf24, (32, 64, 1024), (65536, 1024, 1), 0), out=buf25)
        buf26 = reinterpret_tensor(buf25, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf25  # reuse
        buf30 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_1, softmax], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf26, arg6_1, buf30, 32768, 1024, grid=grid(32768), stream=stream0)
        buf29 = reinterpret_tensor(buf24, (4096, 512), (512, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (4096, 512), (512, 1), 0), reinterpret_tensor(arg4_1, (512, 512), (1, 512), 0), out=buf29)
        del arg4_1
        buf31 = reinterpret_tensor(buf20, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf29, buf31, 2097152, grid=grid(2097152), stream=stream0)
        buf32 = reinterpret_tensor(buf29, (32, 1024, 64), (65536, 64, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf30, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf31, (32, 1024, 64), (65536, 64, 1), 0), out=buf32)
        buf33 = reinterpret_tensor(buf31, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf32, buf33, 2097152, grid=grid(2097152), stream=stream0)
        buf34 = reinterpret_tensor(buf32, (4096, 512), (512, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [attn_output_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (4096, 512), (512, 1), 0), reinterpret_tensor(arg5_1, (512, 512), (1, 512), 0), out=buf34)
        del arg5_1
        buf36 = reinterpret_tensor(buf33, (4, 1024, 512), (524288, 512, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_4, pow_2, variance_1, add_5, rsqrt_1, hidden_states_5, forwarded_states], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7.run(arg0_1, arg1_1, buf34, arg10_1, buf36, 4096, 512, grid=grid(4096), stream=stream0)
        del arg10_1
        buf37 = empty_strided_cuda((4096, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (4096, 512), (512, 1), 0), reinterpret_tensor(arg8_1, (512, 2048), (1, 512), 0), out=buf37)
        del arg8_1
        buf38 = reinterpret_tensor(buf37, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf38, 8388608, grid=grid(8388608), stream=stream0)
        buf39 = reinterpret_tensor(buf36, (4096, 512), (512, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (4096, 2048), (2048, 1), 0), reinterpret_tensor(arg9_1, (2048, 512), (1, 2048), 0), out=buf39)
        del arg9_1
        buf41 = reinterpret_tensor(buf23, (4, 1024, 512), (524288, 512, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_4, hidden_states_10, pow_3, variance_2, add_7, rsqrt_2, hidden_states_11, normed_hidden_states_1], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_9.run(arg0_1, arg1_1, buf34, buf39, arg15_1, buf41, 4096, 512, grid=grid(4096), stream=stream0)
        del arg15_1
        buf42 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (4096, 512), (512, 1), 0), reinterpret_tensor(arg11_1, (512, 512), (1, 512), 0), out=buf42)
        del arg11_1
        buf43 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (4096, 512), (512, 1), 0), reinterpret_tensor(arg12_1, (512, 512), (1, 512), 0), out=buf43)
        del arg12_1
        buf44 = empty_strided_cuda((4, 8, 1024, 64), (524288, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf42, buf44, 2097152, grid=grid(2097152), stream=stream0)
        buf45 = reinterpret_tensor(buf42, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [scores_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf43, buf45, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf46 = reinterpret_tensor(buf30, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf44, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf45, (32, 64, 1024), (65536, 1024, 1), 0), out=buf46)
        buf47 = reinterpret_tensor(buf46, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf46  # reuse
        buf51 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_3, softmax_1], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf47, arg6_1, buf51, 32768, 1024, grid=grid(32768), stream=stream0)
        buf50 = reinterpret_tensor(buf45, (4096, 512), (512, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (4096, 512), (512, 1), 0), reinterpret_tensor(arg13_1, (512, 512), (1, 512), 0), out=buf50)
        del arg13_1
        buf52 = reinterpret_tensor(buf41, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf50, buf52, 2097152, grid=grid(2097152), stream=stream0)
        buf53 = reinterpret_tensor(buf50, (32, 1024, 64), (65536, 64, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf51, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf52, (32, 1024, 64), (65536, 64, 1), 0), out=buf53)
        buf54 = reinterpret_tensor(buf52, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf53, buf54, 2097152, grid=grid(2097152), stream=stream0)
        buf55 = reinterpret_tensor(buf53, (4096, 512), (512, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (4096, 512), (512, 1), 0), reinterpret_tensor(arg14_1, (512, 512), (1, 512), 0), out=buf55)
        del arg14_1
        buf56 = reinterpret_tensor(buf34, (4, 1024, 512), (524288, 512, 1), 0); del buf34  # reuse
        buf58 = reinterpret_tensor(buf54, (4, 1024, 512), (524288, 512, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_4, hidden_states_10, hidden_states_14, pow_4, variance_3, add_9, rsqrt_3, hidden_states_15, forwarded_states_1], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10.run(buf56, arg0_1, arg1_1, buf39, buf55, arg18_1, buf58, 4096, 512, grid=grid(4096), stream=stream0)
        del arg18_1
        buf59 = reinterpret_tensor(buf38, (4096, 2048), (2048, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf58, (4096, 512), (512, 1), 0), reinterpret_tensor(arg16_1, (512, 2048), (1, 512), 0), out=buf59)
        del arg16_1
        buf60 = reinterpret_tensor(buf59, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf60, 8388608, grid=grid(8388608), stream=stream0)
        buf61 = reinterpret_tensor(buf58, (4096, 512), (512, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (4096, 2048), (2048, 1), 0), reinterpret_tensor(arg17_1, (2048, 512), (1, 2048), 0), out=buf61)
        del arg17_1
        buf63 = reinterpret_tensor(buf55, (4, 1024, 512), (524288, 512, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20, pow_5, variance_4, add_11, rsqrt_4, hidden_states_21, normed_hidden_states_2], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf56, buf61, arg23_1, buf63, 4096, 512, grid=grid(4096), stream=stream0)
        del arg23_1
        buf64 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (4096, 512), (512, 1), 0), reinterpret_tensor(arg19_1, (512, 512), (1, 512), 0), out=buf64)
        del arg19_1
        buf65 = reinterpret_tensor(buf44, (4096, 512), (512, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (4096, 512), (512, 1), 0), reinterpret_tensor(arg20_1, (512, 512), (1, 512), 0), out=buf65)
        del arg20_1
        buf66 = reinterpret_tensor(buf43, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf64, buf66, 2097152, grid=grid(2097152), stream=stream0)
        buf67 = reinterpret_tensor(buf64, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf65, buf67, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf68 = reinterpret_tensor(buf51, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf66, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf67, (32, 64, 1024), (65536, 1024, 1), 0), out=buf68)
        buf69 = reinterpret_tensor(buf68, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf68  # reuse
        buf73 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_5, softmax_2], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf69, arg6_1, buf73, 32768, 1024, grid=grid(32768), stream=stream0)
        buf72 = reinterpret_tensor(buf67, (4096, 512), (512, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (4096, 512), (512, 1), 0), reinterpret_tensor(arg21_1, (512, 512), (1, 512), 0), out=buf72)
        del arg21_1
        buf74 = reinterpret_tensor(buf63, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf72, buf74, 2097152, grid=grid(2097152), stream=stream0)
        buf75 = reinterpret_tensor(buf72, (32, 1024, 64), (65536, 64, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf74, (32, 1024, 64), (65536, 64, 1), 0), out=buf75)
        buf76 = reinterpret_tensor(buf74, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf75, buf76, 2097152, grid=grid(2097152), stream=stream0)
        buf77 = reinterpret_tensor(buf75, (4096, 512), (512, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [attn_output_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (4096, 512), (512, 1), 0), reinterpret_tensor(arg22_1, (512, 512), (1, 512), 0), out=buf77)
        del arg22_1
        buf79 = reinterpret_tensor(buf76, (4, 1024, 512), (524288, 512, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_24, pow_6, variance_5, add_13, rsqrt_5, hidden_states_25, forwarded_states_2], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf56, buf61, buf77, arg26_1, buf79, 4096, 512, grid=grid(4096), stream=stream0)
        del arg26_1
        buf80 = reinterpret_tensor(buf60, (4096, 2048), (2048, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (4096, 512), (512, 1), 0), reinterpret_tensor(arg24_1, (512, 2048), (1, 512), 0), out=buf80)
        del arg24_1
        buf81 = reinterpret_tensor(buf80, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_27], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf81, 8388608, grid=grid(8388608), stream=stream0)
        buf82 = reinterpret_tensor(buf79, (4096, 512), (512, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (4096, 2048), (2048, 1), 0), reinterpret_tensor(arg25_1, (2048, 512), (1, 2048), 0), out=buf82)
        del arg25_1
        buf84 = reinterpret_tensor(buf66, (4, 1024, 512), (524288, 512, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_24, hidden_states_30, pow_7, variance_6, add_15, rsqrt_6, hidden_states_31, normed_hidden_states_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf56, buf61, buf77, buf82, arg31_1, buf84, 4096, 512, grid=grid(4096), stream=stream0)
        del arg31_1
        buf85 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (4096, 512), (512, 1), 0), reinterpret_tensor(arg27_1, (512, 512), (1, 512), 0), out=buf85)
        del arg27_1
        buf86 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (4096, 512), (512, 1), 0), reinterpret_tensor(arg28_1, (512, 512), (1, 512), 0), out=buf86)
        del arg28_1
        buf87 = empty_strided_cuda((4, 8, 1024, 64), (524288, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf85, buf87, 2097152, grid=grid(2097152), stream=stream0)
        buf88 = reinterpret_tensor(buf85, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf86, buf88, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf89 = reinterpret_tensor(buf73, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf87, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf88, (32, 64, 1024), (65536, 1024, 1), 0), out=buf89)
        buf90 = reinterpret_tensor(buf89, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf89  # reuse
        buf94 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_7, softmax_3], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf90, arg6_1, buf94, 32768, 1024, grid=grid(32768), stream=stream0)
        buf93 = reinterpret_tensor(buf88, (4096, 512), (512, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (4096, 512), (512, 1), 0), reinterpret_tensor(arg29_1, (512, 512), (1, 512), 0), out=buf93)
        del arg29_1
        buf95 = reinterpret_tensor(buf84, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf93, buf95, 2097152, grid=grid(2097152), stream=stream0)
        buf96 = reinterpret_tensor(buf93, (32, 1024, 64), (65536, 64, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf95, (32, 1024, 64), (65536, 64, 1), 0), out=buf96)
        buf97 = reinterpret_tensor(buf95, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [contiguous_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf96, buf97, 2097152, grid=grid(2097152), stream=stream0)
        buf98 = reinterpret_tensor(buf96, (4096, 512), (512, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [attn_output_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (4096, 512), (512, 1), 0), reinterpret_tensor(arg30_1, (512, 512), (1, 512), 0), out=buf98)
        del arg30_1
        buf99 = buf56; del buf56  # reuse
        buf101 = reinterpret_tensor(buf97, (4, 1024, 512), (524288, 512, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_24, hidden_states_30, hidden_states_34, pow_8, variance_7, add_17, rsqrt_7, hidden_states_35, forwarded_states_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_14.run(buf99, buf61, buf77, buf82, buf98, arg34_1, buf101, 4096, 512, grid=grid(4096), stream=stream0)
        del arg34_1
        buf102 = reinterpret_tensor(buf81, (4096, 2048), (2048, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (4096, 512), (512, 1), 0), reinterpret_tensor(arg32_1, (512, 2048), (1, 512), 0), out=buf102)
        del arg32_1
        buf103 = reinterpret_tensor(buf102, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf103, 8388608, grid=grid(8388608), stream=stream0)
        buf104 = reinterpret_tensor(buf101, (4096, 512), (512, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (4096, 2048), (2048, 1), 0), reinterpret_tensor(arg33_1, (2048, 512), (1, 2048), 0), out=buf104)
        del arg33_1
        buf106 = reinterpret_tensor(buf98, (4, 1024, 512), (524288, 512, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, pow_9, variance_8, add_19, rsqrt_8, hidden_states_41, normed_hidden_states_4], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf99, buf104, arg39_1, buf106, 4096, 512, grid=grid(4096), stream=stream0)
        del arg39_1
        buf107 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (4096, 512), (512, 1), 0), reinterpret_tensor(arg35_1, (512, 512), (1, 512), 0), out=buf107)
        del arg35_1
        buf108 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (4096, 512), (512, 1), 0), reinterpret_tensor(arg36_1, (512, 512), (1, 512), 0), out=buf108)
        del arg36_1
        buf109 = reinterpret_tensor(buf61, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf107, buf109, 2097152, grid=grid(2097152), stream=stream0)
        buf110 = reinterpret_tensor(buf107, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf108, buf110, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf111 = reinterpret_tensor(buf94, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf109, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf110, (32, 64, 1024), (65536, 1024, 1), 0), out=buf111)
        buf112 = reinterpret_tensor(buf111, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf111  # reuse
        buf116 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_9, softmax_4], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf112, arg6_1, buf116, 32768, 1024, grid=grid(32768), stream=stream0)
        buf115 = reinterpret_tensor(buf110, (4096, 512), (512, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (4096, 512), (512, 1), 0), reinterpret_tensor(arg37_1, (512, 512), (1, 512), 0), out=buf115)
        del arg37_1
        buf117 = reinterpret_tensor(buf106, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf115, buf117, 2097152, grid=grid(2097152), stream=stream0)
        buf118 = reinterpret_tensor(buf115, (32, 1024, 64), (65536, 64, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf116, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf117, (32, 1024, 64), (65536, 64, 1), 0), out=buf118)
        buf119 = reinterpret_tensor(buf117, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf118, buf119, 2097152, grid=grid(2097152), stream=stream0)
        buf120 = reinterpret_tensor(buf118, (4096, 512), (512, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [attn_output_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (4096, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 512), (1, 512), 0), out=buf120)
        del arg38_1
        buf122 = reinterpret_tensor(buf119, (4, 1024, 512), (524288, 512, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_44, pow_10, variance_9, add_21, rsqrt_9, hidden_states_45, forwarded_states_4], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf99, buf104, buf120, arg42_1, buf122, 4096, 512, grid=grid(4096), stream=stream0)
        del arg42_1
        buf123 = reinterpret_tensor(buf103, (4096, 2048), (2048, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (4096, 512), (512, 1), 0), reinterpret_tensor(arg40_1, (512, 2048), (1, 512), 0), out=buf123)
        del arg40_1
        buf124 = reinterpret_tensor(buf123, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_47], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf124, 8388608, grid=grid(8388608), stream=stream0)
        buf125 = reinterpret_tensor(buf122, (4096, 512), (512, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_49], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (4096, 2048), (2048, 1), 0), reinterpret_tensor(arg41_1, (2048, 512), (1, 2048), 0), out=buf125)
        del arg41_1
        buf127 = reinterpret_tensor(buf109, (4, 1024, 512), (524288, 512, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_44, hidden_states_50, pow_11, variance_10, add_23, rsqrt_10, hidden_states_51, normed_hidden_states_5], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf99, buf104, buf120, buf125, arg47_1, buf127, 4096, 512, grid=grid(4096), stream=stream0)
        del arg47_1
        buf128 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (4096, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 512), (1, 512), 0), out=buf128)
        del arg43_1
        buf129 = reinterpret_tensor(buf87, (4096, 512), (512, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (4096, 512), (512, 1), 0), reinterpret_tensor(arg44_1, (512, 512), (1, 512), 0), out=buf129)
        del arg44_1
        buf130 = reinterpret_tensor(buf86, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [scores_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf128, buf130, 2097152, grid=grid(2097152), stream=stream0)
        buf131 = reinterpret_tensor(buf128, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [scores_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf129, buf131, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf132 = reinterpret_tensor(buf116, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf130, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf131, (32, 64, 1024), (65536, 1024, 1), 0), out=buf132)
        buf133 = reinterpret_tensor(buf132, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf132  # reuse
        buf137 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_11, softmax_5], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf133, arg6_1, buf137, 32768, 1024, grid=grid(32768), stream=stream0)
        del arg6_1
        buf136 = reinterpret_tensor(buf131, (4096, 512), (512, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (4096, 512), (512, 1), 0), reinterpret_tensor(arg45_1, (512, 512), (1, 512), 0), out=buf136)
        del arg45_1
        buf138 = reinterpret_tensor(buf127, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf136, buf138, 2097152, grid=grid(2097152), stream=stream0)
        buf139 = reinterpret_tensor(buf136, (32, 1024, 64), (65536, 64, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf138, (32, 1024, 64), (65536, 64, 1), 0), out=buf139)
        buf140 = reinterpret_tensor(buf138, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf139, buf140, 2097152, grid=grid(2097152), stream=stream0)
        buf141 = reinterpret_tensor(buf139, (4096, 512), (512, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [attn_output_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (4096, 512), (512, 1), 0), reinterpret_tensor(arg46_1, (512, 512), (1, 512), 0), out=buf141)
        del arg46_1
        buf142 = reinterpret_tensor(buf104, (4, 1024, 512), (524288, 512, 1), 0); del buf104  # reuse
        buf144 = reinterpret_tensor(buf140, (4, 1024, 512), (524288, 512, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_44, hidden_states_50, hidden_states_54, pow_12, variance_11, add_25, rsqrt_11, hidden_states_55, forwarded_states_5], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_15.run(buf142, buf99, buf120, buf125, buf141, arg50_1, buf144, 4096, 512, grid=grid(4096), stream=stream0)
        del arg50_1
        buf145 = reinterpret_tensor(buf124, (4096, 2048), (2048, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_56], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (4096, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 2048), (1, 512), 0), out=buf145)
        del arg48_1
        buf146 = reinterpret_tensor(buf145, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_57], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf146, 8388608, grid=grid(8388608), stream=stream0)
        buf147 = reinterpret_tensor(buf144, (4096, 512), (512, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_59], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (4096, 2048), (2048, 1), 0), reinterpret_tensor(arg49_1, (2048, 512), (1, 2048), 0), out=buf147)
        del arg49_1
        buf149 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_60, pow_13, variance_12, add_27, rsqrt_12, hidden_states_61, hidden_states_62], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf142, buf147, arg51_1, buf149, 4096, 512, grid=grid(4096), stream=stream0)
        del arg51_1
        buf150 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 512), (512, 1), 0), reinterpret_tensor(arg60_1, (512, 512), (1, 512), 0), out=buf150)
        del arg60_1
        buf151 = reinterpret_tensor(buf142, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [scores_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf18, buf151, 2097152, grid=grid(2097152), stream=stream0)
        buf152 = reinterpret_tensor(buf18, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [scores_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf150, buf152, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf153 = reinterpret_tensor(buf137, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [scores_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf151, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf152, (32, 64, 1024), (65536, 1024, 1), 0), out=buf153)
        buf157 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [softmax_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf153, buf157, 32768, 1024, grid=grid(32768), stream=stream0)
        buf156 = reinterpret_tensor(buf152, (4096, 512), (512, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 512), (512, 1), 0), reinterpret_tensor(arg61_1, (512, 512), (1, 512), 0), out=buf156)
        del arg61_1
        buf158 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf156, buf158, 2097152, grid=grid(2097152), stream=stream0)
        buf159 = reinterpret_tensor(buf141, (32, 1024, 64), (65536, 64, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf157, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf158, (32, 1024, 64), (65536, 64, 1), 0), out=buf159)
        buf160 = reinterpret_tensor(buf158, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [contiguous_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf159, buf160, 2097152, grid=grid(2097152), stream=stream0)
        buf161 = reinterpret_tensor(buf159, (4096, 512), (512, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [attn_output_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (4096, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 512), (1, 512), 0), out=buf161)
        del arg62_1
        buf163 = reinterpret_tensor(buf160, (4, 1024, 512), (524288, 512, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds_1, hidden_states_68, layer_output, pow_16, variance_15, add_36, rsqrt_15, hidden_states_72, forwarded_states_6], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_9.run(arg0_1, arg1_1, buf15, buf161, arg66_1, buf163, 4096, 512, grid=grid(4096), stream=stream0)
        del arg66_1
        buf164 = reinterpret_tensor(buf146, (4096, 2048), (2048, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (4096, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 2048), (1, 512), 0), out=buf164)
        del arg64_1
        buf165 = reinterpret_tensor(buf164, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_74], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf165, 8388608, grid=grid(8388608), stream=stream0)
        buf166 = reinterpret_tensor(buf163, (4096, 512), (512, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (4096, 2048), (2048, 1), 0), reinterpret_tensor(arg65_1, (2048, 512), (1, 2048), 0), out=buf166)
        del arg65_1
        buf167 = reinterpret_tensor(buf15, (4, 1024, 512), (524288, 512, 1), 0); del buf15  # reuse
        buf169 = reinterpret_tensor(buf125, (4, 1024, 512), (524288, 512, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds_1, hidden_states_68, layer_output, hidden_states_77, pow_17, variance_16, add_38, rsqrt_16, hidden_states_78, normed_hidden_states_8], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10.run(buf167, arg0_1, arg1_1, buf161, buf166, arg71_1, buf169, 4096, 512, grid=grid(4096), stream=stream0)
        del arg0_1
        del arg71_1
        buf170 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (4096, 512), (512, 1), 0), reinterpret_tensor(arg67_1, (512, 512), (1, 512), 0), out=buf170)
        del arg67_1
        buf171 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (4096, 512), (512, 1), 0), reinterpret_tensor(arg68_1, (512, 512), (1, 512), 0), out=buf171)
        del arg68_1
        buf172 = reinterpret_tensor(buf120, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf170, buf172, 2097152, grid=grid(2097152), stream=stream0)
        buf173 = reinterpret_tensor(buf170, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf171, buf173, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf174 = reinterpret_tensor(buf157, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [scores_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf172, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf173, (32, 64, 1024), (65536, 1024, 1), 0), out=buf174)
        buf175 = reinterpret_tensor(buf174, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf174  # reuse
        buf179 = reinterpret_tensor(buf153, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_17, softmax_8], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf175, arg57_1, buf179, 32768, 1024, grid=grid(32768), stream=stream0)
        buf178 = reinterpret_tensor(buf173, (4096, 512), (512, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (4096, 512), (512, 1), 0), reinterpret_tensor(arg69_1, (512, 512), (1, 512), 0), out=buf178)
        del arg69_1
        buf180 = reinterpret_tensor(buf169, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf178, buf180, 2097152, grid=grid(2097152), stream=stream0)
        buf181 = reinterpret_tensor(buf172, (32, 1024, 64), (65536, 64, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf179, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf180, (32, 1024, 64), (65536, 64, 1), 0), out=buf181)
        buf182 = reinterpret_tensor(buf180, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf181, buf182, 2097152, grid=grid(2097152), stream=stream0)
        buf183 = reinterpret_tensor(buf181, (4096, 512), (512, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [attn_output_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (4096, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 512), (1, 512), 0), out=buf183)
        del arg70_1
        buf185 = reinterpret_tensor(buf182, (4, 1024, 512), (524288, 512, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_81, pow_18, variance_17, add_40, rsqrt_17, hidden_states_82, normed_hidden_states_9], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf167, buf183, arg76_1, buf185, 4096, 512, grid=grid(4096), stream=stream0)
        del arg76_1
        buf186 = reinterpret_tensor(buf130, (4096, 512), (512, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf185, (4096, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 512), (1, 512), 0), out=buf186)
        del arg72_1
        buf187 = reinterpret_tensor(buf185, (4096, 512), (512, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 512), (512, 1), 0), reinterpret_tensor(arg73_1, (512, 512), (1, 512), 0), out=buf187)
        del arg73_1
        buf188 = reinterpret_tensor(buf129, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf186, buf188, 2097152, grid=grid(2097152), stream=stream0)
        buf189 = reinterpret_tensor(buf186, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf187, buf189, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf190 = reinterpret_tensor(buf179, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf188, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf189, (32, 64, 1024), (65536, 1024, 1), 0), out=buf190)
        buf194 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [softmax_9], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf190, buf194, 32768, 1024, grid=grid(32768), stream=stream0)
        buf193 = reinterpret_tensor(buf189, (4096, 512), (512, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [linear_52], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 512), (1, 512), 0), out=buf193)
        del arg74_1
        buf195 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf193, buf195, 2097152, grid=grid(2097152), stream=stream0)
        buf196 = empty_strided_cuda((32, 1024, 64), (65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf194, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf195, (32, 1024, 64), (65536, 64, 1), 0), out=buf196)
        buf197 = reinterpret_tensor(buf195, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [contiguous_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf196, buf197, 2097152, grid=grid(2097152), stream=stream0)
        buf198 = reinterpret_tensor(buf196, (4096, 512), (512, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [attn_output_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (4096, 512), (512, 1), 0), reinterpret_tensor(arg75_1, (512, 512), (1, 512), 0), out=buf198)
        del arg75_1
        buf200 = reinterpret_tensor(buf197, (4, 1024, 512), (524288, 512, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_81, layer_output_1, pow_19, variance_18, add_42, rsqrt_18, hidden_states_85, forwarded_states_7], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf167, buf183, buf198, arg79_1, buf200, 4096, 512, grid=grid(4096), stream=stream0)
        del arg79_1
        buf201 = reinterpret_tensor(buf165, (4096, 2048), (2048, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_86], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (4096, 512), (512, 1), 0), reinterpret_tensor(arg77_1, (512, 2048), (1, 512), 0), out=buf201)
        del arg77_1
        buf202 = reinterpret_tensor(buf201, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf202, 8388608, grid=grid(8388608), stream=stream0)
        buf203 = reinterpret_tensor(buf200, (4096, 512), (512, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_89], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (4096, 2048), (2048, 1), 0), reinterpret_tensor(arg78_1, (2048, 512), (1, 2048), 0), out=buf203)
        del arg78_1
        buf205 = empty_strided_cuda((4, 1024, 512), (524288, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_81, layer_output_1, hidden_states_90, pow_20, variance_19, add_44, rsqrt_19, hidden_states_91, normed_hidden_states_10], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf167, buf183, buf198, buf203, arg84_1, buf205, 4096, 512, grid=grid(4096), stream=stream0)
        del arg84_1
        buf206 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (4096, 512), (512, 1), 0), reinterpret_tensor(arg80_1, (512, 512), (1, 512), 0), out=buf206)
        del arg80_1
        buf207 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (4096, 512), (512, 1), 0), reinterpret_tensor(arg81_1, (512, 512), (1, 512), 0), out=buf207)
        del arg81_1
        buf208 = empty_strided_cuda((4, 8, 1024, 64), (524288, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf206, buf208, 2097152, grid=grid(2097152), stream=stream0)
        buf209 = reinterpret_tensor(buf206, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf207, buf209, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf210 = reinterpret_tensor(buf194, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [scores_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf208, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf209, (32, 64, 1024), (65536, 1024, 1), 0), out=buf210)
        buf211 = reinterpret_tensor(buf210, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf210  # reuse
        buf215 = reinterpret_tensor(buf190, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_21, softmax_10], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf211, arg57_1, buf215, 32768, 1024, grid=grid(32768), stream=stream0)
        buf214 = reinterpret_tensor(buf209, (4096, 512), (512, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (4096, 512), (512, 1), 0), reinterpret_tensor(arg82_1, (512, 512), (1, 512), 0), out=buf214)
        del arg82_1
        buf216 = reinterpret_tensor(buf205, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf214, buf216, 2097152, grid=grid(2097152), stream=stream0)
        buf217 = reinterpret_tensor(buf208, (32, 1024, 64), (65536, 64, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf215, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf216, (32, 1024, 64), (65536, 64, 1), 0), out=buf217)
        buf218 = reinterpret_tensor(buf216, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [contiguous_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf217, buf218, 2097152, grid=grid(2097152), stream=stream0)
        buf219 = reinterpret_tensor(buf217, (4096, 512), (512, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [attn_output_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf218, (4096, 512), (512, 1), 0), reinterpret_tensor(arg83_1, (512, 512), (1, 512), 0), out=buf219)
        del arg83_1
        buf220 = buf167; del buf167  # reuse
        buf222 = reinterpret_tensor(buf218, (4, 1024, 512), (524288, 512, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_81, layer_output_1, hidden_states_90, hidden_states_94, pow_21, variance_20, add_46, rsqrt_20, hidden_states_95, normed_hidden_states_11], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_14.run(buf220, buf183, buf198, buf203, buf219, arg89_1, buf222, 4096, 512, grid=grid(4096), stream=stream0)
        del arg89_1
        buf223 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf222, (4096, 512), (512, 1), 0), reinterpret_tensor(arg85_1, (512, 512), (1, 512), 0), out=buf223)
        del arg85_1
        buf224 = reinterpret_tensor(buf222, (4096, 512), (512, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 512), (1, 512), 0), out=buf224)
        del arg86_1
        buf225 = reinterpret_tensor(buf203, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [scores_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf223, buf225, 2097152, grid=grid(2097152), stream=stream0)
        buf226 = reinterpret_tensor(buf223, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [scores_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf224, buf226, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf227 = reinterpret_tensor(buf215, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [scores_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf225, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf226, (32, 64, 1024), (65536, 1024, 1), 0), out=buf227)
        buf231 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [softmax_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf227, buf231, 32768, 1024, grid=grid(32768), stream=stream0)
        buf230 = reinterpret_tensor(buf226, (4096, 512), (512, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 512), (512, 1), 0), reinterpret_tensor(arg87_1, (512, 512), (1, 512), 0), out=buf230)
        del arg87_1
        buf232 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf230, buf232, 2097152, grid=grid(2097152), stream=stream0)
        buf233 = reinterpret_tensor(buf198, (32, 1024, 64), (65536, 64, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf231, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf232, (32, 1024, 64), (65536, 64, 1), 0), out=buf233)
        buf234 = reinterpret_tensor(buf232, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf233, buf234, 2097152, grid=grid(2097152), stream=stream0)
        buf235 = reinterpret_tensor(buf233, (4096, 512), (512, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [attn_output_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (4096, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 512), (1, 512), 0), out=buf235)
        del arg88_1
        buf237 = reinterpret_tensor(buf234, (4, 1024, 512), (524288, 512, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [layer_output_2, pow_22, variance_21, add_48, rsqrt_21, hidden_states_98, forwarded_states_8], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf220, buf235, arg92_1, buf237, 4096, 512, grid=grid(4096), stream=stream0)
        del arg92_1
        buf238 = reinterpret_tensor(buf202, (4096, 2048), (2048, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_99], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (4096, 512), (512, 1), 0), reinterpret_tensor(arg90_1, (512, 2048), (1, 512), 0), out=buf238)
        del arg90_1
        buf239 = reinterpret_tensor(buf238, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf239, 8388608, grid=grid(8388608), stream=stream0)
        buf240 = reinterpret_tensor(buf237, (4096, 512), (512, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_102], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (4096, 2048), (2048, 1), 0), reinterpret_tensor(arg91_1, (2048, 512), (1, 2048), 0), out=buf240)
        del arg91_1
        buf242 = reinterpret_tensor(buf183, (4, 1024, 512), (524288, 512, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [layer_output_2, hidden_states_103, pow_23, variance_22, add_50, rsqrt_22, hidden_states_104, normed_hidden_states_12], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf220, buf235, buf240, arg97_1, buf242, 4096, 512, grid=grid(4096), stream=stream0)
        del arg97_1
        buf243 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_66], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (4096, 512), (512, 1), 0), reinterpret_tensor(arg93_1, (512, 512), (1, 512), 0), out=buf243)
        del arg93_1
        buf244 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (4096, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 512), (1, 512), 0), out=buf244)
        del arg94_1
        buf245 = empty_strided_cuda((4, 8, 1024, 64), (524288, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf243, buf245, 2097152, grid=grid(2097152), stream=stream0)
        buf246 = reinterpret_tensor(buf243, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf244, buf246, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf247 = reinterpret_tensor(buf231, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf245, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf246, (32, 64, 1024), (65536, 1024, 1), 0), out=buf247)
        buf248 = reinterpret_tensor(buf247, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf247  # reuse
        buf252 = reinterpret_tensor(buf227, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_25, softmax_12], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf248, arg57_1, buf252, 32768, 1024, grid=grid(32768), stream=stream0)
        buf251 = reinterpret_tensor(buf246, (4096, 512), (512, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (4096, 512), (512, 1), 0), reinterpret_tensor(arg95_1, (512, 512), (1, 512), 0), out=buf251)
        del arg95_1
        buf253 = reinterpret_tensor(buf242, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf251, buf253, 2097152, grid=grid(2097152), stream=stream0)
        buf254 = reinterpret_tensor(buf245, (32, 1024, 64), (65536, 64, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf252, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf253, (32, 1024, 64), (65536, 64, 1), 0), out=buf254)
        buf255 = reinterpret_tensor(buf253, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [contiguous_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf254, buf255, 2097152, grid=grid(2097152), stream=stream0)
        buf256 = reinterpret_tensor(buf254, (4096, 512), (512, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [attn_output_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (4096, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 512), (1, 512), 0), out=buf256)
        del arg96_1
        buf258 = reinterpret_tensor(buf255, (4, 1024, 512), (524288, 512, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [layer_output_2, hidden_states_103, hidden_states_107, pow_24, variance_23, add_52, rsqrt_23, hidden_states_108, normed_hidden_states_13], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf220, buf235, buf240, buf256, arg102_1, buf258, 4096, 512, grid=grid(4096), stream=stream0)
        del arg102_1
        buf259 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (4096, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 512), (1, 512), 0), out=buf259)
        del arg98_1
        buf260 = reinterpret_tensor(buf258, (4096, 512), (512, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 512), (512, 1), 0), reinterpret_tensor(arg99_1, (512, 512), (1, 512), 0), out=buf260)
        del arg99_1
        buf261 = empty_strided_cuda((4, 8, 1024, 64), (524288, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf259, buf261, 2097152, grid=grid(2097152), stream=stream0)
        buf262 = reinterpret_tensor(buf259, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [scores_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf260, buf262, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf263 = reinterpret_tensor(buf252, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [scores_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf261, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf262, (32, 64, 1024), (65536, 1024, 1), 0), out=buf263)
        buf267 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [softmax_13], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf263, buf267, 32768, 1024, grid=grid(32768), stream=stream0)
        buf266 = reinterpret_tensor(buf262, (4096, 512), (512, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 512), (512, 1), 0), reinterpret_tensor(arg100_1, (512, 512), (1, 512), 0), out=buf266)
        del arg100_1
        buf268 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf266, buf268, 2097152, grid=grid(2097152), stream=stream0)
        buf269 = empty_strided_cuda((32, 1024, 64), (65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf267, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf268, (32, 1024, 64), (65536, 64, 1), 0), out=buf269)
        buf270 = reinterpret_tensor(buf268, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [contiguous_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf269, buf270, 2097152, grid=grid(2097152), stream=stream0)
        buf271 = reinterpret_tensor(buf269, (4096, 512), (512, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [attn_output_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (4096, 512), (512, 1), 0), reinterpret_tensor(arg101_1, (512, 512), (1, 512), 0), out=buf271)
        del arg101_1
        buf272 = buf220; del buf220  # reuse
        buf274 = reinterpret_tensor(buf270, (4, 1024, 512), (524288, 512, 1), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [layer_output_2, hidden_states_103, hidden_states_107, layer_output_3, pow_25, variance_24, add_54, rsqrt_24, hidden_states_111, forwarded_states_9], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_14.run(buf272, buf235, buf240, buf256, buf271, arg105_1, buf274, 4096, 512, grid=grid(4096), stream=stream0)
        del arg105_1
        buf275 = reinterpret_tensor(buf239, (4096, 2048), (2048, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_112], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (4096, 512), (512, 1), 0), reinterpret_tensor(arg103_1, (512, 2048), (1, 512), 0), out=buf275)
        del arg103_1
        buf276 = reinterpret_tensor(buf275, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_113], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf276, 8388608, grid=grid(8388608), stream=stream0)
        buf277 = reinterpret_tensor(buf274, (4096, 512), (512, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_115], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (4096, 2048), (2048, 1), 0), reinterpret_tensor(arg104_1, (2048, 512), (1, 2048), 0), out=buf277)
        del arg104_1
        buf279 = reinterpret_tensor(buf271, (4, 1024, 512), (524288, 512, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_116, pow_26, variance_25, add_56, rsqrt_25, hidden_states_117, normed_hidden_states_14], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf272, buf277, arg110_1, buf279, 4096, 512, grid=grid(4096), stream=stream0)
        del arg110_1
        buf280 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [linear_76], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (4096, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 512), (1, 512), 0), out=buf280)
        del arg106_1
        buf281 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (4096, 512), (512, 1), 0), reinterpret_tensor(arg107_1, (512, 512), (1, 512), 0), out=buf281)
        del arg107_1
        buf282 = reinterpret_tensor(buf235, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [scores_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf280, buf282, 2097152, grid=grid(2097152), stream=stream0)
        buf283 = reinterpret_tensor(buf280, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [scores_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf281, buf283, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf284 = reinterpret_tensor(buf267, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [scores_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf282, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf283, (32, 64, 1024), (65536, 1024, 1), 0), out=buf284)
        buf285 = reinterpret_tensor(buf284, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf284  # reuse
        buf289 = reinterpret_tensor(buf263, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_29, softmax_14], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf285, arg57_1, buf289, 32768, 1024, grid=grid(32768), stream=stream0)
        buf288 = reinterpret_tensor(buf283, (4096, 512), (512, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (4096, 512), (512, 1), 0), reinterpret_tensor(arg108_1, (512, 512), (1, 512), 0), out=buf288)
        del arg108_1
        buf290 = reinterpret_tensor(buf279, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf288, buf290, 2097152, grid=grid(2097152), stream=stream0)
        buf291 = reinterpret_tensor(buf282, (32, 1024, 64), (65536, 64, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf289, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf290, (32, 1024, 64), (65536, 64, 1), 0), out=buf291)
        buf292 = reinterpret_tensor(buf290, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf291, buf292, 2097152, grid=grid(2097152), stream=stream0)
        buf293 = reinterpret_tensor(buf291, (4096, 512), (512, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [attn_output_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (4096, 512), (512, 1), 0), reinterpret_tensor(arg109_1, (512, 512), (1, 512), 0), out=buf293)
        del arg109_1
        buf295 = reinterpret_tensor(buf292, (4, 1024, 512), (524288, 512, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_116, hidden_states_120, pow_27, variance_26, add_58, rsqrt_26, hidden_states_121, normed_hidden_states_15], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf272, buf277, buf293, arg115_1, buf295, 4096, 512, grid=grid(4096), stream=stream0)
        del arg115_1
        buf296 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_80], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (4096, 512), (512, 1), 0), reinterpret_tensor(arg111_1, (512, 512), (1, 512), 0), out=buf296)
        del arg111_1
        buf297 = reinterpret_tensor(buf295, (4096, 512), (512, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [linear_81], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 512), (1, 512), 0), out=buf297)
        del arg112_1
        buf298 = empty_strided_cuda((4, 8, 1024, 64), (524288, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf296, buf298, 2097152, grid=grid(2097152), stream=stream0)
        buf299 = reinterpret_tensor(buf296, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf297, buf299, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf300 = reinterpret_tensor(buf289, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [scores_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf298, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf299, (32, 64, 1024), (65536, 1024, 1), 0), out=buf300)
        buf304 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [softmax_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf300, buf304, 32768, 1024, grid=grid(32768), stream=stream0)
        buf303 = reinterpret_tensor(buf299, (4096, 512), (512, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 512), (512, 1), 0), reinterpret_tensor(arg113_1, (512, 512), (1, 512), 0), out=buf303)
        del arg113_1
        buf305 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [matmul_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf303, buf305, 2097152, grid=grid(2097152), stream=stream0)
        buf306 = empty_strided_cuda((32, 1024, 64), (65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf304, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf305, (32, 1024, 64), (65536, 64, 1), 0), out=buf306)
        buf307 = reinterpret_tensor(buf305, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [contiguous_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf306, buf307, 2097152, grid=grid(2097152), stream=stream0)
        buf308 = reinterpret_tensor(buf306, (4096, 512), (512, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [attn_output_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (4096, 512), (512, 1), 0), reinterpret_tensor(arg114_1, (512, 512), (1, 512), 0), out=buf308)
        del arg114_1
        buf310 = reinterpret_tensor(buf307, (4, 1024, 512), (524288, 512, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_116, hidden_states_120, layer_output_4, pow_28, variance_27, add_60, rsqrt_27, hidden_states_124, forwarded_states_10], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf272, buf277, buf293, buf308, arg118_1, buf310, 4096, 512, grid=grid(4096), stream=stream0)
        del arg118_1
        buf311 = reinterpret_tensor(buf276, (4096, 2048), (2048, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_125], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (4096, 512), (512, 1), 0), reinterpret_tensor(arg116_1, (512, 2048), (1, 512), 0), out=buf311)
        del arg116_1
        buf312 = reinterpret_tensor(buf311, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_126], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf312, 8388608, grid=grid(8388608), stream=stream0)
        buf313 = reinterpret_tensor(buf310, (4096, 512), (512, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_128], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (4096, 2048), (2048, 1), 0), reinterpret_tensor(arg117_1, (2048, 512), (1, 2048), 0), out=buf313)
        del arg117_1
        buf314 = buf272; del buf272  # reuse
        buf316 = empty_strided_cuda((4, 1024, 512), (524288, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_116, hidden_states_120, layer_output_4, hidden_states_129, pow_29, variance_28, add_62, rsqrt_28, hidden_states_130, normed_hidden_states_16], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_14.run(buf314, buf277, buf293, buf308, buf313, arg123_1, buf316, 4096, 512, grid=grid(4096), stream=stream0)
        del arg123_1
        buf317 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (4096, 512), (512, 1), 0), reinterpret_tensor(arg119_1, (512, 512), (1, 512), 0), out=buf317)
        del arg119_1
        buf318 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [linear_87], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (4096, 512), (512, 1), 0), reinterpret_tensor(arg120_1, (512, 512), (1, 512), 0), out=buf318)
        del arg120_1
        buf319 = reinterpret_tensor(buf293, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [scores_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf317, buf319, 2097152, grid=grid(2097152), stream=stream0)
        buf320 = reinterpret_tensor(buf317, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [scores_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf318, buf320, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf321 = reinterpret_tensor(buf304, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [scores_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf319, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf320, (32, 64, 1024), (65536, 1024, 1), 0), out=buf321)
        buf322 = reinterpret_tensor(buf321, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf321  # reuse
        buf326 = reinterpret_tensor(buf300, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_33, softmax_16], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf322, arg57_1, buf326, 32768, 1024, grid=grid(32768), stream=stream0)
        del arg57_1
        buf325 = reinterpret_tensor(buf320, (4096, 512), (512, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (4096, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 512), (1, 512), 0), out=buf325)
        del arg121_1
        buf327 = reinterpret_tensor(buf316, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [matmul_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf325, buf327, 2097152, grid=grid(2097152), stream=stream0)
        buf328 = reinterpret_tensor(buf319, (32, 1024, 64), (65536, 64, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf326, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf327, (32, 1024, 64), (65536, 64, 1), 0), out=buf328)
        buf329 = reinterpret_tensor(buf327, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [contiguous_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf328, buf329, 2097152, grid=grid(2097152), stream=stream0)
        buf330 = reinterpret_tensor(buf328, (4096, 512), (512, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [attn_output_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf329, (4096, 512), (512, 1), 0), reinterpret_tensor(arg122_1, (512, 512), (1, 512), 0), out=buf330)
        del arg122_1
        buf332 = reinterpret_tensor(buf329, (4, 1024, 512), (524288, 512, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_133, pow_30, variance_29, add_64, rsqrt_29, hidden_states_134, normed_hidden_states_17], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf314, buf330, arg128_1, buf332, 4096, 512, grid=grid(4096), stream=stream0)
        del arg128_1
        buf333 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [linear_90], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (4096, 512), (512, 1), 0), reinterpret_tensor(arg124_1, (512, 512), (1, 512), 0), out=buf333)
        del arg124_1
        buf334 = reinterpret_tensor(buf332, (4096, 512), (512, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 512), (512, 1), 0), reinterpret_tensor(arg125_1, (512, 512), (1, 512), 0), out=buf334)
        del arg125_1
        buf335 = empty_strided_cuda((4, 8, 1024, 64), (524288, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf333, buf335, 2097152, grid=grid(2097152), stream=stream0)
        buf336 = reinterpret_tensor(buf333, (4, 8, 64, 1024), (524288, 65536, 1024, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [scores_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf334, buf336, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf337 = reinterpret_tensor(buf326, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [scores_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf336, (32, 64, 1024), (65536, 1024, 1), 0), out=buf337)
        buf341 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [softmax_17], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf337, buf341, 32768, 1024, grid=grid(32768), stream=stream0)
        del buf337
        buf340 = reinterpret_tensor(buf336, (4096, 512), (512, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [linear_92], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 512), (512, 1), 0), reinterpret_tensor(arg126_1, (512, 512), (1, 512), 0), out=buf340)
        del arg126_1
        buf342 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [matmul_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf340, buf342, 2097152, grid=grid(2097152), stream=stream0)
        buf343 = empty_strided_cuda((32, 1024, 64), (65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf341, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf342, (32, 1024, 64), (65536, 64, 1), 0), out=buf343)
        del buf341
        buf344 = reinterpret_tensor(buf342, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf343, buf344, 2097152, grid=grid(2097152), stream=stream0)
        buf345 = reinterpret_tensor(buf343, (4096, 512), (512, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [attn_output_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (4096, 512), (512, 1), 0), reinterpret_tensor(arg127_1, (512, 512), (1, 512), 0), out=buf345)
        del arg127_1
        buf347 = reinterpret_tensor(buf344, (4, 1024, 512), (524288, 512, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_133, layer_output_5, pow_31, variance_30, add_66, rsqrt_30, hidden_states_137, forwarded_states_11], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf314, buf330, buf345, arg131_1, buf347, 4096, 512, grid=grid(4096), stream=stream0)
        del arg131_1
        buf348 = reinterpret_tensor(buf312, (4096, 2048), (2048, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_138], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf347, (4096, 512), (512, 1), 0), reinterpret_tensor(arg129_1, (512, 2048), (1, 512), 0), out=buf348)
        del arg129_1
        buf349 = reinterpret_tensor(buf348, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf348  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_139], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf349, 8388608, grid=grid(8388608), stream=stream0)
        buf350 = reinterpret_tensor(buf347, (4096, 512), (512, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_141], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (4096, 2048), (2048, 1), 0), reinterpret_tensor(arg130_1, (2048, 512), (1, 2048), 0), out=buf350)
        del arg130_1
        del buf349
        buf352 = empty_strided_cuda((4, 1024, 512), (524288, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_133, layer_output_5, hidden_states_142, pow_32, variance_31, add_68, rsqrt_31, hidden_states_143, hidden_states_144, sequence_output], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_17.run(buf314, buf330, buf345, buf350, arg132_1, buf352, 4096, 512, grid=grid(4096), stream=stream0)
        del arg132_1
        del buf314
        del buf330
        del buf345
        del buf350
        buf353 = empty_strided_cuda((4096, 32128), (32128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (4096, 512), (512, 1), 0), reinterpret_tensor(arg1_1, (512, 32128), (1, 512), 0), out=buf353)
        del arg1_1
        del buf352
        buf354 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        buf355 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_18.run(buf353, buf354, buf355, 4096, 32128, grid=grid(4096), stream=stream0)
        buf356 = empty_strided_cuda((), (), torch.float32)
        buf358 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_19.run(buf358, arg52_1, buf353, buf354, buf355, 1, 4096, grid=grid(1), stream=stream0)
        del arg52_1
        del buf354
        del buf355
    return (buf358, reinterpret_tensor(buf353, (4, 1024, 32128), (32899072, 32128, 1), 0), reinterpret_tensor(buf3, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf10, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf150, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf156, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf171, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf178, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf187, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf193, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf207, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf214, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf224, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf230, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf244, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf251, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf260, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf266, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf281, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf288, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf297, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf303, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf318, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf325, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf334, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf340, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), buf149, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((32128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg53_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('T5ForConditionalGeneration', benchmark_compiled_module)
