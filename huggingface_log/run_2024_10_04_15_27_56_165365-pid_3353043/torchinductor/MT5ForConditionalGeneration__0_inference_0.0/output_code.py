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


# kernel path: /tmp/torchinductor_sahanp/cs/ccslslockdy544fmnuink5aru2o752nntkusmh2ub266ovozzrv2.py
# Topologically Sorted Source Nodes: [inputs_embeds_1, pow_26, variance_17, add_52, rsqrt_17, hidden_states_77, normed_hidden_states_8], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_52 => add_61
#   hidden_states_77 => mul_80
#   inputs_embeds_1 => embedding_2
#   normed_hidden_states_8 => mul_81
#   pow_26 => pow_26
#   rsqrt_17 => rsqrt_17
#   variance_17 => mean_17
# Graph fragment:
#   %embedding_2 : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view_209), kwargs = {})
#   %pow_26 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%embedding_2, 2), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_26, [-1], True), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_17, 1e-06), kwargs = {})
#   %rsqrt_17 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_61,), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding_2, %rsqrt_17), kwargs = {})
#   %mul_81 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg82_1, %mul_80), kwargs = {})
triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
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
        tmp1 = tl.full([XBLOCK, RBLOCK], 250112, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert((0 <= tmp4) & (tmp4 < 250112), "index out of bounds: 0 <= tmp4 < 250112")
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
        tmp12 = tl.full([XBLOCK, RBLOCK], 250112, tl.int32)
        tmp13 = tmp0 + tmp12
        tmp14 = tmp0 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp0)
        tl.device_assert((0 <= tmp15) & (tmp15 < 250112), "index out of bounds: 0 <= tmp15 < 250112")
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


# kernel path: /tmp/torchinductor_sahanp/3g/c3gptpeszlpsfzkfxqxzyvyhvhfadluz3fdpmq5jwlqg6elq36o4.py
# Topologically Sorted Source Nodes: [scores_16], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   scores_16 => clone_67
# Graph fragment:
#   %clone_67 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_32,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192) % 6
    x3 = (xindex // 49152)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (384*x1) + (49152*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lc/clc7ztioiwo5bwkorbp4bncw374rw53cbxphv2papyxfbvgwx6tf.py
# Topologically Sorted Source Nodes: [scores_16], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   scores_16 => clone_68
# Graph fragment:
#   %clone_68 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_33,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (49152*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wo/cwodj5oswc6agmibvwsmz5zhvmukn6qoo6dzftg5zynsouy2rkz6.py
# Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_17, softmax_8], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   extended_attention_mask_5 => mul_78
#   position_bias_1 => add_64
#   scores_17 => add_65
#   softmax_8 => amax_8, div_12, exp_8, sub_13, sum_9
#   sub_2 => sub_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %unsqueeze_9), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, -3.4028234663852886e+38), kwargs = {})
#   %add_64 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_16, %mul_78), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_221, %add_64), kwargs = {})
#   %amax_8 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_223, [-1], True), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_223, %amax_8), kwargs = {})
#   %exp_8 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_13,), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_8, [-1], True), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_8, %sum_9), kwargs = {})
triton_per_fused__softmax_add_mul_rsub_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_rsub_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 6
    tmp0 = tl.load(in_out_ptr0 + (r3 + (128*x4)), None)
    tmp1 = (-1)*((0) * ((0) <= (r3 + ((-1)*x0))) + (r3 + ((-1)*x0)) * ((r3 + ((-1)*x0)) < (0)))
    tmp2 = tl.full([1, 1], 16, tl.int64)
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
    tmp14 = tl.full([1, 1], 31, tl.int64)
    tmp15 = triton_helpers.minimum(tmp13, tmp14)
    tmp16 = tl.where(tmp3, tmp1, tmp15)
    tmp17 = tl.full([1, 1], 0, tl.int64)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([XBLOCK, RBLOCK], 32, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 32), "index out of bounds: 0 <= tmp22 < 32")
    tmp24 = tl.load(in_ptr0 + (x1 + (6*tmp22)), None, eviction_policy='evict_last')
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
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = triton_helpers.max2(tmp35, 1)[:, None]
    tmp38 = tmp34 - tmp37
    tmp39 = tl_math.exp(tmp38)
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.sum(tmp40, 1)[:, None]
    tmp43 = tmp39 / tmp42
    tl.store(out_ptr2 + (r3 + (128*x4)), tmp43, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g3/cg35n3pbwpdtsd22bj363liuuap574hf5maxoiutiegz6ypfqm4b.py
# Topologically Sorted Source Nodes: [contiguous_8], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_8 => clone_71
# Graph fragment:
#   %clone_71 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_105,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 6
    x2 = (xindex // 384) % 128
    x3 = (xindex // 49152)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (8192*x1) + (49152*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/q5/cq5la4duk53b4ln6tynmnrwruhliahmrrlcpr77rcg6l7wzbumiq.py
# Topologically Sorted Source Nodes: [inputs_embeds_1, hidden_states_80, pow_27, variance_18, add_57, rsqrt_18, hidden_states_81, normed_hidden_states_9, inputs_embeds, pow_1, variance, add, rsqrt, hidden_states_1, normed_hidden_states], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_57 => add_67
#   hidden_states_1 => mul_1
#   hidden_states_80 => add_66
#   hidden_states_81 => mul_83
#   inputs_embeds => embedding
#   inputs_embeds_1 => embedding_2
#   normed_hidden_states => mul_2
#   normed_hidden_states_9 => mul_84
#   pow_1 => pow_1
#   pow_27 => pow_27
#   rsqrt => rsqrt
#   rsqrt_18 => rsqrt_18
#   variance => mean
#   variance_18 => mean_18
# Graph fragment:
#   %embedding_2 : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view_209), kwargs = {})
#   %add_66 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding_2, %view_229), kwargs = {})
#   %pow_27 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_66, 2), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_27, [-1], True), kwargs = {})
#   %add_67 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_18, 1e-06), kwargs = {})
#   %rsqrt_18 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_67,), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_66, %rsqrt_18), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg87_1, %mul_83), kwargs = {})
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
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
        tmp1 = tl.full([XBLOCK, RBLOCK], 250112, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert((0 <= tmp4) & (tmp4 < 250112), "index out of bounds: 0 <= tmp4 < 250112")
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
        tmp18 = tl.full([XBLOCK, RBLOCK], 250112, tl.int32)
        tmp19 = tmp0 + tmp18
        tmp20 = tmp0 < 0
        tmp21 = tl.where(tmp20, tmp19, tmp0)
        tl.device_assert((0 <= tmp21) & (tmp21 < 250112), "index out of bounds: 0 <= tmp21 < 250112")
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


# kernel path: /tmp/torchinductor_sahanp/e6/ce6hkzwgcltcebmxac2z3edhwrk7ypkj27l2c2saadzkhpjxfwab.py
# Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_1, softmax], Original ATen: [aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   extended_attention_mask_2 => full_default
#   position_bias => add_4
#   scores_1 => add_5
#   softmax => amax, div_2, exp, sub_2, sum_1
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([16, 1, 1, 128], -0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_4 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_4, %full_default), kwargs = {})
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 6
    tmp0 = tl.load(in_out_ptr0 + (r3 + (128*x4)), None)
    tmp1 = r3 + ((-1)*x0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 > tmp2
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.full([1, 1], 16, tl.int64)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6 + tmp2
    tmp8 = tl_math.abs(r3 + ((-1)*x0))
    tmp9 = tl.full([1, 1], 8, tl.int64)
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
    tmp21 = tl.full([1, 1], 15, tl.int64)
    tmp22 = triton_helpers.minimum(tmp20, tmp21)
    tmp23 = tl.where(tmp10, tmp8, tmp22)
    tmp24 = tmp7 + tmp23
    tmp25 = tl.full([XBLOCK, RBLOCK], 32, tl.int32)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp24 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp24)
    tl.device_assert((0 <= tmp28) & (tmp28 < 32), "index out of bounds: 0 <= tmp28 < 32")
    tmp30 = tl.load(in_ptr0 + (x1 + (6*tmp28)), None, eviction_policy='evict_last')
    tmp31 = -0.0
    tmp32 = tmp30 + tmp31
    tmp33 = tmp0 + tmp32
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    tmp36 = triton_helpers.max2(tmp34, 1)[:, None]
    tmp37 = tmp33 - tmp36
    tmp38 = tl_math.exp(tmp37)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
    tmp41 = tl.sum(tmp39, 1)[:, None]
    tmp42 = tmp38 / tmp41
    tl.store(out_ptr2 + (r3 + (128*x4)), tmp42, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ql/cqlr4annl55bpshc7e6s62k55vm3vkjlhpg6rbyu2lkoaqpcrb5q.py
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
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg11_1, %mul_5), kwargs = {})
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 2048
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
    tmp1 = tl.full([RBLOCK], 250112, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 250112), "index out of bounds: 0 <= tmp4 < 250112")
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


# kernel path: /tmp/torchinductor_sahanp/27/c27rkvobmukfpb7ln52kwhuepfz2kqy5hkye2drrebgo2qihxf2l.py
# Topologically Sorted Source Nodes: [mul_7, pow_3, mul_8, add_6, mul_9, tanh, add_7, hidden_gelu, hidden_states_6], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_6 => add_8
#   add_7 => add_9
#   hidden_gelu => mul_10
#   hidden_states_6 => mul_11
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   pow_3 => pow_3
#   tanh => tanh
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, 0.5), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_22, 3.0), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_3, 0.044715), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_22, %mul_8), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 0.7978845608028654), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_9,), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_9), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %view_24), kwargs = {})
triton_poi_fused_add_mul_pow_tanh_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp14 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 * tmp0
    tmp4 = tmp3 * tmp0
    tmp5 = 0.044715
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tmp8 = 0.7978845608028654
    tmp9 = tmp7 * tmp8
    tmp10 = libdevice.tanh(tmp9)
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp2 * tmp12
    tmp15 = tmp13 * tmp14
    tl.store(in_out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2f/c2fzbmx3muczvyjk6hhry6zrwf3adpwdpco2532o4agfs32iecw5.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_4, hidden_states_9, pow_4, variance_2, add_9, rsqrt_2, hidden_states_10, normed_hidden_states_1], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_9 => add_11
#   hidden_states_10 => mul_12
#   hidden_states_4 => add_6
#   hidden_states_9 => add_10
#   inputs_embeds => embedding
#   normed_hidden_states_1 => mul_13
#   pow_4 => pow_4
#   rsqrt_2 => rsqrt_2
#   variance_2 => mean_2
# Graph fragment:
#   %embedding : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_20), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_26), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_10, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %rsqrt_2), kwargs = {})
#   %mul_13 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg16_1, %mul_12), kwargs = {})
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_9', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 2048
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
    tmp1 = tl.full([RBLOCK], 250112, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 250112), "index out of bounds: 0 <= tmp4 < 250112")
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


# kernel path: /tmp/torchinductor_sahanp/ce/ccel5bokm7gjohvu6bzqrdosnj44iq7o2ikwiyxvj6x3aziqtk4v.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_4, hidden_states_9, hidden_states_13, pow_5, variance_3, add_11, rsqrt_3, hidden_states_14, forwarded_states_1], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_11 => add_14
#   forwarded_states_1 => mul_15
#   hidden_states_13 => add_13
#   hidden_states_14 => mul_14
#   hidden_states_4 => add_6
#   hidden_states_9 => add_10
#   inputs_embeds => embedding
#   pow_5 => pow_5
#   rsqrt_3 => rsqrt_3
#   variance_3 => mean_3
# Graph fragment:
#   %embedding : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_20), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_26), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_46), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_13, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %rsqrt_3), kwargs = {})
#   %mul_15 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg20_1, %mul_14), kwargs = {})
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 2048
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
    tmp1 = tl.full([RBLOCK], 250112, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 250112), "index out of bounds: 0 <= tmp4 < 250112")
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


# kernel path: /tmp/torchinductor_sahanp/d6/cd6jao7xrkc6oqz7la37oje5cgluoayxoywv7rjmcfyhguzyujsh.py
# Topologically Sorted Source Nodes: [hidden_states_18, pow_7, variance_4, add_15, rsqrt_4, hidden_states_19, normed_hidden_states_2], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_15 => add_18
#   hidden_states_18 => add_17
#   hidden_states_19 => mul_21
#   normed_hidden_states_2 => mul_22
#   pow_7 => pow_7
#   rsqrt_4 => rsqrt_4
#   variance_4 => mean_4
# Graph fragment:
#   %add_17 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_52), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_17, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [-1], True), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %rsqrt_4), kwargs = {})
#   %mul_22 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg25_1, %mul_21), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 2048
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


# kernel path: /tmp/torchinductor_sahanp/xz/cxznkq2gut3kg2qmpcdmbsqbud2bsnevpvyx5emgpccmmwqlyrwq.py
# Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_22, pow_8, variance_5, add_17, rsqrt_5, hidden_states_23, forwarded_states_2], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_17 => add_21
#   forwarded_states_2 => mul_24
#   hidden_states_18 => add_17
#   hidden_states_22 => add_20
#   hidden_states_23 => mul_23
#   pow_8 => pow_8
#   rsqrt_5 => rsqrt_5
#   variance_5 => mean_5
# Graph fragment:
#   %add_17 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_52), kwargs = {})
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_72), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_20, 2), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [-1], True), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, 1e-06), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_21,), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_20, %rsqrt_5), kwargs = {})
#   %mul_24 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg29_1, %mul_23), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 2048
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


# kernel path: /tmp/torchinductor_sahanp/zc/czcm4hirykz2aeluqwtlnm3qimckb5qr3lsbvmkajvkkr3lyn3hp.py
# Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_22, hidden_states_27, pow_10, variance_6, add_21, rsqrt_6, hidden_states_28, normed_hidden_states_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_21 => add_25
#   hidden_states_18 => add_17
#   hidden_states_22 => add_20
#   hidden_states_27 => add_24
#   hidden_states_28 => mul_30
#   normed_hidden_states_3 => mul_31
#   pow_10 => pow_10
#   rsqrt_6 => rsqrt_6
#   variance_6 => mean_6
# Graph fragment:
#   %add_17 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_52), kwargs = {})
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_72), kwargs = {})
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %view_78), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_24, 2), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_10, [-1], True), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_6, 1e-06), kwargs = {})
#   %rsqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %rsqrt_6), kwargs = {})
#   %mul_31 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg34_1, %mul_30), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_13', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 2048
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


# kernel path: /tmp/torchinductor_sahanp/5l/c5lmyw3p5yn6u34unisu2vpknrme6xbkielqp7eapw4nhhqpeds2.py
# Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_22, hidden_states_27, hidden_states_31, pow_11, variance_7, add_23, rsqrt_7, hidden_states_32, forwarded_states_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_23 => add_28
#   forwarded_states_3 => mul_33
#   hidden_states_18 => add_17
#   hidden_states_22 => add_20
#   hidden_states_27 => add_24
#   hidden_states_31 => add_27
#   hidden_states_32 => mul_32
#   pow_11 => pow_11
#   rsqrt_7 => rsqrt_7
#   variance_7 => mean_7
# Graph fragment:
#   %add_17 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_52), kwargs = {})
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_72), kwargs = {})
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %view_78), kwargs = {})
#   %add_27 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %view_98), kwargs = {})
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_27, 2), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_11, [-1], True), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_7, 1e-06), kwargs = {})
#   %rsqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_28,), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %rsqrt_7), kwargs = {})
#   %mul_33 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg38_1, %mul_32), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_14', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 2048
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
    tmp7 = tl.load(in_out_ptr0 + (r1 + (512*x0)), None)
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


# kernel path: /tmp/torchinductor_sahanp/zi/czikfvohry5nie3khq67atan6vi25rybnlmbae2gow5hnxooam2p.py
# Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_40, hidden_states_45, hidden_states_49, pow_17, variance_11, add_35, rsqrt_11, hidden_states_50, forwarded_states_5], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_35 => add_42
#   forwarded_states_5 => mul_51
#   hidden_states_36 => add_31
#   hidden_states_40 => add_34
#   hidden_states_45 => add_38
#   hidden_states_49 => add_41
#   hidden_states_50 => mul_50
#   pow_17 => pow_17
#   rsqrt_11 => rsqrt_11
#   variance_11 => mean_11
# Graph fragment:
#   %add_31 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %view_104), kwargs = {})
#   %add_34 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_31, %view_124), kwargs = {})
#   %add_38 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %view_130), kwargs = {})
#   %add_41 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_38, %view_150), kwargs = {})
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_41, 2), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_17, [-1], True), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_11, 1e-06), kwargs = {})
#   %rsqrt_11 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_42,), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_41, %rsqrt_11), kwargs = {})
#   %mul_51 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg56_1, %mul_50), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_15', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 2048
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


# kernel path: /tmp/torchinductor_sahanp/dp/cdp4xhzlcr4q6tzv6lyxd4475jp2tehvctp5lzik52awlnxj2gd3.py
# Topologically Sorted Source Nodes: [softmax_9], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   softmax_9 => amax_9, div_13, exp_9, sub_14, sum_10
# Graph fragment:
#   %amax_9 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_243, [-1], True), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_243, %amax_9), kwargs = {})
#   %exp_9 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_14,), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_9, [-1], True), kwargs = {})
#   %div_13 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_9, %sum_10), kwargs = {})
triton_per_fused__softmax_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None]
    tmp4 = tmp0 - tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp5 / tmp8
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qf/cqfydubapwo6u5thmx6ycgqutoevix7ukqquuzdw57no3lslfzep.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   loss => amax_24, exp_24, sub_29, sum_25
# Graph fragment:
#   %amax_24 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_580, [1], True), kwargs = {})
#   %sub_29 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_580, %amax_24), kwargs = {})
#   %exp_24 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_29,), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [1], True), kwargs = {})
triton_red_fused__log_softmax_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 262144],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 250112
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
        tmp0 = tl.load(in_ptr0 + (r1 + (250112*x0)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (250112*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nb/cnbi6hep37pm6mgi2iivhzz7z55jzfeaw2cddp5xxhl3vmhatt5y.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type_7, div_28, full_default_7, ne_1, ne_2, neg_1, sum_26, sum_27, where_3
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_581, -100), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg_1, %full_default_7), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_3,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_581, -100), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_26, torch.float32), kwargs = {})
#   %div_28 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_27, %convert_element_type_7), kwargs = {})
triton_red_fused_nll_loss_forward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_18', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
        tmp5 = tl.full([XBLOCK, RBLOCK], 250112, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 250112)) | ~(rmask), "index out of bounds: 0 <= tmp8 < 250112")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (250112*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 128), (128, 1))
    assert_size_stride(arg1_1, (250112, 512), (512, 1))
    assert_size_stride(arg2_1, (384, 512), (512, 1))
    assert_size_stride(arg3_1, (384, 512), (512, 1))
    assert_size_stride(arg4_1, (384, 512), (512, 1))
    assert_size_stride(arg5_1, (512, 384), (384, 1))
    assert_size_stride(arg6_1, (32, 6), (6, 1))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (1024, 512), (512, 1))
    assert_size_stride(arg9_1, (1024, 512), (512, 1))
    assert_size_stride(arg10_1, (512, 1024), (1024, 1))
    assert_size_stride(arg11_1, (512, ), (1, ))
    assert_size_stride(arg12_1, (384, 512), (512, 1))
    assert_size_stride(arg13_1, (384, 512), (512, 1))
    assert_size_stride(arg14_1, (384, 512), (512, 1))
    assert_size_stride(arg15_1, (512, 384), (384, 1))
    assert_size_stride(arg16_1, (512, ), (1, ))
    assert_size_stride(arg17_1, (1024, 512), (512, 1))
    assert_size_stride(arg18_1, (1024, 512), (512, 1))
    assert_size_stride(arg19_1, (512, 1024), (1024, 1))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (384, 512), (512, 1))
    assert_size_stride(arg22_1, (384, 512), (512, 1))
    assert_size_stride(arg23_1, (384, 512), (512, 1))
    assert_size_stride(arg24_1, (512, 384), (384, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (1024, 512), (512, 1))
    assert_size_stride(arg27_1, (1024, 512), (512, 1))
    assert_size_stride(arg28_1, (512, 1024), (1024, 1))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (384, 512), (512, 1))
    assert_size_stride(arg31_1, (384, 512), (512, 1))
    assert_size_stride(arg32_1, (384, 512), (512, 1))
    assert_size_stride(arg33_1, (512, 384), (384, 1))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (1024, 512), (512, 1))
    assert_size_stride(arg36_1, (1024, 512), (512, 1))
    assert_size_stride(arg37_1, (512, 1024), (1024, 1))
    assert_size_stride(arg38_1, (512, ), (1, ))
    assert_size_stride(arg39_1, (384, 512), (512, 1))
    assert_size_stride(arg40_1, (384, 512), (512, 1))
    assert_size_stride(arg41_1, (384, 512), (512, 1))
    assert_size_stride(arg42_1, (512, 384), (384, 1))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (1024, 512), (512, 1))
    assert_size_stride(arg45_1, (1024, 512), (512, 1))
    assert_size_stride(arg46_1, (512, 1024), (1024, 1))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (384, 512), (512, 1))
    assert_size_stride(arg49_1, (384, 512), (512, 1))
    assert_size_stride(arg50_1, (384, 512), (512, 1))
    assert_size_stride(arg51_1, (512, 384), (384, 1))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (1024, 512), (512, 1))
    assert_size_stride(arg54_1, (1024, 512), (512, 1))
    assert_size_stride(arg55_1, (512, 1024), (1024, 1))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (384, 512), (512, 1))
    assert_size_stride(arg58_1, (384, 512), (512, 1))
    assert_size_stride(arg59_1, (384, 512), (512, 1))
    assert_size_stride(arg60_1, (512, 384), (384, 1))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (1024, 512), (512, 1))
    assert_size_stride(arg63_1, (1024, 512), (512, 1))
    assert_size_stride(arg64_1, (512, 1024), (1024, 1))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (384, 512), (512, 1))
    assert_size_stride(arg67_1, (384, 512), (512, 1))
    assert_size_stride(arg68_1, (384, 512), (512, 1))
    assert_size_stride(arg69_1, (512, 384), (384, 1))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (1024, 512), (512, 1))
    assert_size_stride(arg72_1, (1024, 512), (512, 1))
    assert_size_stride(arg73_1, (512, 1024), (1024, 1))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (16, 128), (128, 1))
    assert_size_stride(arg77_1, (384, 512), (512, 1))
    assert_size_stride(arg78_1, (384, 512), (512, 1))
    assert_size_stride(arg79_1, (384, 512), (512, 1))
    assert_size_stride(arg80_1, (512, 384), (384, 1))
    assert_size_stride(arg81_1, (32, 6), (6, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (384, 512), (512, 1))
    assert_size_stride(arg84_1, (384, 512), (512, 1))
    assert_size_stride(arg85_1, (384, 512), (512, 1))
    assert_size_stride(arg86_1, (512, 384), (384, 1))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (1024, 512), (512, 1))
    assert_size_stride(arg89_1, (1024, 512), (512, 1))
    assert_size_stride(arg90_1, (512, 1024), (1024, 1))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (384, 512), (512, 1))
    assert_size_stride(arg93_1, (384, 512), (512, 1))
    assert_size_stride(arg94_1, (384, 512), (512, 1))
    assert_size_stride(arg95_1, (512, 384), (384, 1))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg97_1, (384, 512), (512, 1))
    assert_size_stride(arg98_1, (384, 512), (512, 1))
    assert_size_stride(arg99_1, (384, 512), (512, 1))
    assert_size_stride(arg100_1, (512, 384), (384, 1))
    assert_size_stride(arg101_1, (512, ), (1, ))
    assert_size_stride(arg102_1, (1024, 512), (512, 1))
    assert_size_stride(arg103_1, (1024, 512), (512, 1))
    assert_size_stride(arg104_1, (512, 1024), (1024, 1))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (384, 512), (512, 1))
    assert_size_stride(arg107_1, (384, 512), (512, 1))
    assert_size_stride(arg108_1, (384, 512), (512, 1))
    assert_size_stride(arg109_1, (512, 384), (384, 1))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (384, 512), (512, 1))
    assert_size_stride(arg112_1, (384, 512), (512, 1))
    assert_size_stride(arg113_1, (384, 512), (512, 1))
    assert_size_stride(arg114_1, (512, 384), (384, 1))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (1024, 512), (512, 1))
    assert_size_stride(arg117_1, (1024, 512), (512, 1))
    assert_size_stride(arg118_1, (512, 1024), (1024, 1))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (384, 512), (512, 1))
    assert_size_stride(arg121_1, (384, 512), (512, 1))
    assert_size_stride(arg122_1, (384, 512), (512, 1))
    assert_size_stride(arg123_1, (512, 384), (384, 1))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (384, 512), (512, 1))
    assert_size_stride(arg126_1, (384, 512), (512, 1))
    assert_size_stride(arg127_1, (384, 512), (512, 1))
    assert_size_stride(arg128_1, (512, 384), (384, 1))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (1024, 512), (512, 1))
    assert_size_stride(arg131_1, (1024, 512), (512, 1))
    assert_size_stride(arg132_1, (512, 1024), (1024, 1))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (384, 512), (512, 1))
    assert_size_stride(arg135_1, (384, 512), (512, 1))
    assert_size_stride(arg136_1, (384, 512), (512, 1))
    assert_size_stride(arg137_1, (512, 384), (384, 1))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (384, 512), (512, 1))
    assert_size_stride(arg140_1, (384, 512), (512, 1))
    assert_size_stride(arg141_1, (384, 512), (512, 1))
    assert_size_stride(arg142_1, (512, 384), (384, 1))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (1024, 512), (512, 1))
    assert_size_stride(arg145_1, (1024, 512), (512, 1))
    assert_size_stride(arg146_1, (512, 1024), (1024, 1))
    assert_size_stride(arg147_1, (512, ), (1, ))
    assert_size_stride(arg148_1, (384, 512), (512, 1))
    assert_size_stride(arg149_1, (384, 512), (512, 1))
    assert_size_stride(arg150_1, (384, 512), (512, 1))
    assert_size_stride(arg151_1, (512, 384), (384, 1))
    assert_size_stride(arg152_1, (512, ), (1, ))
    assert_size_stride(arg153_1, (384, 512), (512, 1))
    assert_size_stride(arg154_1, (384, 512), (512, 1))
    assert_size_stride(arg155_1, (384, 512), (512, 1))
    assert_size_stride(arg156_1, (512, 384), (384, 1))
    assert_size_stride(arg157_1, (512, ), (1, ))
    assert_size_stride(arg158_1, (1024, 512), (512, 1))
    assert_size_stride(arg159_1, (1024, 512), (512, 1))
    assert_size_stride(arg160_1, (512, 1024), (1024, 1))
    assert_size_stride(arg161_1, (512, ), (1, ))
    assert_size_stride(arg162_1, (384, 512), (512, 1))
    assert_size_stride(arg163_1, (384, 512), (512, 1))
    assert_size_stride(arg164_1, (384, 512), (512, 1))
    assert_size_stride(arg165_1, (512, 384), (384, 1))
    assert_size_stride(arg166_1, (512, ), (1, ))
    assert_size_stride(arg167_1, (384, 512), (512, 1))
    assert_size_stride(arg168_1, (384, 512), (512, 1))
    assert_size_stride(arg169_1, (384, 512), (512, 1))
    assert_size_stride(arg170_1, (512, 384), (384, 1))
    assert_size_stride(arg171_1, (512, ), (1, ))
    assert_size_stride(arg172_1, (1024, 512), (512, 1))
    assert_size_stride(arg173_1, (1024, 512), (512, 1))
    assert_size_stride(arg174_1, (512, 1024), (1024, 1))
    assert_size_stride(arg175_1, (512, ), (1, ))
    assert_size_stride(arg176_1, (384, 512), (512, 1))
    assert_size_stride(arg177_1, (384, 512), (512, 1))
    assert_size_stride(arg178_1, (384, 512), (512, 1))
    assert_size_stride(arg179_1, (512, 384), (384, 1))
    assert_size_stride(arg180_1, (512, ), (1, ))
    assert_size_stride(arg181_1, (384, 512), (512, 1))
    assert_size_stride(arg182_1, (384, 512), (512, 1))
    assert_size_stride(arg183_1, (384, 512), (512, 1))
    assert_size_stride(arg184_1, (512, 384), (384, 1))
    assert_size_stride(arg185_1, (512, ), (1, ))
    assert_size_stride(arg186_1, (1024, 512), (512, 1))
    assert_size_stride(arg187_1, (1024, 512), (512, 1))
    assert_size_stride(arg188_1, (512, 1024), (1024, 1))
    assert_size_stride(arg189_1, (512, ), (1, ))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (250112, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((16, 128, 512), (65536, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds_1, pow_26, variance_17, add_52, rsqrt_17, hidden_states_77, normed_hidden_states_8], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0.run(arg0_1, arg1_1, arg82_1, buf1, 2048, 512, grid=grid(2048), stream=stream0)
        del arg82_1
        buf2 = empty_strided_cuda((2048, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (2048, 512), (512, 1), 0), reinterpret_tensor(arg77_1, (512, 384), (1, 512), 0), out=buf2)
        del arg77_1
        buf3 = empty_strided_cuda((2048, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (2048, 512), (512, 1), 0), reinterpret_tensor(arg78_1, (512, 384), (1, 512), 0), out=buf3)
        del arg78_1
        buf4 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf2, buf4, 786432, grid=grid(786432), stream=stream0)
        buf5 = reinterpret_tensor(buf2, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf3, buf5, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf6 = empty_strided_cuda((96, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf5, (96, 64, 128), (8192, 128, 1), 0), out=buf6)
        buf7 = reinterpret_tensor(buf6, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf6  # reuse
        buf11 = empty_strided_cuda((16, 6, 128, 128), (98304, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_17, softmax_8], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf7, arg81_1, buf11, 12288, 128, grid=grid(12288), stream=stream0)
        buf10 = reinterpret_tensor(buf5, (2048, 384), (384, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (2048, 512), (512, 1), 0), reinterpret_tensor(arg79_1, (512, 384), (1, 512), 0), out=buf10)
        del arg79_1
        buf12 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf10, buf12, 786432, grid=grid(786432), stream=stream0)
        buf13 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf12, (96, 128, 64), (8192, 64, 1), 0), out=buf13)
        buf14 = reinterpret_tensor(buf12, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf13, buf14, 786432, grid=grid(786432), stream=stream0)
        buf15 = reinterpret_tensor(buf1, (2048, 512), (512, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [attn_output_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (2048, 384), (384, 1), 0), reinterpret_tensor(arg80_1, (384, 512), (1, 384), 0), out=buf15)
        del arg80_1
        buf17 = empty_strided_cuda((16, 128, 512), (65536, 512, 1), torch.float32)
        buf20 = empty_strided_cuda((16, 128, 512), (65536, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds_1, hidden_states_80, pow_27, variance_18, add_57, rsqrt_18, hidden_states_81, normed_hidden_states_9, inputs_embeds, pow_1, variance, add, rsqrt, hidden_states_1, normed_hidden_states], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_5.run(arg0_1, arg1_1, buf15, arg87_1, arg7_1, buf17, buf20, 2048, 512, grid=grid(2048), stream=stream0)
        del arg7_1
        del arg87_1
        buf18 = reinterpret_tensor(buf14, (2048, 384), (384, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (2048, 512), (512, 1), 0), reinterpret_tensor(arg83_1, (512, 384), (1, 512), 0), out=buf18)
        del arg83_1
        buf21 = reinterpret_tensor(buf13, (2048, 384), (384, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (2048, 512), (512, 1), 0), reinterpret_tensor(arg2_1, (512, 384), (1, 512), 0), out=buf21)
        del arg2_1
        buf22 = empty_strided_cuda((2048, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (2048, 512), (512, 1), 0), reinterpret_tensor(arg3_1, (512, 384), (1, 512), 0), out=buf22)
        del arg3_1
        buf23 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf21, buf23, 786432, grid=grid(786432), stream=stream0)
        buf24 = reinterpret_tensor(buf21, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf22, buf24, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf25 = reinterpret_tensor(buf11, (96, 128, 128), (16384, 128, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf23, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf24, (96, 64, 128), (8192, 128, 1), 0), out=buf25)
        buf26 = reinterpret_tensor(buf25, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf25  # reuse
        buf30 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_1, softmax], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf26, arg6_1, buf30, 12288, 128, grid=grid(12288), stream=stream0)
        buf29 = reinterpret_tensor(buf24, (2048, 384), (384, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (2048, 512), (512, 1), 0), reinterpret_tensor(arg4_1, (512, 384), (1, 512), 0), out=buf29)
        del arg4_1
        buf31 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf29, buf31, 786432, grid=grid(786432), stream=stream0)
        buf32 = reinterpret_tensor(buf29, (96, 128, 64), (8192, 64, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf30, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf31, (96, 128, 64), (8192, 64, 1), 0), out=buf32)
        buf33 = reinterpret_tensor(buf31, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf32, buf33, 786432, grid=grid(786432), stream=stream0)
        buf34 = reinterpret_tensor(buf20, (2048, 512), (512, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [attn_output_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (2048, 384), (384, 1), 0), reinterpret_tensor(arg5_1, (384, 512), (1, 384), 0), out=buf34)
        del arg5_1
        buf36 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_4, pow_2, variance_1, add_5, rsqrt_1, hidden_states_5, forwarded_states], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7.run(arg0_1, arg1_1, buf34, arg11_1, buf36, 2048, 512, grid=grid(2048), stream=stream0)
        del arg11_1
        buf37 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (2048, 512), (512, 1), 0), reinterpret_tensor(arg8_1, (512, 1024), (1, 512), 0), out=buf37)
        del arg8_1
        buf38 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (2048, 512), (512, 1), 0), reinterpret_tensor(arg9_1, (512, 1024), (1, 512), 0), out=buf38)
        del arg9_1
        buf39 = reinterpret_tensor(buf37, (16, 128, 1024), (131072, 1024, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [mul_7, pow_3, mul_8, add_6, mul_9, tanh, add_7, hidden_gelu, hidden_states_6], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf39, buf38, 2097152, grid=grid(2097152), stream=stream0)
        buf40 = reinterpret_tensor(buf36, (2048, 512), (512, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg10_1, (1024, 512), (1, 1024), 0), out=buf40)
        del arg10_1
        buf42 = empty_strided_cuda((16, 128, 512), (65536, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_4, hidden_states_9, pow_4, variance_2, add_9, rsqrt_2, hidden_states_10, normed_hidden_states_1], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_9.run(arg0_1, arg1_1, buf34, buf40, arg16_1, buf42, 2048, 512, grid=grid(2048), stream=stream0)
        del arg16_1
        buf43 = reinterpret_tensor(buf33, (2048, 384), (384, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (2048, 512), (512, 1), 0), reinterpret_tensor(arg12_1, (512, 384), (1, 512), 0), out=buf43)
        del arg12_1
        buf44 = reinterpret_tensor(buf32, (2048, 384), (384, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (2048, 512), (512, 1), 0), reinterpret_tensor(arg13_1, (512, 384), (1, 512), 0), out=buf44)
        del arg13_1
        buf45 = reinterpret_tensor(buf22, (16, 6, 128, 64), (49152, 8192, 64, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [scores_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf43, buf45, 786432, grid=grid(786432), stream=stream0)
        buf46 = reinterpret_tensor(buf43, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [scores_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf44, buf46, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf47 = reinterpret_tensor(buf30, (96, 128, 128), (16384, 128, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf45, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf46, (96, 64, 128), (8192, 128, 1), 0), out=buf47)
        buf48 = reinterpret_tensor(buf47, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf47  # reuse
        buf52 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_3, softmax_1], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf48, arg6_1, buf52, 12288, 128, grid=grid(12288), stream=stream0)
        buf51 = reinterpret_tensor(buf46, (2048, 384), (384, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (2048, 512), (512, 1), 0), reinterpret_tensor(arg14_1, (512, 384), (1, 512), 0), out=buf51)
        del arg14_1
        buf53 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf51, buf53, 786432, grid=grid(786432), stream=stream0)
        buf54 = reinterpret_tensor(buf51, (96, 128, 64), (8192, 64, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf52, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf53, (96, 128, 64), (8192, 64, 1), 0), out=buf54)
        buf55 = reinterpret_tensor(buf53, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf54, buf55, 786432, grid=grid(786432), stream=stream0)
        buf56 = reinterpret_tensor(buf42, (2048, 512), (512, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (2048, 384), (384, 1), 0), reinterpret_tensor(arg15_1, (384, 512), (1, 384), 0), out=buf56)
        del arg15_1
        buf57 = reinterpret_tensor(buf34, (16, 128, 512), (65536, 512, 1), 0); del buf34  # reuse
        buf59 = empty_strided_cuda((16, 128, 512), (65536, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_4, hidden_states_9, hidden_states_13, pow_5, variance_3, add_11, rsqrt_3, hidden_states_14, forwarded_states_1], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10.run(buf57, arg0_1, arg1_1, buf40, buf56, arg20_1, buf59, 2048, 512, grid=grid(2048), stream=stream0)
        del arg20_1
        buf60 = reinterpret_tensor(buf39, (2048, 1024), (1024, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (2048, 512), (512, 1), 0), reinterpret_tensor(arg17_1, (512, 1024), (1, 512), 0), out=buf60)
        del arg17_1
        buf61 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (2048, 512), (512, 1), 0), reinterpret_tensor(arg18_1, (512, 1024), (1, 512), 0), out=buf61)
        del arg18_1
        buf62 = reinterpret_tensor(buf60, (16, 128, 1024), (131072, 1024, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [mul_16, pow_6, mul_17, add_12, mul_18, tanh_1, add_13, hidden_gelu_1, hidden_states_15], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf62, buf61, 2097152, grid=grid(2097152), stream=stream0)
        buf63 = reinterpret_tensor(buf59, (2048, 512), (512, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg19_1, (1024, 512), (1, 1024), 0), out=buf63)
        del arg19_1
        buf65 = reinterpret_tensor(buf56, (16, 128, 512), (65536, 512, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, pow_7, variance_4, add_15, rsqrt_4, hidden_states_19, normed_hidden_states_2], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf57, buf63, arg25_1, buf65, 2048, 512, grid=grid(2048), stream=stream0)
        del arg25_1
        buf66 = reinterpret_tensor(buf55, (2048, 384), (384, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (2048, 512), (512, 1), 0), reinterpret_tensor(arg21_1, (512, 384), (1, 512), 0), out=buf66)
        del arg21_1
        buf67 = reinterpret_tensor(buf54, (2048, 384), (384, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (2048, 512), (512, 1), 0), reinterpret_tensor(arg22_1, (512, 384), (1, 512), 0), out=buf67)
        del arg22_1
        buf68 = reinterpret_tensor(buf44, (16, 6, 128, 64), (49152, 8192, 64, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf66, buf68, 786432, grid=grid(786432), stream=stream0)
        buf69 = reinterpret_tensor(buf66, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf67, buf69, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf70 = reinterpret_tensor(buf52, (96, 128, 128), (16384, 128, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf69, (96, 64, 128), (8192, 128, 1), 0), out=buf70)
        buf71 = reinterpret_tensor(buf70, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf70  # reuse
        buf75 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_5, softmax_2], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf71, arg6_1, buf75, 12288, 128, grid=grid(12288), stream=stream0)
        buf74 = reinterpret_tensor(buf69, (2048, 384), (384, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (2048, 512), (512, 1), 0), reinterpret_tensor(arg23_1, (512, 384), (1, 512), 0), out=buf74)
        del arg23_1
        buf76 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf74, buf76, 786432, grid=grid(786432), stream=stream0)
        buf77 = reinterpret_tensor(buf74, (96, 128, 64), (8192, 64, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf75, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf76, (96, 128, 64), (8192, 64, 1), 0), out=buf77)
        buf78 = reinterpret_tensor(buf76, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf77, buf78, 786432, grid=grid(786432), stream=stream0)
        buf79 = reinterpret_tensor(buf65, (2048, 512), (512, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [attn_output_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (2048, 384), (384, 1), 0), reinterpret_tensor(arg24_1, (384, 512), (1, 384), 0), out=buf79)
        del arg24_1
        buf81 = reinterpret_tensor(buf40, (16, 128, 512), (65536, 512, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_22, pow_8, variance_5, add_17, rsqrt_5, hidden_states_23, forwarded_states_2], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf57, buf63, buf79, arg29_1, buf81, 2048, 512, grid=grid(2048), stream=stream0)
        del arg29_1
        buf82 = reinterpret_tensor(buf62, (2048, 1024), (1024, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (2048, 512), (512, 1), 0), reinterpret_tensor(arg26_1, (512, 1024), (1, 512), 0), out=buf82)
        del arg26_1
        buf83 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (2048, 512), (512, 1), 0), reinterpret_tensor(arg27_1, (512, 1024), (1, 512), 0), out=buf83)
        del arg27_1
        buf84 = reinterpret_tensor(buf82, (16, 128, 1024), (131072, 1024, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [mul_25, pow_9, mul_26, add_18, mul_27, tanh_2, add_19, hidden_gelu_2, hidden_states_24], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf84, buf83, 2097152, grid=grid(2097152), stream=stream0)
        buf85 = reinterpret_tensor(buf81, (2048, 512), (512, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg28_1, (1024, 512), (1, 1024), 0), out=buf85)
        del arg28_1
        buf87 = empty_strided_cuda((16, 128, 512), (65536, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_22, hidden_states_27, pow_10, variance_6, add_21, rsqrt_6, hidden_states_28, normed_hidden_states_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf57, buf63, buf79, buf85, arg34_1, buf87, 2048, 512, grid=grid(2048), stream=stream0)
        del arg34_1
        buf88 = reinterpret_tensor(buf78, (2048, 384), (384, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (2048, 512), (512, 1), 0), reinterpret_tensor(arg30_1, (512, 384), (1, 512), 0), out=buf88)
        del arg30_1
        buf89 = reinterpret_tensor(buf77, (2048, 384), (384, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (2048, 512), (512, 1), 0), reinterpret_tensor(arg31_1, (512, 384), (1, 512), 0), out=buf89)
        del arg31_1
        buf90 = reinterpret_tensor(buf67, (16, 6, 128, 64), (49152, 8192, 64, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf88, buf90, 786432, grid=grid(786432), stream=stream0)
        buf91 = reinterpret_tensor(buf88, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf89, buf91, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf92 = reinterpret_tensor(buf75, (96, 128, 128), (16384, 128, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf91, (96, 64, 128), (8192, 128, 1), 0), out=buf92)
        buf93 = reinterpret_tensor(buf92, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf92  # reuse
        buf97 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_7, softmax_3], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf93, arg6_1, buf97, 12288, 128, grid=grid(12288), stream=stream0)
        buf96 = reinterpret_tensor(buf91, (2048, 384), (384, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (2048, 512), (512, 1), 0), reinterpret_tensor(arg32_1, (512, 384), (1, 512), 0), out=buf96)
        del arg32_1
        buf98 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf96, buf98, 786432, grid=grid(786432), stream=stream0)
        buf99 = reinterpret_tensor(buf96, (96, 128, 64), (8192, 64, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf97, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf98, (96, 128, 64), (8192, 64, 1), 0), out=buf99)
        buf100 = reinterpret_tensor(buf98, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [contiguous_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf99, buf100, 786432, grid=grid(786432), stream=stream0)
        buf101 = reinterpret_tensor(buf87, (2048, 512), (512, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [attn_output_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (2048, 384), (384, 1), 0), reinterpret_tensor(arg33_1, (384, 512), (1, 384), 0), out=buf101)
        del arg33_1
        buf102 = reinterpret_tensor(buf101, (16, 128, 512), (65536, 512, 1), 0); del buf101  # reuse
        buf104 = empty_strided_cuda((16, 128, 512), (65536, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_22, hidden_states_27, hidden_states_31, pow_11, variance_7, add_23, rsqrt_7, hidden_states_32, forwarded_states_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_14.run(buf102, buf57, buf63, buf79, buf85, arg38_1, buf104, 2048, 512, grid=grid(2048), stream=stream0)
        del arg38_1
        buf105 = reinterpret_tensor(buf84, (2048, 1024), (1024, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (2048, 512), (512, 1), 0), reinterpret_tensor(arg35_1, (512, 1024), (1, 512), 0), out=buf105)
        del arg35_1
        buf106 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (2048, 512), (512, 1), 0), reinterpret_tensor(arg36_1, (512, 1024), (1, 512), 0), out=buf106)
        del arg36_1
        buf107 = reinterpret_tensor(buf105, (16, 128, 1024), (131072, 1024, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [mul_34, pow_12, mul_35, add_24, mul_36, tanh_3, add_25, hidden_gelu_3, hidden_states_33], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf107, buf106, 2097152, grid=grid(2097152), stream=stream0)
        buf108 = reinterpret_tensor(buf104, (2048, 512), (512, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg37_1, (1024, 512), (1, 1024), 0), out=buf108)
        del arg37_1
        buf110 = reinterpret_tensor(buf85, (16, 128, 512), (65536, 512, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_36, pow_13, variance_8, add_27, rsqrt_8, hidden_states_37, normed_hidden_states_4], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf102, buf108, arg43_1, buf110, 2048, 512, grid=grid(2048), stream=stream0)
        del arg43_1
        buf111 = reinterpret_tensor(buf100, (2048, 384), (384, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf110, (2048, 512), (512, 1), 0), reinterpret_tensor(arg39_1, (512, 384), (1, 512), 0), out=buf111)
        del arg39_1
        buf112 = reinterpret_tensor(buf99, (2048, 384), (384, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf110, (2048, 512), (512, 1), 0), reinterpret_tensor(arg40_1, (512, 384), (1, 512), 0), out=buf112)
        del arg40_1
        buf113 = reinterpret_tensor(buf89, (16, 6, 128, 64), (49152, 8192, 64, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf111, buf113, 786432, grid=grid(786432), stream=stream0)
        buf114 = reinterpret_tensor(buf111, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf112, buf114, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf115 = reinterpret_tensor(buf97, (96, 128, 128), (16384, 128, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf113, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf114, (96, 64, 128), (8192, 128, 1), 0), out=buf115)
        buf116 = reinterpret_tensor(buf115, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf115  # reuse
        buf120 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_9, softmax_4], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf116, arg6_1, buf120, 12288, 128, grid=grid(12288), stream=stream0)
        buf119 = reinterpret_tensor(buf114, (2048, 384), (384, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf110, (2048, 512), (512, 1), 0), reinterpret_tensor(arg41_1, (512, 384), (1, 512), 0), out=buf119)
        del arg41_1
        buf121 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf119, buf121, 786432, grid=grid(786432), stream=stream0)
        buf122 = reinterpret_tensor(buf119, (96, 128, 64), (8192, 64, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf121, (96, 128, 64), (8192, 64, 1), 0), out=buf122)
        buf123 = reinterpret_tensor(buf121, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf122, buf123, 786432, grid=grid(786432), stream=stream0)
        buf124 = reinterpret_tensor(buf110, (2048, 512), (512, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [attn_output_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (2048, 384), (384, 1), 0), reinterpret_tensor(arg42_1, (384, 512), (1, 384), 0), out=buf124)
        del arg42_1
        buf126 = reinterpret_tensor(buf79, (16, 128, 512), (65536, 512, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_40, pow_14, variance_9, add_29, rsqrt_9, hidden_states_41, forwarded_states_4], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf102, buf108, buf124, arg47_1, buf126, 2048, 512, grid=grid(2048), stream=stream0)
        del arg47_1
        buf127 = reinterpret_tensor(buf107, (2048, 1024), (1024, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (2048, 512), (512, 1), 0), reinterpret_tensor(arg44_1, (512, 1024), (1, 512), 0), out=buf127)
        del arg44_1
        buf128 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (2048, 512), (512, 1), 0), reinterpret_tensor(arg45_1, (512, 1024), (1, 512), 0), out=buf128)
        del arg45_1
        buf129 = reinterpret_tensor(buf127, (16, 128, 1024), (131072, 1024, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [mul_43, pow_15, mul_44, add_30, mul_45, tanh_4, add_31, hidden_gelu_4, hidden_states_42], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf129, buf128, 2097152, grid=grid(2097152), stream=stream0)
        buf130 = reinterpret_tensor(buf126, (2048, 512), (512, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg46_1, (1024, 512), (1, 1024), 0), out=buf130)
        del arg46_1
        buf132 = reinterpret_tensor(buf63, (16, 128, 512), (65536, 512, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_40, hidden_states_45, pow_16, variance_10, add_33, rsqrt_10, hidden_states_46, normed_hidden_states_5], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf102, buf108, buf124, buf130, arg52_1, buf132, 2048, 512, grid=grid(2048), stream=stream0)
        del arg52_1
        buf133 = reinterpret_tensor(buf123, (2048, 384), (384, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (2048, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 384), (1, 512), 0), out=buf133)
        del arg48_1
        buf134 = reinterpret_tensor(buf122, (2048, 384), (384, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (2048, 512), (512, 1), 0), reinterpret_tensor(arg49_1, (512, 384), (1, 512), 0), out=buf134)
        del arg49_1
        buf135 = reinterpret_tensor(buf112, (16, 6, 128, 64), (49152, 8192, 64, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [scores_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf133, buf135, 786432, grid=grid(786432), stream=stream0)
        buf136 = reinterpret_tensor(buf133, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [scores_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf134, buf136, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf137 = reinterpret_tensor(buf120, (96, 128, 128), (16384, 128, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf135, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf136, (96, 64, 128), (8192, 128, 1), 0), out=buf137)
        buf138 = reinterpret_tensor(buf137, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf137  # reuse
        buf142 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_11, softmax_5], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf138, arg6_1, buf142, 12288, 128, grid=grid(12288), stream=stream0)
        buf141 = reinterpret_tensor(buf136, (2048, 384), (384, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (2048, 512), (512, 1), 0), reinterpret_tensor(arg50_1, (512, 384), (1, 512), 0), out=buf141)
        del arg50_1
        buf143 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf141, buf143, 786432, grid=grid(786432), stream=stream0)
        buf144 = reinterpret_tensor(buf141, (96, 128, 64), (8192, 64, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf142, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf143, (96, 128, 64), (8192, 64, 1), 0), out=buf144)
        buf145 = reinterpret_tensor(buf143, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf144, buf145, 786432, grid=grid(786432), stream=stream0)
        buf146 = reinterpret_tensor(buf132, (2048, 512), (512, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [attn_output_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (2048, 384), (384, 1), 0), reinterpret_tensor(arg51_1, (384, 512), (1, 384), 0), out=buf146)
        del arg51_1
        buf147 = buf102; del buf102  # reuse
        buf149 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_40, hidden_states_45, hidden_states_49, pow_17, variance_11, add_35, rsqrt_11, hidden_states_50, forwarded_states_5], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_15.run(buf147, buf108, buf124, buf130, buf146, arg56_1, buf149, 2048, 512, grid=grid(2048), stream=stream0)
        del arg56_1
        buf150 = reinterpret_tensor(buf129, (2048, 1024), (1024, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (2048, 512), (512, 1), 0), reinterpret_tensor(arg53_1, (512, 1024), (1, 512), 0), out=buf150)
        del arg53_1
        buf151 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (2048, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 1024), (1, 512), 0), out=buf151)
        del arg54_1
        buf152 = reinterpret_tensor(buf150, (16, 128, 1024), (131072, 1024, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [mul_52, pow_18, mul_53, add_36, mul_54, tanh_5, add_37, hidden_gelu_5, hidden_states_51], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf152, buf151, 2097152, grid=grid(2097152), stream=stream0)
        buf153 = reinterpret_tensor(buf149, (2048, 512), (512, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_53], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg55_1, (1024, 512), (1, 1024), 0), out=buf153)
        del arg55_1
        buf155 = reinterpret_tensor(buf146, (16, 128, 512), (65536, 512, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_54, pow_19, variance_12, add_39, rsqrt_12, hidden_states_55, normed_hidden_states_6], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf147, buf153, arg61_1, buf155, 2048, 512, grid=grid(2048), stream=stream0)
        del arg61_1
        buf156 = reinterpret_tensor(buf145, (2048, 384), (384, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (2048, 512), (512, 1), 0), reinterpret_tensor(arg57_1, (512, 384), (1, 512), 0), out=buf156)
        del arg57_1
        buf157 = reinterpret_tensor(buf144, (2048, 384), (384, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (2048, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 384), (1, 512), 0), out=buf157)
        del arg58_1
        buf158 = reinterpret_tensor(buf134, (16, 6, 128, 64), (49152, 8192, 64, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf156, buf158, 786432, grid=grid(786432), stream=stream0)
        buf159 = reinterpret_tensor(buf156, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf157, buf159, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf160 = reinterpret_tensor(buf142, (96, 128, 128), (16384, 128, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf158, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf159, (96, 64, 128), (8192, 128, 1), 0), out=buf160)
        buf161 = reinterpret_tensor(buf160, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf160  # reuse
        buf165 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_13, softmax_6], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf161, arg6_1, buf165, 12288, 128, grid=grid(12288), stream=stream0)
        buf164 = reinterpret_tensor(buf159, (2048, 384), (384, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (2048, 512), (512, 1), 0), reinterpret_tensor(arg59_1, (512, 384), (1, 512), 0), out=buf164)
        del arg59_1
        buf166 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf164, buf166, 786432, grid=grid(786432), stream=stream0)
        buf167 = reinterpret_tensor(buf164, (96, 128, 64), (8192, 64, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf165, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf166, (96, 128, 64), (8192, 64, 1), 0), out=buf167)
        buf168 = reinterpret_tensor(buf166, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [contiguous_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf167, buf168, 786432, grid=grid(786432), stream=stream0)
        buf169 = reinterpret_tensor(buf155, (2048, 512), (512, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [attn_output_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (2048, 384), (384, 1), 0), reinterpret_tensor(arg60_1, (384, 512), (1, 384), 0), out=buf169)
        del arg60_1
        buf171 = reinterpret_tensor(buf130, (16, 128, 512), (65536, 512, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_54, hidden_states_58, pow_20, variance_13, add_41, rsqrt_13, hidden_states_59, forwarded_states_6], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf147, buf153, buf169, arg65_1, buf171, 2048, 512, grid=grid(2048), stream=stream0)
        del arg65_1
        buf172 = reinterpret_tensor(buf152, (2048, 1024), (1024, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (2048, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 1024), (1, 512), 0), out=buf172)
        del arg62_1
        buf173 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (2048, 512), (512, 1), 0), reinterpret_tensor(arg63_1, (512, 1024), (1, 512), 0), out=buf173)
        del arg63_1
        buf174 = reinterpret_tensor(buf172, (16, 128, 1024), (131072, 1024, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [mul_61, pow_21, mul_62, add_42, mul_63, tanh_6, add_43, hidden_gelu_6, hidden_states_60], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf174, buf173, 2097152, grid=grid(2097152), stream=stream0)
        buf175 = reinterpret_tensor(buf171, (2048, 512), (512, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf174, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg64_1, (1024, 512), (1, 1024), 0), out=buf175)
        del arg64_1
        buf177 = reinterpret_tensor(buf124, (16, 128, 512), (65536, 512, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_54, hidden_states_58, hidden_states_63, pow_22, variance_14, add_45, rsqrt_14, hidden_states_64, normed_hidden_states_7], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf147, buf153, buf169, buf175, arg70_1, buf177, 2048, 512, grid=grid(2048), stream=stream0)
        del arg70_1
        buf178 = reinterpret_tensor(buf168, (2048, 384), (384, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (2048, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 384), (1, 512), 0), out=buf178)
        del arg66_1
        buf179 = reinterpret_tensor(buf167, (2048, 384), (384, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (2048, 512), (512, 1), 0), reinterpret_tensor(arg67_1, (512, 384), (1, 512), 0), out=buf179)
        del arg67_1
        buf180 = reinterpret_tensor(buf157, (16, 6, 128, 64), (49152, 8192, 64, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [scores_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf178, buf180, 786432, grid=grid(786432), stream=stream0)
        buf181 = reinterpret_tensor(buf178, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [scores_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf179, buf181, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf182 = reinterpret_tensor(buf165, (96, 128, 128), (16384, 128, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [scores_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf181, (96, 64, 128), (8192, 128, 1), 0), out=buf182)
        buf183 = reinterpret_tensor(buf182, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf182  # reuse
        buf187 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_2, position_bias, scores_15, softmax_7], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_6.run(buf183, arg6_1, buf187, 12288, 128, grid=grid(12288), stream=stream0)
        del arg6_1
        buf186 = reinterpret_tensor(buf181, (2048, 384), (384, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (2048, 512), (512, 1), 0), reinterpret_tensor(arg68_1, (512, 384), (1, 512), 0), out=buf186)
        del arg68_1
        buf188 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf186, buf188, 786432, grid=grid(786432), stream=stream0)
        buf189 = reinterpret_tensor(buf186, (96, 128, 64), (8192, 64, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf187, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf188, (96, 128, 64), (8192, 64, 1), 0), out=buf189)
        buf190 = reinterpret_tensor(buf188, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [contiguous_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf189, buf190, 786432, grid=grid(786432), stream=stream0)
        buf191 = reinterpret_tensor(buf177, (2048, 512), (512, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [attn_output_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (2048, 384), (384, 1), 0), reinterpret_tensor(arg69_1, (384, 512), (1, 384), 0), out=buf191)
        del arg69_1
        buf192 = buf147; del buf147  # reuse
        buf194 = reinterpret_tensor(buf108, (16, 128, 512), (65536, 512, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_54, hidden_states_58, hidden_states_63, hidden_states_67, pow_23, variance_15, add_47, rsqrt_15, hidden_states_68, forwarded_states_7], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_15.run(buf192, buf153, buf169, buf175, buf191, arg74_1, buf194, 2048, 512, grid=grid(2048), stream=stream0)
        del arg74_1
        buf195 = reinterpret_tensor(buf174, (2048, 1024), (1024, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf194, (2048, 512), (512, 1), 0), reinterpret_tensor(arg71_1, (512, 1024), (1, 512), 0), out=buf195)
        del arg71_1
        buf196 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf194, (2048, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 1024), (1, 512), 0), out=buf196)
        del arg72_1
        buf197 = reinterpret_tensor(buf195, (16, 128, 1024), (131072, 1024, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [mul_70, pow_24, mul_71, add_48, mul_72, tanh_7, add_49, hidden_gelu_7, hidden_states_69], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf197, buf196, 2097152, grid=grid(2097152), stream=stream0)
        buf198 = reinterpret_tensor(buf194, (2048, 512), (512, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_71], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 512), (1, 1024), 0), out=buf198)
        del arg73_1
        buf200 = reinterpret_tensor(buf191, (16, 128, 512), (65536, 512, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_72, pow_25, variance_16, add_51, rsqrt_16, hidden_states_73, hidden_states_74], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf192, buf198, arg75_1, buf200, 2048, 512, grid=grid(2048), stream=stream0)
        del arg75_1
        buf201 = reinterpret_tensor(buf190, (2048, 384), (384, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 384), (1, 512), 0), out=buf201)
        del arg84_1
        buf202 = reinterpret_tensor(buf189, (16, 6, 128, 64), (49152, 8192, 64, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf18, buf202, 786432, grid=grid(786432), stream=stream0)
        buf203 = reinterpret_tensor(buf18, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf201, buf203, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf204 = reinterpret_tensor(buf187, (96, 128, 128), (16384, 128, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf202, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf203, (96, 64, 128), (8192, 128, 1), 0), out=buf204)
        buf208 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [softmax_9], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf204, buf208, 12288, 128, grid=grid(12288), stream=stream0)
        buf207 = reinterpret_tensor(buf203, (2048, 384), (384, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg85_1, (512, 384), (1, 512), 0), out=buf207)
        del arg85_1
        buf209 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf207, buf209, 786432, grid=grid(786432), stream=stream0)
        buf210 = reinterpret_tensor(buf179, (96, 128, 64), (8192, 64, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf208, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf209, (96, 128, 64), (8192, 64, 1), 0), out=buf210)
        buf211 = reinterpret_tensor(buf209, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [contiguous_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf210, buf211, 786432, grid=grid(786432), stream=stream0)
        buf212 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [attn_output_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (2048, 384), (384, 1), 0), reinterpret_tensor(arg86_1, (384, 512), (1, 384), 0), out=buf212)
        del arg86_1
        buf214 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds_1, hidden_states_80, layer_output, pow_28, variance_19, add_60, rsqrt_19, hidden_states_84, forwarded_states_8], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_9.run(arg0_1, arg1_1, buf15, buf212, arg91_1, buf214, 2048, 512, grid=grid(2048), stream=stream0)
        del arg91_1
        buf215 = reinterpret_tensor(buf197, (2048, 1024), (1024, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [linear_64], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (2048, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 1024), (1, 512), 0), out=buf215)
        del arg88_1
        buf216 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (2048, 512), (512, 1), 0), reinterpret_tensor(arg89_1, (512, 1024), (1, 512), 0), out=buf216)
        del arg89_1
        buf217 = reinterpret_tensor(buf215, (16, 128, 1024), (131072, 1024, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [mul_87, pow_29, mul_88, add_61, mul_89, tanh_8, add_62, hidden_gelu_8, hidden_states_85], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf217, buf216, 2097152, grid=grid(2097152), stream=stream0)
        buf218 = reinterpret_tensor(buf214, (2048, 512), (512, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg90_1, (1024, 512), (1, 1024), 0), out=buf218)
        del arg90_1
        buf219 = reinterpret_tensor(buf15, (16, 128, 512), (65536, 512, 1), 0); del buf15  # reuse
        buf221 = reinterpret_tensor(buf175, (16, 128, 512), (65536, 512, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds_1, hidden_states_80, layer_output, hidden_states_88, pow_30, variance_20, add_64, rsqrt_20, hidden_states_89, normed_hidden_states_10], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10.run(buf219, arg0_1, arg1_1, buf212, buf218, arg96_1, buf221, 2048, 512, grid=grid(2048), stream=stream0)
        del arg0_1
        del arg1_1
        del arg96_1
        buf222 = reinterpret_tensor(buf211, (2048, 384), (384, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (2048, 512), (512, 1), 0), reinterpret_tensor(arg92_1, (512, 384), (1, 512), 0), out=buf222)
        del arg92_1
        buf223 = reinterpret_tensor(buf210, (2048, 384), (384, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (2048, 512), (512, 1), 0), reinterpret_tensor(arg93_1, (512, 384), (1, 512), 0), out=buf223)
        del arg93_1
        buf224 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf222, buf224, 786432, grid=grid(786432), stream=stream0)
        buf225 = reinterpret_tensor(buf222, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf223, buf225, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf226 = reinterpret_tensor(buf208, (96, 128, 128), (16384, 128, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [scores_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf224, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf225, (96, 64, 128), (8192, 128, 1), 0), out=buf226)
        buf227 = reinterpret_tensor(buf226, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf226  # reuse
        buf231 = reinterpret_tensor(buf204, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_21, softmax_10], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf227, arg81_1, buf231, 12288, 128, grid=grid(12288), stream=stream0)
        buf230 = reinterpret_tensor(buf225, (2048, 384), (384, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [linear_69], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (2048, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 384), (1, 512), 0), out=buf230)
        del arg94_1
        buf232 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf230, buf232, 786432, grid=grid(786432), stream=stream0)
        buf233 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf231, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf232, (96, 128, 64), (8192, 64, 1), 0), out=buf233)
        buf234 = reinterpret_tensor(buf232, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [contiguous_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf233, buf234, 786432, grid=grid(786432), stream=stream0)
        buf235 = reinterpret_tensor(buf221, (2048, 512), (512, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [attn_output_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (2048, 384), (384, 1), 0), reinterpret_tensor(arg95_1, (384, 512), (1, 384), 0), out=buf235)
        del arg95_1
        buf237 = reinterpret_tensor(buf218, (16, 128, 512), (65536, 512, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_92, pow_31, variance_21, add_66, rsqrt_21, hidden_states_93, normed_hidden_states_11], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf219, buf235, arg101_1, buf237, 2048, 512, grid=grid(2048), stream=stream0)
        del arg101_1
        buf238 = reinterpret_tensor(buf234, (2048, 384), (384, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (2048, 512), (512, 1), 0), reinterpret_tensor(arg97_1, (512, 384), (1, 512), 0), out=buf238)
        del arg97_1
        buf239 = reinterpret_tensor(buf233, (2048, 384), (384, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 384), (1, 512), 0), out=buf239)
        del arg98_1
        buf240 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf238, buf240, 786432, grid=grid(786432), stream=stream0)
        buf241 = reinterpret_tensor(buf238, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [scores_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf239, buf241, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf242 = reinterpret_tensor(buf231, (96, 128, 128), (16384, 128, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [scores_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf240, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf241, (96, 64, 128), (8192, 128, 1), 0), out=buf242)
        buf246 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [softmax_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf242, buf246, 12288, 128, grid=grid(12288), stream=stream0)
        buf245 = reinterpret_tensor(buf241, (2048, 384), (384, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [linear_73], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg99_1, (512, 384), (1, 512), 0), out=buf245)
        del arg99_1
        buf247 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf245, buf247, 786432, grid=grid(786432), stream=stream0)
        buf248 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf247, (96, 128, 64), (8192, 64, 1), 0), out=buf248)
        buf249 = reinterpret_tensor(buf247, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf248, buf249, 786432, grid=grid(786432), stream=stream0)
        buf250 = reinterpret_tensor(buf237, (2048, 512), (512, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [attn_output_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (2048, 384), (384, 1), 0), reinterpret_tensor(arg100_1, (384, 512), (1, 384), 0), out=buf250)
        del arg100_1
        buf252 = reinterpret_tensor(buf212, (16, 128, 512), (65536, 512, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_92, layer_output_1, pow_32, variance_22, add_68, rsqrt_22, hidden_states_96, forwarded_states_9], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf219, buf235, buf250, arg105_1, buf252, 2048, 512, grid=grid(2048), stream=stream0)
        del arg105_1
        buf253 = reinterpret_tensor(buf217, (2048, 1024), (1024, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [linear_75], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (2048, 512), (512, 1), 0), reinterpret_tensor(arg102_1, (512, 1024), (1, 512), 0), out=buf253)
        del arg102_1
        buf254 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (2048, 512), (512, 1), 0), reinterpret_tensor(arg103_1, (512, 1024), (1, 512), 0), out=buf254)
        del arg103_1
        buf255 = reinterpret_tensor(buf253, (16, 128, 1024), (131072, 1024, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [mul_98, pow_33, mul_99, add_69, mul_100, tanh_9, add_70, hidden_gelu_9, hidden_states_97], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf255, buf254, 2097152, grid=grid(2097152), stream=stream0)
        buf256 = reinterpret_tensor(buf252, (2048, 512), (512, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_99], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg104_1, (1024, 512), (1, 1024), 0), out=buf256)
        del arg104_1
        buf258 = reinterpret_tensor(buf169, (16, 128, 512), (65536, 512, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_92, layer_output_1, hidden_states_100, pow_34, variance_23, add_72, rsqrt_23, hidden_states_101, normed_hidden_states_12], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf219, buf235, buf250, buf256, arg110_1, buf258, 2048, 512, grid=grid(2048), stream=stream0)
        del arg110_1
        buf259 = reinterpret_tensor(buf249, (2048, 384), (384, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (2048, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 384), (1, 512), 0), out=buf259)
        del arg106_1
        buf260 = reinterpret_tensor(buf248, (2048, 384), (384, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (2048, 512), (512, 1), 0), reinterpret_tensor(arg107_1, (512, 384), (1, 512), 0), out=buf260)
        del arg107_1
        buf261 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf259, buf261, 786432, grid=grid(786432), stream=stream0)
        buf262 = reinterpret_tensor(buf259, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf260, buf262, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf263 = reinterpret_tensor(buf246, (96, 128, 128), (16384, 128, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf261, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf262, (96, 64, 128), (8192, 128, 1), 0), out=buf263)
        buf264 = reinterpret_tensor(buf263, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf263  # reuse
        buf268 = reinterpret_tensor(buf242, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_25, softmax_12], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf264, arg81_1, buf268, 12288, 128, grid=grid(12288), stream=stream0)
        buf267 = reinterpret_tensor(buf262, (2048, 384), (384, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [linear_80], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (2048, 512), (512, 1), 0), reinterpret_tensor(arg108_1, (512, 384), (1, 512), 0), out=buf267)
        del arg108_1
        buf269 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf267, buf269, 786432, grid=grid(786432), stream=stream0)
        buf270 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf268, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf269, (96, 128, 64), (8192, 64, 1), 0), out=buf270)
        buf271 = reinterpret_tensor(buf269, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [contiguous_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf270, buf271, 786432, grid=grid(786432), stream=stream0)
        buf272 = reinterpret_tensor(buf258, (2048, 512), (512, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [attn_output_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf271, (2048, 384), (384, 1), 0), reinterpret_tensor(arg109_1, (384, 512), (1, 384), 0), out=buf272)
        del arg109_1
        buf273 = buf219; del buf219  # reuse
        buf275 = reinterpret_tensor(buf153, (16, 128, 512), (65536, 512, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_92, layer_output_1, hidden_states_100, hidden_states_104, pow_35, variance_24, add_74, rsqrt_24, hidden_states_105, normed_hidden_states_13], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_15.run(buf273, buf235, buf250, buf256, buf272, arg115_1, buf275, 2048, 512, grid=grid(2048), stream=stream0)
        del arg115_1
        buf276 = reinterpret_tensor(buf271, (2048, 384), (384, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (2048, 512), (512, 1), 0), reinterpret_tensor(arg111_1, (512, 384), (1, 512), 0), out=buf276)
        del arg111_1
        buf277 = reinterpret_tensor(buf270, (2048, 384), (384, 1), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [linear_83], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 384), (1, 512), 0), out=buf277)
        del arg112_1
        buf278 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf276, buf278, 786432, grid=grid(786432), stream=stream0)
        buf279 = reinterpret_tensor(buf276, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [scores_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf277, buf279, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf280 = reinterpret_tensor(buf268, (96, 128, 128), (16384, 128, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [scores_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf279, (96, 64, 128), (8192, 128, 1), 0), out=buf280)
        buf284 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [softmax_13], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf280, buf284, 12288, 128, grid=grid(12288), stream=stream0)
        buf283 = reinterpret_tensor(buf279, (2048, 384), (384, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [linear_84], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg113_1, (512, 384), (1, 512), 0), out=buf283)
        del arg113_1
        buf285 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf283, buf285, 786432, grid=grid(786432), stream=stream0)
        buf286 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf284, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf285, (96, 128, 64), (8192, 64, 1), 0), out=buf286)
        buf287 = reinterpret_tensor(buf285, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [contiguous_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf286, buf287, 786432, grid=grid(786432), stream=stream0)
        buf288 = reinterpret_tensor(buf275, (2048, 512), (512, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [attn_output_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (2048, 384), (384, 1), 0), reinterpret_tensor(arg114_1, (384, 512), (1, 384), 0), out=buf288)
        del arg114_1
        buf290 = reinterpret_tensor(buf272, (16, 128, 512), (65536, 512, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [layer_output_2, pow_36, variance_25, add_76, rsqrt_25, hidden_states_108, forwarded_states_10], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf273, buf288, arg119_1, buf290, 2048, 512, grid=grid(2048), stream=stream0)
        del arg119_1
        buf291 = reinterpret_tensor(buf255, (2048, 1024), (1024, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (2048, 512), (512, 1), 0), reinterpret_tensor(arg116_1, (512, 1024), (1, 512), 0), out=buf291)
        del arg116_1
        buf292 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (2048, 512), (512, 1), 0), reinterpret_tensor(arg117_1, (512, 1024), (1, 512), 0), out=buf292)
        del arg117_1
        buf293 = reinterpret_tensor(buf291, (16, 128, 1024), (131072, 1024, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [mul_109, pow_37, mul_110, add_77, mul_111, tanh_10, add_78, hidden_gelu_10, hidden_states_109], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf293, buf292, 2097152, grid=grid(2097152), stream=stream0)
        buf294 = reinterpret_tensor(buf290, (2048, 512), (512, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg118_1, (1024, 512), (1, 1024), 0), out=buf294)
        del arg118_1
        buf296 = reinterpret_tensor(buf256, (16, 128, 512), (65536, 512, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [layer_output_2, hidden_states_112, pow_38, variance_26, add_80, rsqrt_26, hidden_states_113, normed_hidden_states_14], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf273, buf288, buf294, arg124_1, buf296, 2048, 512, grid=grid(2048), stream=stream0)
        del arg124_1
        buf297 = reinterpret_tensor(buf287, (2048, 384), (384, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (2048, 512), (512, 1), 0), reinterpret_tensor(arg120_1, (512, 384), (1, 512), 0), out=buf297)
        del arg120_1
        buf298 = reinterpret_tensor(buf286, (2048, 384), (384, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [linear_90], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (2048, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 384), (1, 512), 0), out=buf298)
        del arg121_1
        buf299 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf297, buf299, 786432, grid=grid(786432), stream=stream0)
        buf300 = reinterpret_tensor(buf297, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [scores_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf298, buf300, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf301 = reinterpret_tensor(buf284, (96, 128, 128), (16384, 128, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [scores_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf299, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf300, (96, 64, 128), (8192, 128, 1), 0), out=buf301)
        buf302 = reinterpret_tensor(buf301, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf301  # reuse
        buf306 = reinterpret_tensor(buf280, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_29, softmax_14], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf302, arg81_1, buf306, 12288, 128, grid=grid(12288), stream=stream0)
        buf305 = reinterpret_tensor(buf300, (2048, 384), (384, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (2048, 512), (512, 1), 0), reinterpret_tensor(arg122_1, (512, 384), (1, 512), 0), out=buf305)
        del arg122_1
        buf307 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf305, buf307, 786432, grid=grid(786432), stream=stream0)
        buf308 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf306, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf307, (96, 128, 64), (8192, 64, 1), 0), out=buf308)
        buf309 = reinterpret_tensor(buf307, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf308, buf309, 786432, grid=grid(786432), stream=stream0)
        buf310 = reinterpret_tensor(buf296, (2048, 512), (512, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [attn_output_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (2048, 384), (384, 1), 0), reinterpret_tensor(arg123_1, (384, 512), (1, 384), 0), out=buf310)
        del arg123_1
        buf312 = reinterpret_tensor(buf250, (16, 128, 512), (65536, 512, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [layer_output_2, hidden_states_112, hidden_states_116, pow_39, variance_27, add_82, rsqrt_27, hidden_states_117, normed_hidden_states_15], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf273, buf288, buf294, buf310, arg129_1, buf312, 2048, 512, grid=grid(2048), stream=stream0)
        del arg129_1
        buf313 = reinterpret_tensor(buf309, (2048, 384), (384, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (2048, 512), (512, 1), 0), reinterpret_tensor(arg125_1, (512, 384), (1, 512), 0), out=buf313)
        del arg125_1
        buf314 = reinterpret_tensor(buf308, (2048, 384), (384, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [linear_94], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg126_1, (512, 384), (1, 512), 0), out=buf314)
        del arg126_1
        buf315 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf313, buf315, 786432, grid=grid(786432), stream=stream0)
        buf316 = reinterpret_tensor(buf313, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf314, buf316, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf317 = reinterpret_tensor(buf306, (96, 128, 128), (16384, 128, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [scores_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf315, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf316, (96, 64, 128), (8192, 128, 1), 0), out=buf317)
        buf321 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [softmax_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf317, buf321, 12288, 128, grid=grid(12288), stream=stream0)
        buf320 = reinterpret_tensor(buf316, (2048, 384), (384, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [linear_95], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg127_1, (512, 384), (1, 512), 0), out=buf320)
        del arg127_1
        buf322 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [matmul_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf320, buf322, 786432, grid=grid(786432), stream=stream0)
        buf323 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf321, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf322, (96, 128, 64), (8192, 64, 1), 0), out=buf323)
        buf324 = reinterpret_tensor(buf322, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [contiguous_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf323, buf324, 786432, grid=grid(786432), stream=stream0)
        buf325 = reinterpret_tensor(buf312, (2048, 512), (512, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [attn_output_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf324, (2048, 384), (384, 1), 0), reinterpret_tensor(arg128_1, (384, 512), (1, 384), 0), out=buf325)
        del arg128_1
        buf326 = buf273; del buf273  # reuse
        buf328 = reinterpret_tensor(buf235, (16, 128, 512), (65536, 512, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [layer_output_2, hidden_states_112, hidden_states_116, layer_output_3, pow_40, variance_28, add_84, rsqrt_28, hidden_states_120, forwarded_states_11], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_15.run(buf326, buf288, buf294, buf310, buf325, arg133_1, buf328, 2048, 512, grid=grid(2048), stream=stream0)
        del arg133_1
        buf329 = reinterpret_tensor(buf293, (2048, 1024), (1024, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [linear_97], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf328, (2048, 512), (512, 1), 0), reinterpret_tensor(arg130_1, (512, 1024), (1, 512), 0), out=buf329)
        del arg130_1
        buf330 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf328, (2048, 512), (512, 1), 0), reinterpret_tensor(arg131_1, (512, 1024), (1, 512), 0), out=buf330)
        del arg131_1
        buf331 = reinterpret_tensor(buf329, (16, 128, 1024), (131072, 1024, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [mul_120, pow_41, mul_121, add_85, mul_122, tanh_11, add_86, hidden_gelu_11, hidden_states_121], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf331, buf330, 2097152, grid=grid(2097152), stream=stream0)
        buf332 = reinterpret_tensor(buf328, (2048, 512), (512, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_123], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf331, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg132_1, (1024, 512), (1, 1024), 0), out=buf332)
        del arg132_1
        buf334 = reinterpret_tensor(buf325, (16, 128, 512), (65536, 512, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_124, pow_42, variance_29, add_88, rsqrt_29, hidden_states_125, normed_hidden_states_16], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf326, buf332, arg138_1, buf334, 2048, 512, grid=grid(2048), stream=stream0)
        del arg138_1
        buf335 = reinterpret_tensor(buf324, (2048, 384), (384, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [linear_100], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (2048, 512), (512, 1), 0), reinterpret_tensor(arg134_1, (512, 384), (1, 512), 0), out=buf335)
        del arg134_1
        buf336 = reinterpret_tensor(buf323, (2048, 384), (384, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [linear_101], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (2048, 512), (512, 1), 0), reinterpret_tensor(arg135_1, (512, 384), (1, 512), 0), out=buf336)
        del arg135_1
        buf337 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf335, buf337, 786432, grid=grid(786432), stream=stream0)
        buf338 = reinterpret_tensor(buf335, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [scores_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf336, buf338, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf339 = reinterpret_tensor(buf321, (96, 128, 128), (16384, 128, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [scores_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf337, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf338, (96, 64, 128), (8192, 128, 1), 0), out=buf339)
        buf340 = reinterpret_tensor(buf339, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf339  # reuse
        buf344 = reinterpret_tensor(buf317, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_33, softmax_16], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf340, arg81_1, buf344, 12288, 128, grid=grid(12288), stream=stream0)
        buf343 = reinterpret_tensor(buf338, (2048, 384), (384, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [linear_102], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (2048, 512), (512, 1), 0), reinterpret_tensor(arg136_1, (512, 384), (1, 512), 0), out=buf343)
        del arg136_1
        buf345 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [matmul_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf343, buf345, 786432, grid=grid(786432), stream=stream0)
        buf346 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf344, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf345, (96, 128, 64), (8192, 64, 1), 0), out=buf346)
        buf347 = reinterpret_tensor(buf345, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [contiguous_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf346, buf347, 786432, grid=grid(786432), stream=stream0)
        buf348 = reinterpret_tensor(buf334, (2048, 512), (512, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [attn_output_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf347, (2048, 384), (384, 1), 0), reinterpret_tensor(arg137_1, (384, 512), (1, 384), 0), out=buf348)
        del arg137_1
        buf350 = reinterpret_tensor(buf310, (16, 128, 512), (65536, 512, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_124, hidden_states_128, pow_43, variance_30, add_90, rsqrt_30, hidden_states_129, normed_hidden_states_17], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf326, buf332, buf348, arg143_1, buf350, 2048, 512, grid=grid(2048), stream=stream0)
        del arg143_1
        buf351 = reinterpret_tensor(buf347, (2048, 384), (384, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [linear_104], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (2048, 512), (512, 1), 0), reinterpret_tensor(arg139_1, (512, 384), (1, 512), 0), out=buf351)
        del arg139_1
        buf352 = reinterpret_tensor(buf346, (2048, 384), (384, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [linear_105], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg140_1, (512, 384), (1, 512), 0), out=buf352)
        del arg140_1
        buf353 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf351, buf353, 786432, grid=grid(786432), stream=stream0)
        buf354 = reinterpret_tensor(buf351, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [scores_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf352, buf354, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf355 = reinterpret_tensor(buf344, (96, 128, 128), (16384, 128, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [scores_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf353, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf354, (96, 64, 128), (8192, 128, 1), 0), out=buf355)
        buf359 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [softmax_17], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf355, buf359, 12288, 128, grid=grid(12288), stream=stream0)
        buf358 = reinterpret_tensor(buf354, (2048, 384), (384, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg141_1, (512, 384), (1, 512), 0), out=buf358)
        del arg141_1
        buf360 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [matmul_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf358, buf360, 786432, grid=grid(786432), stream=stream0)
        buf361 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf359, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf360, (96, 128, 64), (8192, 64, 1), 0), out=buf361)
        buf362 = reinterpret_tensor(buf360, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf361, buf362, 786432, grid=grid(786432), stream=stream0)
        buf363 = reinterpret_tensor(buf350, (2048, 512), (512, 1), 0); del buf350  # reuse
        # Topologically Sorted Source Nodes: [attn_output_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf362, (2048, 384), (384, 1), 0), reinterpret_tensor(arg142_1, (384, 512), (1, 384), 0), out=buf363)
        del arg142_1
        buf365 = reinterpret_tensor(buf294, (16, 128, 512), (65536, 512, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_124, hidden_states_128, layer_output_4, pow_44, variance_31, add_92, rsqrt_31, hidden_states_132, forwarded_states_12], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf326, buf332, buf348, buf363, arg147_1, buf365, 2048, 512, grid=grid(2048), stream=stream0)
        del arg147_1
        buf366 = reinterpret_tensor(buf331, (2048, 1024), (1024, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [linear_108], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (2048, 512), (512, 1), 0), reinterpret_tensor(arg144_1, (512, 1024), (1, 512), 0), out=buf366)
        del arg144_1
        buf367 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (2048, 512), (512, 1), 0), reinterpret_tensor(arg145_1, (512, 1024), (1, 512), 0), out=buf367)
        del arg145_1
        buf368 = reinterpret_tensor(buf366, (16, 128, 1024), (131072, 1024, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [mul_131, pow_45, mul_132, add_93, mul_133, tanh_12, add_94, hidden_gelu_12, hidden_states_133], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf368, buf367, 2097152, grid=grid(2097152), stream=stream0)
        buf369 = reinterpret_tensor(buf365, (2048, 512), (512, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_135], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg146_1, (1024, 512), (1, 1024), 0), out=buf369)
        del arg146_1
        buf370 = buf326; del buf326  # reuse
        buf372 = reinterpret_tensor(buf288, (16, 128, 512), (65536, 512, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_124, hidden_states_128, layer_output_4, hidden_states_136, pow_46, variance_32, add_96, rsqrt_32, hidden_states_137, normed_hidden_states_18], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_15.run(buf370, buf332, buf348, buf363, buf369, arg152_1, buf372, 2048, 512, grid=grid(2048), stream=stream0)
        del arg152_1
        buf373 = reinterpret_tensor(buf362, (2048, 384), (384, 1), 0); del buf362  # reuse
        # Topologically Sorted Source Nodes: [linear_111], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (2048, 512), (512, 1), 0), reinterpret_tensor(arg148_1, (512, 384), (1, 512), 0), out=buf373)
        del arg148_1
        buf374 = reinterpret_tensor(buf361, (2048, 384), (384, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [linear_112], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (2048, 512), (512, 1), 0), reinterpret_tensor(arg149_1, (512, 384), (1, 512), 0), out=buf374)
        del arg149_1
        buf375 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf373, buf375, 786432, grid=grid(786432), stream=stream0)
        buf376 = reinterpret_tensor(buf373, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [scores_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf374, buf376, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf377 = reinterpret_tensor(buf359, (96, 128, 128), (16384, 128, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [scores_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf375, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf376, (96, 64, 128), (8192, 128, 1), 0), out=buf377)
        buf378 = reinterpret_tensor(buf377, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf377  # reuse
        buf382 = reinterpret_tensor(buf355, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf355  # reuse
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_37, softmax_18], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf378, arg81_1, buf382, 12288, 128, grid=grid(12288), stream=stream0)
        buf381 = reinterpret_tensor(buf376, (2048, 384), (384, 1), 0); del buf376  # reuse
        # Topologically Sorted Source Nodes: [linear_113], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (2048, 512), (512, 1), 0), reinterpret_tensor(arg150_1, (512, 384), (1, 512), 0), out=buf381)
        del arg150_1
        buf383 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [matmul_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf381, buf383, 786432, grid=grid(786432), stream=stream0)
        buf384 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf382, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf383, (96, 128, 64), (8192, 64, 1), 0), out=buf384)
        buf385 = reinterpret_tensor(buf383, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [contiguous_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf384, buf385, 786432, grid=grid(786432), stream=stream0)
        buf386 = reinterpret_tensor(buf372, (2048, 512), (512, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [attn_output_37], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf385, (2048, 384), (384, 1), 0), reinterpret_tensor(arg151_1, (384, 512), (1, 384), 0), out=buf386)
        del arg151_1
        buf388 = reinterpret_tensor(buf369, (16, 128, 512), (65536, 512, 1), 0); del buf369  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_140, pow_47, variance_33, add_98, rsqrt_33, hidden_states_141, normed_hidden_states_19], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf370, buf386, arg157_1, buf388, 2048, 512, grid=grid(2048), stream=stream0)
        del arg157_1
        buf389 = reinterpret_tensor(buf385, (2048, 384), (384, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [linear_115], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf388, (2048, 512), (512, 1), 0), reinterpret_tensor(arg153_1, (512, 384), (1, 512), 0), out=buf389)
        del arg153_1
        buf390 = reinterpret_tensor(buf384, (2048, 384), (384, 1), 0); del buf384  # reuse
        # Topologically Sorted Source Nodes: [linear_116], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg154_1, (512, 384), (1, 512), 0), out=buf390)
        del arg154_1
        buf391 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf389, buf391, 786432, grid=grid(786432), stream=stream0)
        buf392 = reinterpret_tensor(buf389, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [scores_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf390, buf392, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf393 = reinterpret_tensor(buf382, (96, 128, 128), (16384, 128, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [scores_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf391, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf392, (96, 64, 128), (8192, 128, 1), 0), out=buf393)
        buf397 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [softmax_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf393, buf397, 12288, 128, grid=grid(12288), stream=stream0)
        buf396 = reinterpret_tensor(buf392, (2048, 384), (384, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [linear_117], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg155_1, (512, 384), (1, 512), 0), out=buf396)
        del arg155_1
        buf398 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [matmul_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf396, buf398, 786432, grid=grid(786432), stream=stream0)
        buf399 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf397, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf398, (96, 128, 64), (8192, 64, 1), 0), out=buf399)
        buf400 = reinterpret_tensor(buf398, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [contiguous_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf399, buf400, 786432, grid=grid(786432), stream=stream0)
        buf401 = reinterpret_tensor(buf388, (2048, 512), (512, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [attn_output_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (2048, 384), (384, 1), 0), reinterpret_tensor(arg156_1, (384, 512), (1, 384), 0), out=buf401)
        del arg156_1
        buf403 = reinterpret_tensor(buf363, (16, 128, 512), (65536, 512, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_140, layer_output_5, pow_48, variance_34, add_100, rsqrt_34, hidden_states_144, forwarded_states_13], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf370, buf386, buf401, arg161_1, buf403, 2048, 512, grid=grid(2048), stream=stream0)
        del arg161_1
        buf404 = reinterpret_tensor(buf368, (2048, 1024), (1024, 1), 0); del buf368  # reuse
        # Topologically Sorted Source Nodes: [linear_119], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf403, (2048, 512), (512, 1), 0), reinterpret_tensor(arg158_1, (512, 1024), (1, 512), 0), out=buf404)
        del arg158_1
        buf405 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf403, (2048, 512), (512, 1), 0), reinterpret_tensor(arg159_1, (512, 1024), (1, 512), 0), out=buf405)
        del arg159_1
        buf406 = reinterpret_tensor(buf404, (16, 128, 1024), (131072, 1024, 1), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [mul_142, pow_49, mul_143, add_101, mul_144, tanh_13, add_102, hidden_gelu_13, hidden_states_145], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf406, buf405, 2097152, grid=grid(2097152), stream=stream0)
        buf407 = reinterpret_tensor(buf403, (2048, 512), (512, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_147], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg160_1, (1024, 512), (1, 1024), 0), out=buf407)
        del arg160_1
        buf409 = reinterpret_tensor(buf348, (16, 128, 512), (65536, 512, 1), 0); del buf348  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_140, layer_output_5, hidden_states_148, pow_50, variance_35, add_104, rsqrt_35, hidden_states_149, normed_hidden_states_20], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf370, buf386, buf401, buf407, arg166_1, buf409, 2048, 512, grid=grid(2048), stream=stream0)
        del arg166_1
        buf410 = reinterpret_tensor(buf400, (2048, 384), (384, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [linear_122], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (2048, 512), (512, 1), 0), reinterpret_tensor(arg162_1, (512, 384), (1, 512), 0), out=buf410)
        del arg162_1
        buf411 = reinterpret_tensor(buf399, (2048, 384), (384, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [linear_123], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (2048, 512), (512, 1), 0), reinterpret_tensor(arg163_1, (512, 384), (1, 512), 0), out=buf411)
        del arg163_1
        buf412 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf410, buf412, 786432, grid=grid(786432), stream=stream0)
        buf413 = reinterpret_tensor(buf410, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [scores_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf411, buf413, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf414 = reinterpret_tensor(buf397, (96, 128, 128), (16384, 128, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [scores_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf412, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf413, (96, 64, 128), (8192, 128, 1), 0), out=buf414)
        buf415 = reinterpret_tensor(buf414, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf414  # reuse
        buf419 = reinterpret_tensor(buf393, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_41, softmax_20], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf415, arg81_1, buf419, 12288, 128, grid=grid(12288), stream=stream0)
        buf418 = reinterpret_tensor(buf413, (2048, 384), (384, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (2048, 512), (512, 1), 0), reinterpret_tensor(arg164_1, (512, 384), (1, 512), 0), out=buf418)
        del arg164_1
        buf420 = buf412; del buf412  # reuse
        # Topologically Sorted Source Nodes: [matmul_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf418, buf420, 786432, grid=grid(786432), stream=stream0)
        buf421 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf419, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf420, (96, 128, 64), (8192, 64, 1), 0), out=buf421)
        buf422 = reinterpret_tensor(buf420, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf420  # reuse
        # Topologically Sorted Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf421, buf422, 786432, grid=grid(786432), stream=stream0)
        buf423 = reinterpret_tensor(buf409, (2048, 512), (512, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [attn_output_41], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (2048, 384), (384, 1), 0), reinterpret_tensor(arg165_1, (384, 512), (1, 384), 0), out=buf423)
        del arg165_1
        buf424 = buf370; del buf370  # reuse
        buf426 = reinterpret_tensor(buf332, (16, 128, 512), (65536, 512, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_140, layer_output_5, hidden_states_148, hidden_states_152, pow_51, variance_36, add_106, rsqrt_36, hidden_states_153, normed_hidden_states_21], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_15.run(buf424, buf386, buf401, buf407, buf423, arg171_1, buf426, 2048, 512, grid=grid(2048), stream=stream0)
        del arg171_1
        buf427 = reinterpret_tensor(buf422, (2048, 384), (384, 1), 0); del buf422  # reuse
        # Topologically Sorted Source Nodes: [linear_126], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf426, (2048, 512), (512, 1), 0), reinterpret_tensor(arg167_1, (512, 384), (1, 512), 0), out=buf427)
        del arg167_1
        buf428 = reinterpret_tensor(buf421, (2048, 384), (384, 1), 0); del buf421  # reuse
        # Topologically Sorted Source Nodes: [linear_127], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg168_1, (512, 384), (1, 512), 0), out=buf428)
        del arg168_1
        buf429 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf427, buf429, 786432, grid=grid(786432), stream=stream0)
        buf430 = reinterpret_tensor(buf427, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [scores_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf428, buf430, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf431 = reinterpret_tensor(buf419, (96, 128, 128), (16384, 128, 1), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [scores_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf429, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf430, (96, 64, 128), (8192, 128, 1), 0), out=buf431)
        buf435 = buf415; del buf415  # reuse
        # Topologically Sorted Source Nodes: [softmax_21], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf431, buf435, 12288, 128, grid=grid(12288), stream=stream0)
        buf434 = reinterpret_tensor(buf430, (2048, 384), (384, 1), 0); del buf430  # reuse
        # Topologically Sorted Source Nodes: [linear_128], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg169_1, (512, 384), (1, 512), 0), out=buf434)
        del arg169_1
        buf436 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [matmul_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf434, buf436, 786432, grid=grid(786432), stream=stream0)
        buf437 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_43], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf435, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf436, (96, 128, 64), (8192, 64, 1), 0), out=buf437)
        buf438 = reinterpret_tensor(buf436, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [contiguous_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf437, buf438, 786432, grid=grid(786432), stream=stream0)
        buf439 = reinterpret_tensor(buf426, (2048, 512), (512, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [attn_output_43], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf438, (2048, 384), (384, 1), 0), reinterpret_tensor(arg170_1, (384, 512), (1, 384), 0), out=buf439)
        del arg170_1
        buf441 = reinterpret_tensor(buf423, (16, 128, 512), (65536, 512, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [layer_output_6, pow_52, variance_37, add_108, rsqrt_37, hidden_states_156, forwarded_states_14], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf424, buf439, arg175_1, buf441, 2048, 512, grid=grid(2048), stream=stream0)
        del arg175_1
        buf442 = reinterpret_tensor(buf406, (2048, 1024), (1024, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [linear_130], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf441, (2048, 512), (512, 1), 0), reinterpret_tensor(arg172_1, (512, 1024), (1, 512), 0), out=buf442)
        del arg172_1
        buf443 = buf405; del buf405  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf441, (2048, 512), (512, 1), 0), reinterpret_tensor(arg173_1, (512, 1024), (1, 512), 0), out=buf443)
        del arg173_1
        buf444 = reinterpret_tensor(buf442, (16, 128, 1024), (131072, 1024, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [mul_153, pow_53, mul_154, add_109, mul_155, tanh_14, add_110, hidden_gelu_14, hidden_states_157], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf444, buf443, 2097152, grid=grid(2097152), stream=stream0)
        buf445 = reinterpret_tensor(buf441, (2048, 512), (512, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_159], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf444, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg174_1, (1024, 512), (1, 1024), 0), out=buf445)
        del arg174_1
        buf447 = reinterpret_tensor(buf407, (16, 128, 512), (65536, 512, 1), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [layer_output_6, hidden_states_160, pow_54, variance_38, add_112, rsqrt_38, hidden_states_161, normed_hidden_states_22], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf424, buf439, buf445, arg180_1, buf447, 2048, 512, grid=grid(2048), stream=stream0)
        del arg180_1
        buf448 = reinterpret_tensor(buf438, (2048, 384), (384, 1), 0); del buf438  # reuse
        # Topologically Sorted Source Nodes: [linear_133], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (2048, 512), (512, 1), 0), reinterpret_tensor(arg176_1, (512, 384), (1, 512), 0), out=buf448)
        del arg176_1
        buf449 = reinterpret_tensor(buf437, (2048, 384), (384, 1), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [linear_134], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (2048, 512), (512, 1), 0), reinterpret_tensor(arg177_1, (512, 384), (1, 512), 0), out=buf449)
        del arg177_1
        buf450 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf448, buf450, 786432, grid=grid(786432), stream=stream0)
        buf451 = reinterpret_tensor(buf448, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [scores_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf449, buf451, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf452 = reinterpret_tensor(buf435, (96, 128, 128), (16384, 128, 1), 0); del buf435  # reuse
        # Topologically Sorted Source Nodes: [scores_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf450, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf451, (96, 64, 128), (8192, 128, 1), 0), out=buf452)
        buf453 = reinterpret_tensor(buf452, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf452  # reuse
        buf457 = reinterpret_tensor(buf431, (16, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf431  # reuse
        # Topologically Sorted Source Nodes: [sub_2, extended_attention_mask_5, position_bias_1, scores_45, softmax_22], Original ATen: [aten.rsub, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_rsub_3.run(buf453, arg81_1, buf457, 12288, 128, grid=grid(12288), stream=stream0)
        del arg81_1
        buf456 = reinterpret_tensor(buf451, (2048, 384), (384, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [linear_135], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (2048, 512), (512, 1), 0), reinterpret_tensor(arg178_1, (512, 384), (1, 512), 0), out=buf456)
        del arg178_1
        buf458 = buf450; del buf450  # reuse
        # Topologically Sorted Source Nodes: [matmul_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf456, buf458, 786432, grid=grid(786432), stream=stream0)
        buf459 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf457, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf458, (96, 128, 64), (8192, 64, 1), 0), out=buf459)
        buf460 = reinterpret_tensor(buf458, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf458  # reuse
        # Topologically Sorted Source Nodes: [contiguous_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf459, buf460, 786432, grid=grid(786432), stream=stream0)
        buf461 = reinterpret_tensor(buf447, (2048, 512), (512, 1), 0); del buf447  # reuse
        # Topologically Sorted Source Nodes: [attn_output_45], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf460, (2048, 384), (384, 1), 0), reinterpret_tensor(arg179_1, (384, 512), (1, 384), 0), out=buf461)
        del arg179_1
        buf463 = reinterpret_tensor(buf401, (16, 128, 512), (65536, 512, 1), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [layer_output_6, hidden_states_160, hidden_states_164, pow_55, variance_39, add_114, rsqrt_39, hidden_states_165, normed_hidden_states_23], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf424, buf439, buf445, buf461, arg185_1, buf463, 2048, 512, grid=grid(2048), stream=stream0)
        del arg185_1
        buf464 = reinterpret_tensor(buf460, (2048, 384), (384, 1), 0); del buf460  # reuse
        # Topologically Sorted Source Nodes: [linear_137], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (2048, 512), (512, 1), 0), reinterpret_tensor(arg181_1, (512, 384), (1, 512), 0), out=buf464)
        del arg181_1
        buf465 = reinterpret_tensor(buf459, (2048, 384), (384, 1), 0); del buf459  # reuse
        # Topologically Sorted Source Nodes: [linear_138], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg182_1, (512, 384), (1, 512), 0), out=buf465)
        del arg182_1
        buf466 = empty_strided_cuda((16, 6, 128, 64), (49152, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scores_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf464, buf466, 786432, grid=grid(786432), stream=stream0)
        buf467 = reinterpret_tensor(buf464, (16, 6, 64, 128), (49152, 8192, 128, 1), 0); del buf464  # reuse
        # Topologically Sorted Source Nodes: [scores_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf465, buf467, 6144, 128, grid=grid(6144, 128), stream=stream0)
        buf468 = reinterpret_tensor(buf457, (96, 128, 128), (16384, 128, 1), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [scores_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf466, (96, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf467, (96, 64, 128), (8192, 128, 1), 0), out=buf468)
        buf472 = buf453; del buf453  # reuse
        # Topologically Sorted Source Nodes: [softmax_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf468, buf472, 12288, 128, grid=grid(12288), stream=stream0)
        del buf468
        buf471 = reinterpret_tensor(buf467, (2048, 384), (384, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [linear_139], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 512), (512, 1), 0), reinterpret_tensor(arg183_1, (512, 384), (1, 512), 0), out=buf471)
        del arg183_1
        buf473 = buf466; del buf466  # reuse
        # Topologically Sorted Source Nodes: [matmul_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf471, buf473, 786432, grid=grid(786432), stream=stream0)
        buf474 = empty_strided_cuda((96, 128, 64), (8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf472, (96, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf473, (96, 128, 64), (8192, 64, 1), 0), out=buf474)
        del buf472
        buf475 = reinterpret_tensor(buf473, (16, 128, 6, 64), (49152, 384, 64, 1), 0); del buf473  # reuse
        # Topologically Sorted Source Nodes: [contiguous_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf474, buf475, 786432, grid=grid(786432), stream=stream0)
        del buf474
        buf476 = reinterpret_tensor(buf463, (2048, 512), (512, 1), 0); del buf463  # reuse
        # Topologically Sorted Source Nodes: [attn_output_47], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (2048, 384), (384, 1), 0), reinterpret_tensor(arg184_1, (384, 512), (1, 384), 0), out=buf476)
        del arg184_1
        del buf475
        buf477 = buf424; del buf424  # reuse
        buf479 = reinterpret_tensor(buf386, (16, 128, 512), (65536, 512, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [layer_output_6, hidden_states_160, hidden_states_164, layer_output_7, pow_56, variance_40, add_116, rsqrt_40, hidden_states_168, forwarded_states_15], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_15.run(buf477, buf439, buf445, buf461, buf476, arg189_1, buf479, 2048, 512, grid=grid(2048), stream=stream0)
        del arg189_1
        del buf439
        del buf445
        del buf461
        buf480 = reinterpret_tensor(buf444, (2048, 1024), (1024, 1), 0); del buf444  # reuse
        # Topologically Sorted Source Nodes: [linear_141], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf479, (2048, 512), (512, 1), 0), reinterpret_tensor(arg186_1, (512, 1024), (1, 512), 0), out=buf480)
        del arg186_1
        buf481 = buf443; del buf443  # reuse
        # Topologically Sorted Source Nodes: [hidden_linear_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf479, (2048, 512), (512, 1), 0), reinterpret_tensor(arg187_1, (512, 1024), (1, 512), 0), out=buf481)
        del arg187_1
        buf482 = reinterpret_tensor(buf480, (16, 128, 1024), (131072, 1024, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [mul_164, pow_57, mul_165, add_117, mul_166, tanh_15, add_118, hidden_gelu_15, hidden_states_169], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf482, buf481, 2097152, grid=grid(2097152), stream=stream0)
        del buf481
        buf483 = reinterpret_tensor(buf479, (2048, 512), (512, 1), 0); del buf479  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_171], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf482, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg188_1, (1024, 512), (1, 1024), 0), out=buf483)
        del arg188_1
        del buf482
        buf485 = reinterpret_tensor(buf476, (16, 128, 512), (65536, 512, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_172, pow_58, variance_41, add_120, rsqrt_41, hidden_states_173, hidden_states_174], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf477, buf483, arg190_1, buf485, 2048, 512, grid=grid(2048), stream=stream0)
        del arg190_1
        del buf477
        del buf483
        buf486 = empty_strided_cuda((2048, 250112), (250112, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (2048, 512), (512, 1), 0), reinterpret_tensor(arg191_1, (512, 250112), (1, 512), 0), out=buf486)
        del arg191_1
        del buf485
        buf487 = empty_strided_cuda((2048, 1), (1, 2048), torch.float32)
        buf488 = empty_strided_cuda((2048, 1), (1, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_17.run(buf486, buf487, buf488, 2048, 250112, grid=grid(2048), stream=stream0)
        buf489 = empty_strided_cuda((), (), torch.float32)
        buf491 = buf489; del buf489  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_18.run(buf491, arg76_1, buf486, buf487, buf488, 1, 2048, grid=grid(1), stream=stream0)
        del arg76_1
        del buf487
        del buf488
    return (buf491, reinterpret_tensor(buf486, (16, 128, 250112), (32014336, 250112, 1), 0), reinterpret_tensor(buf3, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf10, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf201, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf207, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf223, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf230, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf239, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf245, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf260, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf267, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf277, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf283, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf298, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf305, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf314, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf320, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf336, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf343, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf352, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf358, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf374, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf381, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf390, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf396, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf411, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf418, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf428, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf434, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf449, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf456, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf465, (16, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf471, (16, 6, 128, 64), (49152, 64, 384, 1), 0), buf200, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((250112, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((16, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg77_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((32, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((250112, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MT5ForConditionalGeneration', benchmark_compiled_module)
