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


# kernel path: /tmp/torchinductor_sahanp/ui/cuif2ljvib45h5w3k46ctm3m6olmisupsbfcu3n75bdpj7ohrebu.py
# Topologically Sorted Source Nodes: [inputs_embeds, position_embeddings, embeddings, mean, sub_1, sub, pow_1, variance, add, sqrt, hidden_states_1, mul, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add => add_1
#   embeddings => add
#   embeddings_1 => add_2
#   hidden_states_1 => div
#   inputs_embeds => embedding
#   mean => mean
#   mul => mul
#   position_embeddings => embedding_1
#   pow_1 => pow_1
#   sqrt => sqrt
#   sub => sub
#   sub_1 => sub_1
#   variance => mean_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %arg0_1, 0), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg1_1), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %mean), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %mean), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-07), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %sqrt), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, %div), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %arg5_1), kwargs = {})
triton_red_fused_add_div_embedding_mean_mul_pow_sqrt_sub_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_embedding_mean_mul_pow_sqrt_sub_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x0 = xindex % 512
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp1 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert((0 <= tmp4) & (tmp4 < 50265), "index out of bounds: 0 <= tmp4 < 50265")
        tmp6 = tl.load(in_ptr1 + (r2 + (768*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.full([XBLOCK, RBLOCK], 512, tl.int32)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp7 < 0
        tmp11 = tl.where(tmp10, tmp9, tmp7)
        tl.device_assert((0 <= tmp11) & (tmp11 < 512), "index out of bounds: 0 <= tmp11 < 512")
        tmp13 = tl.load(in_ptr3 + (r2 + (768*tmp11)), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp6 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    _tmp36 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp18 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp19 = tmp0 + tmp18
        tmp20 = tmp0 < 0
        tmp21 = tl.where(tmp20, tmp19, tmp0)
        tl.device_assert((0 <= tmp21) & (tmp21 < 50265), "index out of bounds: 0 <= tmp21 < 50265")
        tmp23 = tl.load(in_ptr1 + (r2 + (768*tmp21)), rmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.full([XBLOCK, RBLOCK], 512, tl.int32)
        tmp25 = tmp7 + tmp24
        tmp26 = tmp7 < 0
        tmp27 = tl.where(tmp26, tmp25, tmp7)
        tl.device_assert((0 <= tmp27) & (tmp27 < 512), "index out of bounds: 0 <= tmp27 < 512")
        tmp29 = tl.load(in_ptr3 + (r2 + (768*tmp27)), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tmp23 + tmp29
        tmp31 = 768.0
        tmp32 = tmp16 / tmp31
        tmp33 = tmp30 - tmp32
        tmp34 = tmp33 * tmp33
        tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
        tmp37 = _tmp36 + tmp35
        _tmp36 = tl.where(rmask, tmp37, _tmp36)
    tmp36 = tl.sum(_tmp36, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp38 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp61 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp39 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp40 = tmp0 + tmp39
        tmp41 = tmp0 < 0
        tmp42 = tl.where(tmp41, tmp40, tmp0)
        tl.device_assert((0 <= tmp42) & (tmp42 < 50265), "index out of bounds: 0 <= tmp42 < 50265")
        tmp44 = tl.load(in_ptr1 + (r2 + (768*tmp42)), rmask, eviction_policy='evict_first', other=0.0)
        tmp45 = tl.full([XBLOCK, RBLOCK], 512, tl.int32)
        tmp46 = tmp7 + tmp45
        tmp47 = tmp7 < 0
        tmp48 = tl.where(tmp47, tmp46, tmp7)
        tl.device_assert((0 <= tmp48) & (tmp48 < 512), "index out of bounds: 0 <= tmp48 < 512")
        tmp50 = tl.load(in_ptr3 + (r2 + (768*tmp48)), rmask, eviction_policy='evict_first', other=0.0)
        tmp51 = tmp44 + tmp50
        tmp52 = 768.0
        tmp53 = tmp16 / tmp52
        tmp54 = tmp51 - tmp53
        tmp55 = tmp36 / tmp52
        tmp56 = 1e-07
        tmp57 = tmp55 + tmp56
        tmp58 = libdevice.sqrt(tmp57)
        tmp59 = tmp54 / tmp58
        tmp60 = tmp38 * tmp59
        tmp62 = tmp60 + tmp61
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp62, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/va/cvacpg5behuri4gghti7xwudrabsk275yl2orbghfm2xbpphrlhe.py
# Topologically Sorted Source Nodes: [query_layer_1, scale, query_layer_2, attention_scores], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
# Source node to ATen node mapping:
#   attention_scores => clone
#   query_layer_1 => add_3
#   query_layer_2 => div_1
#   scale => full_default_1
# Graph fragment:
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, %permute_2), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 8.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_3, %full_default_1), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_div_sqrt_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_div_sqrt_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*x2) + (2304*x1) + (1179648*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4i/c4it4gemllvjgtnu64j3h6dmmoyq7f5xxn3mz2hodrcodg6qwwaf.py
# Topologically Sorted Source Nodes: [attention_scores], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attention_scores => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64) % 12
    y2 = (yindex // 768)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (64 + y0 + (192*y1) + (2304*x3) + (1179648*y2)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (512*y4)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yj/cyjng4xsbcmpqxzsg2axbqkonxm5nrzov2eupce2zwcpcmmwpsfv.py
# Topologically Sorted Source Nodes: [masked_fill_, rmask, tensor_1, output, output_1], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
# Source node to ATen node mapping:
#   masked_fill_ => full_default_4, where_1
#   output => where
#   output_1 => amax, div_2, exp, sub_2, sum_1
#   rmask => full_default_2
#   tensor_1 => full_default_3
# Graph fragment:
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([16, 1, 512, 512], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%full_default_2, %full_default_3, %view_7), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where, [-1], True), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_2, %full_default_4, %div_2), kwargs = {})
triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 98304
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
    tmp1 = tl.full([1], False, tl.int1)
    tmp2 = -3.4028234663852886e+38
    tmp3 = tl.where(tmp1, tmp2, tmp0)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp4, 0))
    tmp7 = tmp3 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp8 / tmp11
    tmp13 = 0.0
    tmp14 = tl.where(tmp1, tmp13, tmp12)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kt/cktndr4bo6qhhh5r6exn7htev5xwt5aj6twz2hnrltcmqcjnglkj.py
# Topologically Sorted Source Nodes: [value_layer_1, context_layer], Original ATen: [aten.add, aten.clone]
# Source node to ATen node mapping:
#   context_layer => clone_2
#   value_layer_1 => add_4
# Graph fragment:
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, %permute_3), kwargs = {})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (192*x2) + (2304*x1) + (1179648*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2x/c2xwul74fmbqmp6kz4xnpzuatembhwld2crcac623yljbmk43z6m.py
# Topologically Sorted Source Nodes: [context_layer_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   context_layer_1 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768) % 512
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (32768*x1) + (393216*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6r/c6rioo6plys3ylg5fdryk5t2svw4fwnopc2cs6gnmgs7rae4ubnd.py
# Topologically Sorted Source Nodes: [add_4, mean_3, sub_3, sub_2, pow_2, variance_1, add_5, sqrt_2, hidden_states_5, mul_4, y_1], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_5
#   add_5 => add_6
#   hidden_states_5 => div_3
#   mean_3 => mean_2
#   mul_4 => mul_4
#   pow_2 => pow_2
#   sqrt_2 => sqrt_2
#   sub_2 => sub_3
#   sub_3 => sub_4
#   variance_1 => mean_3
#   y_1 => add_7
# Graph fragment:
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_14, %add_2), kwargs = {})
#   %mean_2 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_5, [-1], True), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %mean_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %mean_2), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_3, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-07), kwargs = {})
#   %sqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_4, %sqrt_2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg11_1, %div_3), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg12_1), kwargs = {})
triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    tmp17 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = 768.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp4 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp16 / tmp9
    tmp19 = 1e-07
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp11 / tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/i7/ci7lyuabvwljdcnfr3uil4daaous3mnsxkchro6q4mzhwfhapthk.py
# Topologically Sorted Source Nodes: [hidden_states_8], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_8 => add_8, erf, mul_5, mul_6, mul_7
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, 0.5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_8), kwargs = {})
triton_poi_fused_gelu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/2v/c2volb5qwuqr3iawv4l6mssljalhg7p2fuojjeudfyz7gay5znq3.py
# Topologically Sorted Source Nodes: [start_logits_1, start_loss], Original ATen: [aten.clone, aten._log_softmax]
# Source node to ATen node mapping:
#   start_logits_1 => clone_48
#   start_loss => amax_12, exp_12, sub_62, sum_13
# Graph fragment:
#   %clone_48 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_1,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_12 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_48, [1], True), kwargs = {})
#   %sub_62 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_48, %amax_12), kwargs = {})
#   %exp_12 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_62,), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_12, [1], True), kwargs = {})
triton_red_fused__log_softmax_clone_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_clone_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/d6/cd6acm4g3knf6j2gpay6enx3v2xifs4nnakjibiirdy2emo4t2bx.py
# Topologically Sorted Source Nodes: [end_logits_1, end_loss], Original ATen: [aten.clone, aten._log_softmax]
# Source node to ATen node mapping:
#   end_logits_1 => clone_49
#   end_loss => amax_13, exp_13, sub_64, sum_16
# Graph fragment:
#   %clone_49 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_2,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_13 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_49, [1], True), kwargs = {})
#   %sub_64 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_49, %amax_13), kwargs = {})
#   %exp_13 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_64,), kwargs = {})
#   %sum_16 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_13, [1], True), kwargs = {})
triton_red_fused__log_softmax_clone_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_clone_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/5q/c5qccny3ad6nfbqzqs2akfcmbtsij5t7ctwkvsgomifw6sf7fnla.py
# Topologically Sorted Source Nodes: [start_positions, start_loss, end_positions, end_loss, add_98, total_loss], Original ATen: [aten.clamp, aten.nll_loss_forward, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_98 => add_111
#   end_loss => convert_element_type_13, div_50, full_default_52, ne_4, ne_5, neg_1, sum_17, sum_18, where_27
#   end_positions => clamp_max_1, clamp_min_1
#   start_loss => convert_element_type_12, div_49, full_default_50, ne_1, ne_2, neg, sum_14, sum_15, where_25
#   start_positions => clamp_max, clamp_min
#   total_loss => div_51
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg164_1, 0), kwargs = {})
#   %clamp_max : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 512), kwargs = {})
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max, 512), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_3,), kwargs = {})
#   %full_default_50 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_25 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_50), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_25,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max, 512), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_14, torch.float32), kwargs = {})
#   %div_49 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_15, %convert_element_type_12), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg165_1, 0), kwargs = {})
#   %clamp_max_1 : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 512), kwargs = {})
#   %ne_4 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max_1, 512), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_4,), kwargs = {})
#   %full_default_52 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_27 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %neg_1, %full_default_52), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_27,), kwargs = {})
#   %ne_5 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max_1, 512), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_5,), kwargs = {})
#   %convert_element_type_13 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_17, torch.float32), kwargs = {})
#   %div_50 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_18, %convert_element_type_13), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_49, %div_50), kwargs = {})
#   %div_51 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_111, 2), kwargs = {})
triton_per_fused_add_clamp_div_nll_loss_forward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_div_nll_loss_forward_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 512), (512, 1))
    assert_size_stride(arg1_1, (1, 512), (512, 1))
    assert_size_stride(arg2_1, (50265, 768), (768, 1))
    assert_size_stride(arg3_1, (512, 768), (768, 1))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (2304, 768), (768, 1))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, 768), (768, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (3072, 768), (768, 1))
    assert_size_stride(arg14_1, (3072, ), (1, ))
    assert_size_stride(arg15_1, (768, 3072), (3072, 1))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (2304, 768), (768, 1))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, 768), (768, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (3072, 768), (768, 1))
    assert_size_stride(arg27_1, (3072, ), (1, ))
    assert_size_stride(arg28_1, (768, 3072), (3072, 1))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (2304, 768), (768, 1))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, 768), (768, 1))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (3072, 768), (768, 1))
    assert_size_stride(arg40_1, (3072, ), (1, ))
    assert_size_stride(arg41_1, (768, 3072), (3072, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (2304, 768), (768, 1))
    assert_size_stride(arg46_1, (768, ), (1, ))
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
    assert_size_stride(arg58_1, (2304, 768), (768, 1))
    assert_size_stride(arg59_1, (768, ), (1, ))
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
    assert_size_stride(arg71_1, (2304, 768), (768, 1))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, 768), (768, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (3072, 768), (768, 1))
    assert_size_stride(arg79_1, (3072, ), (1, ))
    assert_size_stride(arg80_1, (768, 3072), (3072, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (2304, 768), (768, 1))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, 768), (768, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (3072, 768), (768, 1))
    assert_size_stride(arg92_1, (3072, ), (1, ))
    assert_size_stride(arg93_1, (768, 3072), (3072, 1))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (2304, 768), (768, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, 768), (768, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (3072, 768), (768, 1))
    assert_size_stride(arg105_1, (3072, ), (1, ))
    assert_size_stride(arg106_1, (768, 3072), (3072, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (2304, 768), (768, 1))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, 768), (768, 1))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (3072, 768), (768, 1))
    assert_size_stride(arg118_1, (3072, ), (1, ))
    assert_size_stride(arg119_1, (768, 3072), (3072, 1))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (2304, 768), (768, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, 768), (768, 1))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (3072, 768), (768, 1))
    assert_size_stride(arg131_1, (3072, ), (1, ))
    assert_size_stride(arg132_1, (768, 3072), (3072, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (2304, 768), (768, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, 768), (768, 1))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (3072, 768), (768, 1))
    assert_size_stride(arg144_1, (3072, ), (1, ))
    assert_size_stride(arg145_1, (768, 3072), (3072, 1))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (2304, 768), (768, 1))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (768, 768), (768, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (3072, 768), (768, 1))
    assert_size_stride(arg157_1, (3072, ), (1, ))
    assert_size_stride(arg158_1, (768, 3072), (3072, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (2, 768), (768, 1))
    assert_size_stride(arg163_1, (2, ), (1, ))
    assert_size_stride(arg164_1, (16, ), (1, ))
    assert_size_stride(arg165_1, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((16, 512, 768), (393216, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, position_embeddings, embeddings, mean, sub_1, sub, pow_1, variance, add, sqrt, hidden_states_1, mul, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_embedding_mean_mul_pow_sqrt_sub_0.run(arg0_1, arg2_1, arg1_1, arg3_1, arg4_1, arg5_1, buf2, 8192, 768, grid=grid(8192), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf3 = empty_strided_cuda((8192, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [qp], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (8192, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 2304), (1, 768), 0), out=buf3)
        del arg6_1
        buf4 = empty_strided_cuda((16, 12, 512, 64), (393216, 32768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [query_layer_1, scale, query_layer_2, attention_scores], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
        triton_poi_fused_add_clone_div_sqrt_1.run(buf3, arg7_1, buf4, 6291456, grid=grid(6291456), stream=stream0)
        del arg7_1
        buf5 = empty_strided_cuda((16, 12, 64, 512), (393216, 32768, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf3, buf5, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf6 = empty_strided_cuda((192, 512, 512), (262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention_scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf5, (192, 64, 512), (32768, 512, 1), 0), out=buf6)
        buf9 = empty_strided_cuda((16, 12, 512, 512), (3145728, 262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [masked_fill_, rmask, tensor_1, output, output_1], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf6, buf9, 98304, 512, grid=grid(98304), stream=stream0)
        buf10 = reinterpret_tensor(buf5, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [value_layer_1, context_layer], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_4.run(buf3, arg8_1, buf10, 6291456, grid=grid(6291456), stream=stream0)
        del arg8_1
        buf11 = reinterpret_tensor(buf4, (192, 512, 64), (32768, 64, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [context_layer], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (192, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf10, (192, 512, 64), (32768, 64, 1), 0), out=buf11)
        buf12 = reinterpret_tensor(buf10, (16, 512, 12, 64), (393216, 768, 64, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [context_layer_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf11, buf12, 6291456, grid=grid(6291456), stream=stream0)
        buf13 = reinterpret_tensor(buf11, (8192, 768), (768, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf12, (8192, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), out=buf13)
        del arg9_1
        buf16 = reinterpret_tensor(buf12, (16, 512, 768), (393216, 768, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [add_4, mean_3, sub_3, sub_2, pow_2, variance_1, add_5, sqrt_2, hidden_states_5, mul_4, y_1], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf13, arg10_1, buf2, arg11_1, arg12_1, buf16, 8192, 768, grid=grid(8192), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        buf17 = empty_strided_cuda((8192, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf16, (8192, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 3072), (1, 768), 0), out=buf17)
        del arg13_1
        buf18 = reinterpret_tensor(buf17, (16, 512, 3072), (1572864, 3072, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf18, arg14_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg14_1
        buf19 = reinterpret_tensor(buf2, (8192, 768), (768, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg15_1, (3072, 768), (1, 3072), 0), out=buf19)
        del arg15_1
        buf22 = reinterpret_tensor(buf13, (16, 512, 768), (393216, 768, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [add_7, mean_6, sub_5, sub_4, pow_3, variance_2, add_8, sqrt_3, hidden_states_11, mul_5, y_2], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf19, arg16_1, buf16, arg17_1, arg18_1, buf22, 8192, 768, grid=grid(8192), stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        buf23 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [qp_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (8192, 768), (768, 1), 0), reinterpret_tensor(arg19_1, (768, 2304), (1, 768), 0), out=buf23)
        del arg19_1
        buf24 = reinterpret_tensor(buf19, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [query_layer_4, scale_1, query_layer_5, attention_scores_1], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
        triton_poi_fused_add_clone_div_sqrt_1.run(buf23, arg20_1, buf24, 6291456, grid=grid(6291456), stream=stream0)
        del arg20_1
        buf25 = reinterpret_tensor(buf16, (16, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf23, buf25, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf26 = reinterpret_tensor(buf9, (192, 512, 512), (262144, 512, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf25, (192, 64, 512), (32768, 512, 1), 0), out=buf26)
        buf29 = reinterpret_tensor(buf6, (16, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__1, rmask_1, tensor_3, output_2, output_3], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf26, buf29, 98304, 512, grid=grid(98304), stream=stream0)
        buf30 = reinterpret_tensor(buf25, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [value_layer_3, context_layer_3], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_4.run(buf23, arg21_1, buf30, 6291456, grid=grid(6291456), stream=stream0)
        del arg21_1
        buf31 = reinterpret_tensor(buf24, (192, 512, 64), (32768, 64, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [context_layer_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (192, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf30, (192, 512, 64), (32768, 64, 1), 0), out=buf31)
        buf32 = reinterpret_tensor(buf30, (16, 512, 12, 64), (393216, 768, 64, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [context_layer_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf31, buf32, 6291456, grid=grid(6291456), stream=stream0)
        buf33 = reinterpret_tensor(buf31, (8192, 768), (768, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (8192, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 768), (1, 768), 0), out=buf33)
        del arg22_1
        buf36 = reinterpret_tensor(buf32, (16, 512, 768), (393216, 768, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [add_12, mean_9, sub_7, sub_6, pow_4, variance_3, add_13, sqrt_5, hidden_states_15, mul_7, y_3], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf33, arg23_1, buf22, arg24_1, arg25_1, buf36, 8192, 768, grid=grid(8192), stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        buf37 = reinterpret_tensor(buf18, (8192, 3072), (3072, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf36, (8192, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 3072), (1, 768), 0), out=buf37)
        del arg26_1
        buf38 = reinterpret_tensor(buf37, (16, 512, 3072), (1572864, 3072, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf38, arg27_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg27_1
        buf39 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg28_1, (3072, 768), (1, 3072), 0), out=buf39)
        del arg28_1
        buf42 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [add_15, mean_12, sub_9, sub_8, pow_5, variance_4, add_16, sqrt_6, hidden_states_21, mul_8, y_4], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf39, arg29_1, buf36, arg30_1, arg31_1, buf42, 8192, 768, grid=grid(8192), stream=stream0)
        del arg29_1
        del arg30_1
        del arg31_1
        buf43 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [qp_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (8192, 768), (768, 1), 0), reinterpret_tensor(arg32_1, (768, 2304), (1, 768), 0), out=buf43)
        del arg32_1
        buf44 = reinterpret_tensor(buf39, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [query_layer_7, scale_2, query_layer_8, attention_scores_2], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
        triton_poi_fused_add_clone_div_sqrt_1.run(buf43, arg33_1, buf44, 6291456, grid=grid(6291456), stream=stream0)
        del arg33_1
        buf45 = reinterpret_tensor(buf36, (16, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf43, buf45, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf46 = reinterpret_tensor(buf29, (192, 512, 512), (262144, 512, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf44, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf45, (192, 64, 512), (32768, 512, 1), 0), out=buf46)
        buf49 = reinterpret_tensor(buf26, (16, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__2, rmask_2, tensor_5, output_4, output_5], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf46, buf49, 98304, 512, grid=grid(98304), stream=stream0)
        buf50 = reinterpret_tensor(buf45, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [value_layer_5, context_layer_6], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_4.run(buf43, arg34_1, buf50, 6291456, grid=grid(6291456), stream=stream0)
        del arg34_1
        buf51 = reinterpret_tensor(buf44, (192, 512, 64), (32768, 64, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [context_layer_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (192, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf50, (192, 512, 64), (32768, 64, 1), 0), out=buf51)
        buf52 = reinterpret_tensor(buf50, (16, 512, 12, 64), (393216, 768, 64, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [context_layer_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf51, buf52, 6291456, grid=grid(6291456), stream=stream0)
        buf53 = reinterpret_tensor(buf51, (8192, 768), (768, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (8192, 768), (768, 1), 0), reinterpret_tensor(arg35_1, (768, 768), (1, 768), 0), out=buf53)
        del arg35_1
        buf56 = reinterpret_tensor(buf52, (16, 512, 768), (393216, 768, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [add_20, mean_15, sub_11, sub_10, pow_6, variance_5, add_21, sqrt_8, hidden_states_25, mul_10, y_5], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf53, arg36_1, buf42, arg37_1, arg38_1, buf56, 8192, 768, grid=grid(8192), stream=stream0)
        del arg36_1
        del arg37_1
        del arg38_1
        buf57 = reinterpret_tensor(buf38, (8192, 3072), (3072, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (8192, 768), (768, 1), 0), reinterpret_tensor(arg39_1, (768, 3072), (1, 768), 0), out=buf57)
        del arg39_1
        buf58 = reinterpret_tensor(buf57, (16, 512, 3072), (1572864, 3072, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf58, arg40_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg40_1
        buf59 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg41_1, (3072, 768), (1, 3072), 0), out=buf59)
        del arg41_1
        buf62 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [add_23, mean_18, sub_13, sub_12, pow_7, variance_6, add_24, sqrt_9, hidden_states_31, mul_11, y_6], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf59, arg42_1, buf56, arg43_1, arg44_1, buf62, 8192, 768, grid=grid(8192), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        buf63 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [qp_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (8192, 768), (768, 1), 0), reinterpret_tensor(arg45_1, (768, 2304), (1, 768), 0), out=buf63)
        del arg45_1
        buf64 = reinterpret_tensor(buf59, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [query_layer_10, scale_3, query_layer_11, attention_scores_3], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
        triton_poi_fused_add_clone_div_sqrt_1.run(buf63, arg46_1, buf64, 6291456, grid=grid(6291456), stream=stream0)
        del arg46_1
        buf65 = reinterpret_tensor(buf56, (16, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf63, buf65, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf66 = reinterpret_tensor(buf49, (192, 512, 512), (262144, 512, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf64, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf65, (192, 64, 512), (32768, 512, 1), 0), out=buf66)
        buf69 = reinterpret_tensor(buf46, (16, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__3, rmask_3, tensor_7, output_6, output_7], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf66, buf69, 98304, 512, grid=grid(98304), stream=stream0)
        buf70 = reinterpret_tensor(buf65, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [value_layer_7, context_layer_9], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_4.run(buf63, arg47_1, buf70, 6291456, grid=grid(6291456), stream=stream0)
        del arg47_1
        buf71 = reinterpret_tensor(buf64, (192, 512, 64), (32768, 64, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [context_layer_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf69, (192, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf70, (192, 512, 64), (32768, 64, 1), 0), out=buf71)
        buf72 = reinterpret_tensor(buf70, (16, 512, 12, 64), (393216, 768, 64, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [context_layer_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf71, buf72, 6291456, grid=grid(6291456), stream=stream0)
        buf73 = reinterpret_tensor(buf71, (8192, 768), (768, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (8192, 768), (768, 1), 0), reinterpret_tensor(arg48_1, (768, 768), (1, 768), 0), out=buf73)
        del arg48_1
        buf76 = reinterpret_tensor(buf72, (16, 512, 768), (393216, 768, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [add_28, mean_21, sub_15, sub_14, pow_8, variance_7, add_29, sqrt_11, hidden_states_35, mul_13, y_7], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf73, arg49_1, buf62, arg50_1, arg51_1, buf76, 8192, 768, grid=grid(8192), stream=stream0)
        del arg49_1
        del arg50_1
        del arg51_1
        buf77 = reinterpret_tensor(buf58, (8192, 3072), (3072, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (8192, 768), (768, 1), 0), reinterpret_tensor(arg52_1, (768, 3072), (1, 768), 0), out=buf77)
        del arg52_1
        buf78 = reinterpret_tensor(buf77, (16, 512, 3072), (1572864, 3072, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_38], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf78, arg53_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg53_1
        buf79 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg54_1, (3072, 768), (1, 3072), 0), out=buf79)
        del arg54_1
        buf82 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [add_31, mean_24, sub_17, sub_16, pow_9, variance_8, add_32, sqrt_12, hidden_states_41, mul_14, y_8], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf79, arg55_1, buf76, arg56_1, arg57_1, buf82, 8192, 768, grid=grid(8192), stream=stream0)
        del arg55_1
        del arg56_1
        del arg57_1
        buf83 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [qp_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (8192, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 2304), (1, 768), 0), out=buf83)
        del arg58_1
        buf84 = reinterpret_tensor(buf79, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [query_layer_13, scale_4, query_layer_14, attention_scores_4], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
        triton_poi_fused_add_clone_div_sqrt_1.run(buf83, arg59_1, buf84, 6291456, grid=grid(6291456), stream=stream0)
        del arg59_1
        buf85 = reinterpret_tensor(buf76, (16, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf83, buf85, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf86 = reinterpret_tensor(buf69, (192, 512, 512), (262144, 512, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf85, (192, 64, 512), (32768, 512, 1), 0), out=buf86)
        buf89 = reinterpret_tensor(buf66, (16, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__4, rmask_4, tensor_9, output_8, output_9], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf86, buf89, 98304, 512, grid=grid(98304), stream=stream0)
        buf90 = reinterpret_tensor(buf85, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [value_layer_9, context_layer_12], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_4.run(buf83, arg60_1, buf90, 6291456, grid=grid(6291456), stream=stream0)
        del arg60_1
        buf91 = reinterpret_tensor(buf84, (192, 512, 64), (32768, 64, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [context_layer_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf89, (192, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf90, (192, 512, 64), (32768, 64, 1), 0), out=buf91)
        buf92 = reinterpret_tensor(buf90, (16, 512, 12, 64), (393216, 768, 64, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [context_layer_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf91, buf92, 6291456, grid=grid(6291456), stream=stream0)
        buf93 = reinterpret_tensor(buf91, (8192, 768), (768, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (8192, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 768), (1, 768), 0), out=buf93)
        del arg61_1
        buf96 = reinterpret_tensor(buf92, (16, 512, 768), (393216, 768, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [add_36, mean_27, sub_19, sub_18, pow_10, variance_9, add_37, sqrt_14, hidden_states_45, mul_16, y_9], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf93, arg62_1, buf82, arg63_1, arg64_1, buf96, 8192, 768, grid=grid(8192), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        buf97 = reinterpret_tensor(buf78, (8192, 3072), (3072, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (8192, 768), (768, 1), 0), reinterpret_tensor(arg65_1, (768, 3072), (1, 768), 0), out=buf97)
        del arg65_1
        buf98 = reinterpret_tensor(buf97, (16, 512, 3072), (1572864, 3072, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_48], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf98, arg66_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg66_1
        buf99 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg67_1, (3072, 768), (1, 3072), 0), out=buf99)
        del arg67_1
        buf102 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [add_39, mean_30, sub_21, sub_20, pow_11, variance_10, add_40, sqrt_15, hidden_states_51, mul_17, y_10], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf99, arg68_1, buf96, arg69_1, arg70_1, buf102, 8192, 768, grid=grid(8192), stream=stream0)
        del arg68_1
        del arg69_1
        del arg70_1
        buf103 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [qp_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (8192, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 2304), (1, 768), 0), out=buf103)
        del arg71_1
        buf104 = reinterpret_tensor(buf99, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [query_layer_16, scale_5, query_layer_17, attention_scores_5], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
        triton_poi_fused_add_clone_div_sqrt_1.run(buf103, arg72_1, buf104, 6291456, grid=grid(6291456), stream=stream0)
        del arg72_1
        buf105 = reinterpret_tensor(buf96, (16, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf103, buf105, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf106 = reinterpret_tensor(buf89, (192, 512, 512), (262144, 512, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf104, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf105, (192, 64, 512), (32768, 512, 1), 0), out=buf106)
        buf109 = reinterpret_tensor(buf86, (16, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__5, rmask_5, tensor_11, output_10, output_11], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf106, buf109, 98304, 512, grid=grid(98304), stream=stream0)
        buf110 = reinterpret_tensor(buf105, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [value_layer_11, context_layer_15], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_4.run(buf103, arg73_1, buf110, 6291456, grid=grid(6291456), stream=stream0)
        del arg73_1
        buf111 = reinterpret_tensor(buf104, (192, 512, 64), (32768, 64, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [context_layer_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf109, (192, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf110, (192, 512, 64), (32768, 64, 1), 0), out=buf111)
        buf112 = reinterpret_tensor(buf110, (16, 512, 12, 64), (393216, 768, 64, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [context_layer_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf111, buf112, 6291456, grid=grid(6291456), stream=stream0)
        buf113 = reinterpret_tensor(buf111, (8192, 768), (768, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf112, (8192, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), out=buf113)
        del arg74_1
        buf116 = reinterpret_tensor(buf112, (16, 512, 768), (393216, 768, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [add_44, mean_33, sub_23, sub_22, pow_12, variance_11, add_45, sqrt_17, hidden_states_55, mul_19, y_11], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf113, arg75_1, buf102, arg76_1, arg77_1, buf116, 8192, 768, grid=grid(8192), stream=stream0)
        del arg75_1
        del arg76_1
        del arg77_1
        buf117 = reinterpret_tensor(buf98, (8192, 3072), (3072, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (8192, 768), (768, 1), 0), reinterpret_tensor(arg78_1, (768, 3072), (1, 768), 0), out=buf117)
        del arg78_1
        buf118 = reinterpret_tensor(buf117, (16, 512, 3072), (1572864, 3072, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf118, arg79_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg79_1
        buf119 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg80_1, (3072, 768), (1, 3072), 0), out=buf119)
        del arg80_1
        buf122 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [add_47, mean_36, sub_25, sub_24, pow_13, variance_12, add_48, sqrt_18, hidden_states_61, mul_20, y_12], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf119, arg81_1, buf116, arg82_1, arg83_1, buf122, 8192, 768, grid=grid(8192), stream=stream0)
        del arg81_1
        del arg82_1
        del arg83_1
        buf123 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [qp_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (8192, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 2304), (1, 768), 0), out=buf123)
        del arg84_1
        buf124 = reinterpret_tensor(buf119, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [query_layer_19, scale_6, query_layer_20, attention_scores_6], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
        triton_poi_fused_add_clone_div_sqrt_1.run(buf123, arg85_1, buf124, 6291456, grid=grid(6291456), stream=stream0)
        del arg85_1
        buf125 = reinterpret_tensor(buf116, (16, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf123, buf125, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf126 = reinterpret_tensor(buf109, (192, 512, 512), (262144, 512, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf124, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf125, (192, 64, 512), (32768, 512, 1), 0), out=buf126)
        buf129 = reinterpret_tensor(buf106, (16, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__6, rmask_6, tensor_13, output_12, output_13], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf126, buf129, 98304, 512, grid=grid(98304), stream=stream0)
        buf130 = reinterpret_tensor(buf125, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [value_layer_13, context_layer_18], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_4.run(buf123, arg86_1, buf130, 6291456, grid=grid(6291456), stream=stream0)
        del arg86_1
        buf131 = reinterpret_tensor(buf124, (192, 512, 64), (32768, 64, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [context_layer_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf129, (192, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf130, (192, 512, 64), (32768, 64, 1), 0), out=buf131)
        buf132 = reinterpret_tensor(buf130, (16, 512, 12, 64), (393216, 768, 64, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [context_layer_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf131, buf132, 6291456, grid=grid(6291456), stream=stream0)
        buf133 = reinterpret_tensor(buf131, (8192, 768), (768, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (8192, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 768), (1, 768), 0), out=buf133)
        del arg87_1
        buf136 = reinterpret_tensor(buf132, (16, 512, 768), (393216, 768, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [add_52, mean_39, sub_27, sub_26, pow_14, variance_13, add_53, sqrt_20, hidden_states_65, mul_22, y_13], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf133, arg88_1, buf122, arg89_1, arg90_1, buf136, 8192, 768, grid=grid(8192), stream=stream0)
        del arg88_1
        del arg89_1
        del arg90_1
        buf137 = reinterpret_tensor(buf118, (8192, 3072), (3072, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (8192, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 3072), (1, 768), 0), out=buf137)
        del arg91_1
        buf138 = reinterpret_tensor(buf137, (16, 512, 3072), (1572864, 3072, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_68], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf138, arg92_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg92_1
        buf139 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg93_1, (3072, 768), (1, 3072), 0), out=buf139)
        del arg93_1
        buf142 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [add_55, mean_42, sub_29, sub_28, pow_15, variance_14, add_56, sqrt_21, hidden_states_71, mul_23, y_14], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf139, arg94_1, buf136, arg95_1, arg96_1, buf142, 8192, 768, grid=grid(8192), stream=stream0)
        del arg94_1
        del arg95_1
        del arg96_1
        buf143 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [qp_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf142, (8192, 768), (768, 1), 0), reinterpret_tensor(arg97_1, (768, 2304), (1, 768), 0), out=buf143)
        del arg97_1
        buf144 = reinterpret_tensor(buf139, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [query_layer_22, scale_7, query_layer_23, attention_scores_7], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
        triton_poi_fused_add_clone_div_sqrt_1.run(buf143, arg98_1, buf144, 6291456, grid=grid(6291456), stream=stream0)
        del arg98_1
        buf145 = reinterpret_tensor(buf136, (16, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf143, buf145, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf146 = reinterpret_tensor(buf129, (192, 512, 512), (262144, 512, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf144, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf145, (192, 64, 512), (32768, 512, 1), 0), out=buf146)
        buf149 = reinterpret_tensor(buf126, (16, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__7, rmask_7, tensor_15, output_14, output_15], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf146, buf149, 98304, 512, grid=grid(98304), stream=stream0)
        buf150 = reinterpret_tensor(buf145, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [value_layer_15, context_layer_21], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_4.run(buf143, arg99_1, buf150, 6291456, grid=grid(6291456), stream=stream0)
        del arg99_1
        buf151 = reinterpret_tensor(buf144, (192, 512, 64), (32768, 64, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [context_layer_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf149, (192, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf150, (192, 512, 64), (32768, 64, 1), 0), out=buf151)
        buf152 = reinterpret_tensor(buf150, (16, 512, 12, 64), (393216, 768, 64, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [context_layer_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf151, buf152, 6291456, grid=grid(6291456), stream=stream0)
        buf153 = reinterpret_tensor(buf151, (8192, 768), (768, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf152, (8192, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 768), (1, 768), 0), out=buf153)
        del arg100_1
        buf156 = reinterpret_tensor(buf152, (16, 512, 768), (393216, 768, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [add_60, mean_45, sub_31, sub_30, pow_16, variance_15, add_61, sqrt_23, hidden_states_75, mul_25, y_15], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf153, arg101_1, buf142, arg102_1, arg103_1, buf156, 8192, 768, grid=grid(8192), stream=stream0)
        del arg101_1
        del arg102_1
        del arg103_1
        buf157 = reinterpret_tensor(buf138, (8192, 3072), (3072, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (8192, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 3072), (1, 768), 0), out=buf157)
        del arg104_1
        buf158 = reinterpret_tensor(buf157, (16, 512, 3072), (1572864, 3072, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf158, arg105_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg105_1
        buf159 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg106_1, (3072, 768), (1, 3072), 0), out=buf159)
        del arg106_1
        buf162 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [add_63, mean_48, sub_33, sub_32, pow_17, variance_16, add_64, sqrt_24, hidden_states_81, mul_26, y_16], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf159, arg107_1, buf156, arg108_1, arg109_1, buf162, 8192, 768, grid=grid(8192), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        buf163 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [qp_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (8192, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 2304), (1, 768), 0), out=buf163)
        del arg110_1
        buf164 = reinterpret_tensor(buf159, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [query_layer_25, scale_8, query_layer_26, attention_scores_8], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
        triton_poi_fused_add_clone_div_sqrt_1.run(buf163, arg111_1, buf164, 6291456, grid=grid(6291456), stream=stream0)
        del arg111_1
        buf165 = reinterpret_tensor(buf156, (16, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf163, buf165, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf166 = reinterpret_tensor(buf149, (192, 512, 512), (262144, 512, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf164, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf165, (192, 64, 512), (32768, 512, 1), 0), out=buf166)
        buf169 = reinterpret_tensor(buf146, (16, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__8, rmask_8, tensor_17, output_16, output_17], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf166, buf169, 98304, 512, grid=grid(98304), stream=stream0)
        buf170 = reinterpret_tensor(buf165, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [value_layer_17, context_layer_24], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_4.run(buf163, arg112_1, buf170, 6291456, grid=grid(6291456), stream=stream0)
        del arg112_1
        buf171 = reinterpret_tensor(buf164, (192, 512, 64), (32768, 64, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [context_layer_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf169, (192, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf170, (192, 512, 64), (32768, 64, 1), 0), out=buf171)
        buf172 = reinterpret_tensor(buf170, (16, 512, 12, 64), (393216, 768, 64, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [context_layer_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf171, buf172, 6291456, grid=grid(6291456), stream=stream0)
        buf173 = reinterpret_tensor(buf171, (8192, 768), (768, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (8192, 768), (768, 1), 0), reinterpret_tensor(arg113_1, (768, 768), (1, 768), 0), out=buf173)
        del arg113_1
        buf176 = reinterpret_tensor(buf172, (16, 512, 768), (393216, 768, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [add_68, mean_51, sub_35, sub_34, pow_18, variance_17, add_69, sqrt_26, hidden_states_85, mul_28, y_17], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf173, arg114_1, buf162, arg115_1, arg116_1, buf176, 8192, 768, grid=grid(8192), stream=stream0)
        del arg114_1
        del arg115_1
        del arg116_1
        buf177 = reinterpret_tensor(buf158, (8192, 3072), (3072, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (8192, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 3072), (1, 768), 0), out=buf177)
        del arg117_1
        buf178 = reinterpret_tensor(buf177, (16, 512, 3072), (1572864, 3072, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf178, arg118_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg118_1
        buf179 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg119_1, (3072, 768), (1, 3072), 0), out=buf179)
        del arg119_1
        buf182 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [add_71, mean_54, sub_37, sub_36, pow_19, variance_18, add_72, sqrt_27, hidden_states_91, mul_29, y_18], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf179, arg120_1, buf176, arg121_1, arg122_1, buf182, 8192, 768, grid=grid(8192), stream=stream0)
        del arg120_1
        del arg121_1
        del arg122_1
        buf183 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [qp_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (8192, 768), (768, 1), 0), reinterpret_tensor(arg123_1, (768, 2304), (1, 768), 0), out=buf183)
        del arg123_1
        buf184 = reinterpret_tensor(buf179, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [query_layer_28, scale_9, query_layer_29, attention_scores_9], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
        triton_poi_fused_add_clone_div_sqrt_1.run(buf183, arg124_1, buf184, 6291456, grid=grid(6291456), stream=stream0)
        del arg124_1
        buf185 = reinterpret_tensor(buf176, (16, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf183, buf185, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf186 = reinterpret_tensor(buf169, (192, 512, 512), (262144, 512, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf185, (192, 64, 512), (32768, 512, 1), 0), out=buf186)
        buf189 = reinterpret_tensor(buf166, (16, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__9, rmask_9, tensor_19, output_18, output_19], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf186, buf189, 98304, 512, grid=grid(98304), stream=stream0)
        buf190 = reinterpret_tensor(buf185, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [value_layer_19, context_layer_27], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_4.run(buf183, arg125_1, buf190, 6291456, grid=grid(6291456), stream=stream0)
        del arg125_1
        buf191 = reinterpret_tensor(buf184, (192, 512, 64), (32768, 64, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [context_layer_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf189, (192, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf190, (192, 512, 64), (32768, 64, 1), 0), out=buf191)
        buf192 = reinterpret_tensor(buf190, (16, 512, 12, 64), (393216, 768, 64, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [context_layer_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf191, buf192, 6291456, grid=grid(6291456), stream=stream0)
        buf193 = reinterpret_tensor(buf191, (8192, 768), (768, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf192, (8192, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 768), (1, 768), 0), out=buf193)
        del arg126_1
        buf196 = reinterpret_tensor(buf192, (16, 512, 768), (393216, 768, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [add_76, mean_57, sub_39, sub_38, pow_20, variance_19, add_77, sqrt_29, hidden_states_95, mul_31, y_19], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf193, arg127_1, buf182, arg128_1, arg129_1, buf196, 8192, 768, grid=grid(8192), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        buf197 = reinterpret_tensor(buf178, (8192, 3072), (3072, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (8192, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 3072), (1, 768), 0), out=buf197)
        del arg130_1
        buf198 = reinterpret_tensor(buf197, (16, 512, 3072), (1572864, 3072, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_98], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf198, arg131_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg131_1
        buf199 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg132_1, (3072, 768), (1, 3072), 0), out=buf199)
        del arg132_1
        buf202 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [add_79, mean_60, sub_41, sub_40, pow_21, variance_20, add_80, sqrt_30, hidden_states_101, mul_32, y_20], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf199, arg133_1, buf196, arg134_1, arg135_1, buf202, 8192, 768, grid=grid(8192), stream=stream0)
        del arg133_1
        del arg134_1
        del arg135_1
        buf203 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [qp_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (8192, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 2304), (1, 768), 0), out=buf203)
        del arg136_1
        buf204 = reinterpret_tensor(buf199, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [query_layer_31, scale_10, query_layer_32, attention_scores_10], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
        triton_poi_fused_add_clone_div_sqrt_1.run(buf203, arg137_1, buf204, 6291456, grid=grid(6291456), stream=stream0)
        del arg137_1
        buf205 = reinterpret_tensor(buf196, (16, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf203, buf205, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf206 = reinterpret_tensor(buf189, (192, 512, 512), (262144, 512, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf204, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf205, (192, 64, 512), (32768, 512, 1), 0), out=buf206)
        buf209 = reinterpret_tensor(buf186, (16, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__10, rmask_10, tensor_21, output_20, output_21], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf206, buf209, 98304, 512, grid=grid(98304), stream=stream0)
        buf210 = reinterpret_tensor(buf205, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [value_layer_21, context_layer_30], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_4.run(buf203, arg138_1, buf210, 6291456, grid=grid(6291456), stream=stream0)
        del arg138_1
        buf211 = reinterpret_tensor(buf204, (192, 512, 64), (32768, 64, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [context_layer_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf209, (192, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf210, (192, 512, 64), (32768, 64, 1), 0), out=buf211)
        buf212 = reinterpret_tensor(buf210, (16, 512, 12, 64), (393216, 768, 64, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [context_layer_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf211, buf212, 6291456, grid=grid(6291456), stream=stream0)
        buf213 = reinterpret_tensor(buf211, (8192, 768), (768, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (8192, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), out=buf213)
        del arg139_1
        buf216 = reinterpret_tensor(buf212, (16, 512, 768), (393216, 768, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [add_84, mean_63, sub_43, sub_42, pow_22, variance_21, add_85, sqrt_32, hidden_states_105, mul_34, y_21], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf213, arg140_1, buf202, arg141_1, arg142_1, buf216, 8192, 768, grid=grid(8192), stream=stream0)
        del arg140_1
        del arg141_1
        del arg142_1
        buf217 = reinterpret_tensor(buf198, (8192, 3072), (3072, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf216, (8192, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 3072), (1, 768), 0), out=buf217)
        del arg143_1
        buf218 = reinterpret_tensor(buf217, (16, 512, 3072), (1572864, 3072, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_108], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf218, arg144_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg144_1
        buf219 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg145_1, (3072, 768), (1, 3072), 0), out=buf219)
        del arg145_1
        buf222 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [add_87, mean_66, sub_45, sub_44, pow_23, variance_22, add_88, sqrt_33, hidden_states_111, mul_35, y_22], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf219, arg146_1, buf216, arg147_1, arg148_1, buf222, 8192, 768, grid=grid(8192), stream=stream0)
        del arg146_1
        del arg147_1
        del arg148_1
        buf223 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [qp_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf222, (8192, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 2304), (1, 768), 0), out=buf223)
        del arg149_1
        buf224 = reinterpret_tensor(buf219, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [query_layer_34, scale_11, query_layer_35, attention_scores_11], Original ATen: [aten.add, aten.sqrt, aten.div, aten.clone]
        triton_poi_fused_add_clone_div_sqrt_1.run(buf223, arg150_1, buf224, 6291456, grid=grid(6291456), stream=stream0)
        del arg150_1
        buf225 = reinterpret_tensor(buf216, (16, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf223, buf225, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf226 = reinterpret_tensor(buf209, (192, 512, 512), (262144, 512, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [attention_scores_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf224, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf225, (192, 64, 512), (32768, 512, 1), 0), out=buf226)
        buf229 = reinterpret_tensor(buf206, (16, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__11, rmask_11, tensor_23, output_22, output_23], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf226, buf229, 98304, 512, grid=grid(98304), stream=stream0)
        del buf226
        buf230 = reinterpret_tensor(buf225, (16, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [value_layer_23, context_layer_33], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_4.run(buf223, arg151_1, buf230, 6291456, grid=grid(6291456), stream=stream0)
        del arg151_1
        del buf223
        buf231 = reinterpret_tensor(buf224, (192, 512, 64), (32768, 64, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [context_layer_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf229, (192, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf230, (192, 512, 64), (32768, 64, 1), 0), out=buf231)
        del buf229
        buf232 = reinterpret_tensor(buf230, (16, 512, 12, 64), (393216, 768, 64, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [context_layer_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf231, buf232, 6291456, grid=grid(6291456), stream=stream0)
        buf233 = reinterpret_tensor(buf231, (8192, 768), (768, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (8192, 768), (768, 1), 0), reinterpret_tensor(arg152_1, (768, 768), (1, 768), 0), out=buf233)
        del arg152_1
        buf236 = reinterpret_tensor(buf232, (16, 512, 768), (393216, 768, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [add_92, mean_69, sub_47, sub_46, pow_24, variance_23, add_93, sqrt_35, hidden_states_115, mul_37, y_23], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf233, arg153_1, buf222, arg154_1, arg155_1, buf236, 8192, 768, grid=grid(8192), stream=stream0)
        del arg153_1
        del arg154_1
        del arg155_1
        buf237 = reinterpret_tensor(buf218, (8192, 3072), (3072, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf236, (8192, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 3072), (1, 768), 0), out=buf237)
        del arg156_1
        buf238 = reinterpret_tensor(buf237, (16, 512, 3072), (1572864, 3072, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_118], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf238, arg157_1, 25165824, grid=grid(25165824), stream=stream0)
        del arg157_1
        buf239 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (8192, 3072), (3072, 1), 0), reinterpret_tensor(arg158_1, (3072, 768), (1, 3072), 0), out=buf239)
        del arg158_1
        del buf238
        buf242 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [add_95, mean_72, sub_49, sub_48, pow_25, variance_24, add_96, sqrt_36, hidden_states_121, mul_38, y_24], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_6.run(buf239, arg159_1, buf236, arg160_1, arg161_1, buf242, 8192, 768, grid=grid(8192), stream=stream0)
        del arg159_1
        del arg160_1
        del arg161_1
        del buf236
        del buf239
        buf243 = empty_strided_cuda((8192, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (8192, 768), (768, 1), 0), reinterpret_tensor(arg162_1, (768, 2), (1, 768), 0), out=buf243)
        del arg162_1
        del buf242
        buf244 = empty_strided_cuda((16, 512), (512, 1), torch.float32)
        buf245 = empty_strided_cuda((16, 1), (1, 16), torch.float32)
        buf246 = empty_strided_cuda((16, 1), (1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [start_logits_1, start_loss], Original ATen: [aten.clone, aten._log_softmax]
        triton_red_fused__log_softmax_clone_8.run(buf243, arg163_1, buf244, buf245, buf246, 16, 512, grid=grid(16), stream=stream0)
        buf249 = empty_strided_cuda((16, 512), (512, 1), torch.float32)
        buf250 = empty_strided_cuda((16, 1), (1, 16), torch.float32)
        buf251 = empty_strided_cuda((16, 1), (1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [end_logits_1, end_loss], Original ATen: [aten.clone, aten._log_softmax]
        triton_red_fused__log_softmax_clone_9.run(buf243, arg163_1, buf249, buf250, buf251, 16, 512, grid=grid(16), stream=stream0)
        del arg163_1
        del buf243
        buf247 = empty_strided_cuda((), (), torch.float32)
        buf254 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [start_positions, start_loss, end_positions, end_loss, add_98, total_loss], Original ATen: [aten.clamp, aten.nll_loss_forward, aten.add, aten.div]
        triton_per_fused_add_clamp_div_nll_loss_forward_10.run(buf254, arg164_1, buf244, buf245, buf246, arg165_1, buf249, buf250, buf251, 1, 16, grid=grid(1), stream=stream0)
        del arg164_1
        del arg165_1
        del buf245
        del buf246
        del buf250
        del buf251
    return (buf254, buf244, buf249, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg58_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg71_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg165_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DebertaForQuestionAnswering', benchmark_compiled_module)
