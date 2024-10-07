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


# kernel path: /tmp/torchinductor_sahanp/ai/caihga2yyajc427zmmfrvrw4bug66jcvzm33xh57lcswot3m2dhv.py
# Topologically Sorted Source Nodes: [embedding, inputs_embeds, positions, embed_pos, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embed_pos => embedding_1
#   embedding => embedding
#   hidden_states => add
#   hidden_states_2 => add_1, add_2, mul_1, mul_2, rsqrt, sub, var_mean
#   inputs_embeds => mul
#   positions => iota
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %view, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, 1.0), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %iota), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg4_1), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg5_1), kwargs = {})
triton_red_fused_add_arange_embedding_mul_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_mul_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x0 = xindex % 128
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr2 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp23 = tl.load(in_ptr2 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/fa/cfagjxhnczlbhklarvynmwq4suib5wy3trs56kslkzbbjofej7qn.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %_scaled_dot_product_efficient_attention_default_23 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%unsqueeze_default_69, %unsqueeze_default_70, %unsqueeze_default_71, None, False), kwargs = {scale: 1.0})
triton_poi_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*(x2 % 16)) + (1024*x1) + (131072*(x2 // 16))), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*(x2 % 16))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d6/cd65rb3yw47c6fudk6qumlfoe3fp5guynd2gnjun745smbewac45.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %_scaled_dot_product_efficient_attention_default_23 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%unsqueeze_default_69, %unsqueeze_default_70, %unsqueeze_default_71, None, False), kwargs = {scale: 1.0})
triton_poi_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*(x2 % 16)) + (1024*x1) + (131072*(x2 // 16))), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*(x2 % 16))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wy/cwyef4bg4qq2ahirjo6trgz4ftlraqjpheohrw4jb7t45oibfp56.py
# Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_output_3 => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 128
    x2 = (xindex // 131072)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x2) + (32768*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ax/cax3dhdqcd7qzebaw4nekjnoueltuorzksthkcnqb44yt3u5gq7i.py
# Topologically Sorted Source Nodes: [embedding, inputs_embeds, positions, embed_pos, hidden_states, hidden_states_4, hidden_states_5], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embed_pos => embedding_1
#   embedding => embedding
#   hidden_states => add
#   hidden_states_4 => add_3
#   hidden_states_5 => add_4, add_5, mul_4, mul_5, rsqrt_1, sub_2, var_mean_1
#   inputs_embeds => mul
#   positions => iota
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %view, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, 1.0), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %iota), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %embedding_1), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_16), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_3), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg14_1), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg15_1), kwargs = {})
triton_per_fused_add_arange_embedding_mul_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_arange_embedding_mul_native_layer_norm_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 4096
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
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (r2 + (1024*x0)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_out_ptr0 + (r2 + (1024*x3)), None)
    tmp12 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp1 = tl.full([RBLOCK], 50265, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 50265), "index out of bounds: 0 <= tmp4 < 50265")
    tmp6 = tl.load(in_ptr1 + (r2 + (1024*tmp4)), None)
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = tl.full([1], 1024, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp15 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = tmp14 - tmp22
    tmp29 = 1024.0
    tmp30 = tmp27 / tmp29
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tl.store(in_out_ptr0 + (r2 + (1024*x3)), tmp14, None)
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp38, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ia/cia5nubqbh2c5ercojtqqb7yj55iekxlhrrz6x5n7dskkrbtkz5u.py
# Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_6 => add_6, erf, mul_6, mul_7, mul_8
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_18, 0.5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_18, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_7,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %add_6), kwargs = {})
triton_poi_fused_gelu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
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


# kernel path: /tmp/torchinductor_sahanp/vt/cvt2kw65u7weeckp25idzxnuvjoja5taz4fmn3hvgpyzlheljlc7.py
# Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_10 => add_7
#   hidden_states_11 => add_8, add_9, mul_10, mul_9, rsqrt_2, sub_3, var_mean_2
# Graph fragment:
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_20), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_7, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_5), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %arg20_1), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %arg21_1), kwargs = {})
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 4096
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


# kernel path: /tmp/torchinductor_sahanp/rb/crbjbxcplu7fuz2tshmhcb3fccn23t677pgbrvax2masifjod62k.py
# Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_10 => add_7
#   hidden_states_13 => add_10
#   hidden_states_14 => add_11, add_12, mul_12, mul_13, rsqrt_3, sub_5, var_mean_3
# Graph fragment:
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_20), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_36), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_10, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_10, %getitem_7), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_3), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %arg30_1), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %arg31_1), kwargs = {})
triton_per_fused_add_native_layer_norm_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 4096
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
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, None)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4w/c4wrzmw4ie27wk7pv7icdjppgsbysczg6cedzfcblrdhru5cphdr.py
# Topologically Sorted Source Nodes: [contiguous_38], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_38 => clone_100
# Graph fragment:
#   %clone_100 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_137,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192) % 16
    x3 = (xindex // 131072)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1) + (131072*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ij/cijsieuiwvk2eiudbwboxtip3lcsnxkxsh4mlyofwwqkqxvdn3e3.py
# Topologically Sorted Source Nodes: [key_states_24], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   key_states_24 => clone_98
# Graph fragment:
#   %clone_98 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_134,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192) % 16
    x3 = (xindex // 131072)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1) + (131072*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rt/crtjlmnsojeghojqbds77b2zn5yxryzkojrfi5o7qlga5bh7bc4h.py
# Topologically Sorted Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_weights_27 => amax_12, div_12, exp_12, sub_38, sum_13
# Graph fragment:
#   %amax_12 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_256, [-1], True), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_256, %amax_12), kwargs = {})
#   %exp_12 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_38,), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_12, [-1], True), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_12, %sum_13), kwargs = {})
triton_per_fused__softmax_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), None)
    tmp1 = r2
    tmp2 = 1 + x0
    tmp3 = tmp1 < tmp2
    tmp4 = 0.0
    tmp5 = -3.4028234663852886e+38
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 + tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = triton_helpers.max2(tmp8, 1)[:, None]
    tmp11 = tmp7 - tmp10
    tmp12 = tl_math.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp16 = tmp12 / tmp15
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/k3/ck3dxmk7o2marrm6y6gtrjx5cms5t3kz3tpbvlogacppcefncjjb.py
# Topologically Sorted Source Nodes: [attn_output_63], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_output_63 => clone_102
# Graph fragment:
#   %clone_102 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_139,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 16
    x2 = (xindex // 1024) % 128
    x3 = (xindex // 131072)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (8192*x1) + (131072*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rf/crffxr4qb5ld2pruvgjayvmwhq7gl5nzmguw2zjlm4op2nhf4gxv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_372, %full_default_4], 1), kwargs = {})
triton_poi_fused_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/w7/cw76e566mz66ihbpnlbucqu4yhavdsjmxqwhzkjd5k44isdeon6n.py
# Topologically Sorted Source Nodes: [lm_logits, masked_lm_loss], Original ATen: [aten.add, aten._log_softmax]
# Source node to ATen node mapping:
#   lm_logits => add_223
#   masked_lm_loss => amax_36, exp_36, sub_98, sum_37
# Graph fragment:
#   %add_223 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_700, %arg513_1), kwargs = {})
#   %amax_36 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_701, [1], True), kwargs = {})
#   %sub_98 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_701, %amax_36), kwargs = {})
#   %exp_36 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_98,), kwargs = {})
#   %sum_37 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_36, [1], True), kwargs = {})
triton_red_fused__log_softmax_add_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_add_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
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


# kernel path: /tmp/torchinductor_sahanp/c7/cc7svvvg56ibpep2veepjk4dzjrrga6vs4sm2ypbd5gd5i5kwy4s.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   masked_lm_loss => convert_element_type, div_36, full_default_3, ne_1, ne_2, neg, sum_38, sum_39, where_2
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_702, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_3), kwargs = {})
#   %sum_39 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_2,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_702, -100), kwargs = {})
#   %sum_38 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_38, torch.float32), kwargs = {})
#   %div_36 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_39, %convert_element_type), kwargs = {})
triton_red_fused_nll_loss_forward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_14', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    assert_size_stride(arg0_1, (32, 128), (128, 1))
    assert_size_stride(arg1_1, (32, 128), (128, 1))
    assert_size_stride(arg2_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg3_1, (1024, 1024), (1024, 1))
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
    assert_size_stride(arg198_1, (1024, 1024), (1024, 1))
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
        buf3 = empty_strided_cuda((32, 128, 1024), (131072, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, positions, embed_pos, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_arange_embedding_mul_native_layer_norm_0.run(arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, buf3, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg6_1, (1024, 1024), (1, 1024), 0), out=buf4)
        del arg6_1
        buf5 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg8_1, (1024, 1024), (1, 1024), 0), out=buf5)
        del arg8_1
        buf6 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg10_1, (1024, 1024), (1, 1024), 0), out=buf6)
        del arg10_1
        buf7 = reinterpret_tensor(buf3, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf4, arg7_1, buf7, 4194304, grid=grid(4194304), stream=stream0)
        del arg7_1
        buf8 = reinterpret_tensor(buf4, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf5, arg9_1, buf8, 4194304, grid=grid(4194304), stream=stream0)
        del arg9_1
        buf9 = reinterpret_tensor(buf5, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf6, arg11_1, buf9, 4194304, grid=grid(4194304), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf7, buf8, buf9, None, False, scale=1.0)
        buf11 = buf10[0]
        del buf10
        buf15 = reinterpret_tensor(buf9, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf11, buf15, 4194304, grid=grid(4194304), stream=stream0)
        buf16 = reinterpret_tensor(buf11, (4096, 1024), (1024, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg12_1, (1024, 1024), (1, 1024), 0), out=buf16)
        del arg12_1
        buf17 = reinterpret_tensor(buf16, (32, 128, 1024), (131072, 1024, 1), 0); del buf16  # reuse
        buf21 = reinterpret_tensor(buf15, (32, 128, 1024), (131072, 1024, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, positions, embed_pos, hidden_states, hidden_states_4, hidden_states_5], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
        triton_per_fused_add_arange_embedding_mul_native_layer_norm_4.run(buf17, arg1_1, arg2_1, arg3_1, arg13_1, arg14_1, arg15_1, buf21, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg13_1
        del arg14_1
        del arg15_1
        del arg3_1
        buf22 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg16_1, (1024, 4096), (1, 1024), 0), out=buf22)
        del arg16_1
        buf23 = reinterpret_tensor(buf22, (32, 128, 4096), (524288, 4096, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf23, arg17_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg17_1
        buf24 = reinterpret_tensor(buf21, (4096, 1024), (1024, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg18_1, (4096, 1024), (1, 4096), 0), out=buf24)
        del arg18_1
        buf28 = reinterpret_tensor(buf8, (32, 128, 1024), (131072, 1024, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf17, buf24, arg19_1, arg20_1, arg21_1, buf28, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg20_1
        del arg21_1
        buf29 = reinterpret_tensor(buf7, (4096, 1024), (1024, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg22_1, (1024, 1024), (1, 1024), 0), out=buf29)
        del arg22_1
        buf30 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg24_1, (1024, 1024), (1, 1024), 0), out=buf30)
        del arg24_1
        buf31 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg26_1, (1024, 1024), (1, 1024), 0), out=buf31)
        del arg26_1
        buf32 = reinterpret_tensor(buf28, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf29, arg23_1, buf32, 4194304, grid=grid(4194304), stream=stream0)
        del arg23_1
        buf33 = reinterpret_tensor(buf29, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf30, arg25_1, buf33, 4194304, grid=grid(4194304), stream=stream0)
        del arg25_1
        buf34 = reinterpret_tensor(buf30, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf31, arg27_1, buf34, 4194304, grid=grid(4194304), stream=stream0)
        del arg27_1
        del buf31
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf35 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf32, buf33, buf34, None, False, scale=1.0)
        buf36 = buf35[0]
        del buf35
        buf40 = reinterpret_tensor(buf34, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf36, buf40, 4194304, grid=grid(4194304), stream=stream0)
        buf41 = reinterpret_tensor(buf36, (4096, 1024), (1024, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg28_1, (1024, 1024), (1, 1024), 0), out=buf41)
        del arg28_1
        buf42 = reinterpret_tensor(buf41, (32, 128, 1024), (131072, 1024, 1), 0); del buf41  # reuse
        buf46 = reinterpret_tensor(buf40, (32, 128, 1024), (131072, 1024, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf42, buf17, buf24, arg19_1, arg29_1, arg30_1, arg31_1, buf46, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg19_1
        del arg29_1
        del arg30_1
        del arg31_1
        buf47 = reinterpret_tensor(buf23, (4096, 4096), (4096, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg32_1, (1024, 4096), (1, 1024), 0), out=buf47)
        del arg32_1
        buf48 = reinterpret_tensor(buf47, (32, 128, 4096), (524288, 4096, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf48, arg33_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg33_1
        buf49 = reinterpret_tensor(buf46, (4096, 1024), (1024, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf48, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg34_1, (4096, 1024), (1, 4096), 0), out=buf49)
        del arg34_1
        buf53 = reinterpret_tensor(buf24, (32, 128, 1024), (131072, 1024, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf42, buf49, arg35_1, arg36_1, arg37_1, buf53, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg36_1
        del arg37_1
        buf54 = reinterpret_tensor(buf17, (4096, 1024), (1024, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg38_1, (1024, 1024), (1, 1024), 0), out=buf54)
        del arg38_1
        buf55 = reinterpret_tensor(buf33, (4096, 1024), (1024, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg40_1, (1024, 1024), (1, 1024), 0), out=buf55)
        del arg40_1
        buf56 = reinterpret_tensor(buf32, (4096, 1024), (1024, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg42_1, (1024, 1024), (1, 1024), 0), out=buf56)
        del arg42_1
        buf57 = reinterpret_tensor(buf53, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf54, arg39_1, buf57, 4194304, grid=grid(4194304), stream=stream0)
        del arg39_1
        buf58 = reinterpret_tensor(buf54, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf55, arg41_1, buf58, 4194304, grid=grid(4194304), stream=stream0)
        del arg41_1
        buf59 = reinterpret_tensor(buf55, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf56, arg43_1, buf59, 4194304, grid=grid(4194304), stream=stream0)
        del arg43_1
        del buf56
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf60 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf57, buf58, buf59, None, False, scale=1.0)
        buf61 = buf60[0]
        del buf60
        buf65 = reinterpret_tensor(buf59, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf61, buf65, 4194304, grid=grid(4194304), stream=stream0)
        buf66 = reinterpret_tensor(buf61, (4096, 1024), (1024, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg44_1, (1024, 1024), (1, 1024), 0), out=buf66)
        del arg44_1
        buf67 = reinterpret_tensor(buf66, (32, 128, 1024), (131072, 1024, 1), 0); del buf66  # reuse
        buf71 = reinterpret_tensor(buf65, (32, 128, 1024), (131072, 1024, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_22, hidden_states_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf67, buf42, buf49, arg35_1, arg45_1, arg46_1, arg47_1, buf71, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg35_1
        del arg45_1
        del arg46_1
        del arg47_1
        buf72 = reinterpret_tensor(buf48, (4096, 4096), (4096, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg48_1, (1024, 4096), (1, 1024), 0), out=buf72)
        del arg48_1
        buf73 = reinterpret_tensor(buf72, (32, 128, 4096), (524288, 4096, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf73, arg49_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg49_1
        buf74 = reinterpret_tensor(buf71, (4096, 1024), (1024, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg50_1, (4096, 1024), (1, 4096), 0), out=buf74)
        del arg50_1
        buf78 = reinterpret_tensor(buf49, (32, 128, 1024), (131072, 1024, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf67, buf74, arg51_1, arg52_1, arg53_1, buf78, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg52_1
        del arg53_1
        buf79 = reinterpret_tensor(buf42, (4096, 1024), (1024, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg54_1, (1024, 1024), (1, 1024), 0), out=buf79)
        del arg54_1
        buf80 = reinterpret_tensor(buf58, (4096, 1024), (1024, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg56_1, (1024, 1024), (1, 1024), 0), out=buf80)
        del arg56_1
        buf81 = reinterpret_tensor(buf57, (4096, 1024), (1024, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg58_1, (1024, 1024), (1, 1024), 0), out=buf81)
        del arg58_1
        buf82 = reinterpret_tensor(buf78, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf79, arg55_1, buf82, 4194304, grid=grid(4194304), stream=stream0)
        del arg55_1
        buf83 = reinterpret_tensor(buf79, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf80, arg57_1, buf83, 4194304, grid=grid(4194304), stream=stream0)
        del arg57_1
        buf84 = reinterpret_tensor(buf80, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf81, arg59_1, buf84, 4194304, grid=grid(4194304), stream=stream0)
        del arg59_1
        del buf81
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf85 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf82, buf83, buf84, None, False, scale=1.0)
        buf86 = buf85[0]
        del buf85
        buf90 = reinterpret_tensor(buf84, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf86, buf90, 4194304, grid=grid(4194304), stream=stream0)
        buf91 = reinterpret_tensor(buf86, (4096, 1024), (1024, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg60_1, (1024, 1024), (1, 1024), 0), out=buf91)
        del arg60_1
        buf92 = reinterpret_tensor(buf91, (32, 128, 1024), (131072, 1024, 1), 0); del buf91  # reuse
        buf96 = reinterpret_tensor(buf90, (32, 128, 1024), (131072, 1024, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_31, hidden_states_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf92, buf67, buf74, arg51_1, arg61_1, arg62_1, arg63_1, buf96, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg51_1
        del arg61_1
        del arg62_1
        del arg63_1
        buf97 = reinterpret_tensor(buf73, (4096, 4096), (4096, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg64_1, (1024, 4096), (1, 1024), 0), out=buf97)
        del arg64_1
        buf98 = reinterpret_tensor(buf97, (32, 128, 4096), (524288, 4096, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_33], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf98, arg65_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg65_1
        buf99 = reinterpret_tensor(buf96, (4096, 1024), (1024, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg66_1, (4096, 1024), (1, 4096), 0), out=buf99)
        del arg66_1
        buf103 = reinterpret_tensor(buf74, (32, 128, 1024), (131072, 1024, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf92, buf99, arg67_1, arg68_1, arg69_1, buf103, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg68_1
        del arg69_1
        buf104 = reinterpret_tensor(buf67, (4096, 1024), (1024, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg70_1, (1024, 1024), (1, 1024), 0), out=buf104)
        del arg70_1
        buf105 = reinterpret_tensor(buf83, (4096, 1024), (1024, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg72_1, (1024, 1024), (1, 1024), 0), out=buf105)
        del arg72_1
        buf106 = reinterpret_tensor(buf82, (4096, 1024), (1024, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg74_1, (1024, 1024), (1, 1024), 0), out=buf106)
        del arg74_1
        buf107 = reinterpret_tensor(buf103, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf104, arg71_1, buf107, 4194304, grid=grid(4194304), stream=stream0)
        del arg71_1
        buf108 = reinterpret_tensor(buf104, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf105, arg73_1, buf108, 4194304, grid=grid(4194304), stream=stream0)
        del arg73_1
        buf109 = reinterpret_tensor(buf105, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf106, arg75_1, buf109, 4194304, grid=grid(4194304), stream=stream0)
        del arg75_1
        del buf106
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf110 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf107, buf108, buf109, None, False, scale=1.0)
        buf111 = buf110[0]
        del buf110
        buf115 = reinterpret_tensor(buf109, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf111, buf115, 4194304, grid=grid(4194304), stream=stream0)
        buf116 = reinterpret_tensor(buf111, (4096, 1024), (1024, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf115, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg76_1, (1024, 1024), (1, 1024), 0), out=buf116)
        del arg76_1
        buf117 = reinterpret_tensor(buf116, (32, 128, 1024), (131072, 1024, 1), 0); del buf116  # reuse
        buf121 = reinterpret_tensor(buf115, (32, 128, 1024), (131072, 1024, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_40, hidden_states_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf117, buf92, buf99, arg67_1, arg77_1, arg78_1, arg79_1, buf121, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg67_1
        del arg77_1
        del arg78_1
        del arg79_1
        buf122 = reinterpret_tensor(buf98, (4096, 4096), (4096, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg80_1, (1024, 4096), (1, 1024), 0), out=buf122)
        del arg80_1
        buf123 = reinterpret_tensor(buf122, (32, 128, 4096), (524288, 4096, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf123, arg81_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg81_1
        buf124 = reinterpret_tensor(buf121, (4096, 1024), (1024, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg82_1, (4096, 1024), (1, 4096), 0), out=buf124)
        del arg82_1
        buf128 = reinterpret_tensor(buf99, (32, 128, 1024), (131072, 1024, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf117, buf124, arg83_1, arg84_1, arg85_1, buf128, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg84_1
        del arg85_1
        buf129 = reinterpret_tensor(buf92, (4096, 1024), (1024, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg86_1, (1024, 1024), (1, 1024), 0), out=buf129)
        del arg86_1
        buf130 = reinterpret_tensor(buf108, (4096, 1024), (1024, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg88_1, (1024, 1024), (1, 1024), 0), out=buf130)
        del arg88_1
        buf131 = reinterpret_tensor(buf107, (4096, 1024), (1024, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg90_1, (1024, 1024), (1, 1024), 0), out=buf131)
        del arg90_1
        buf132 = reinterpret_tensor(buf128, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf129, arg87_1, buf132, 4194304, grid=grid(4194304), stream=stream0)
        del arg87_1
        buf133 = reinterpret_tensor(buf129, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf130, arg89_1, buf133, 4194304, grid=grid(4194304), stream=stream0)
        del arg89_1
        buf134 = reinterpret_tensor(buf130, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf131, arg91_1, buf134, 4194304, grid=grid(4194304), stream=stream0)
        del arg91_1
        del buf131
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf135 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf132, buf133, buf134, None, False, scale=1.0)
        buf136 = buf135[0]
        del buf135
        buf140 = reinterpret_tensor(buf134, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf136, buf140, 4194304, grid=grid(4194304), stream=stream0)
        buf141 = reinterpret_tensor(buf136, (4096, 1024), (1024, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg92_1, (1024, 1024), (1, 1024), 0), out=buf141)
        del arg92_1
        buf142 = reinterpret_tensor(buf141, (32, 128, 1024), (131072, 1024, 1), 0); del buf141  # reuse
        buf146 = reinterpret_tensor(buf140, (32, 128, 1024), (131072, 1024, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_49, hidden_states_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf142, buf117, buf124, arg83_1, arg93_1, arg94_1, arg95_1, buf146, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg83_1
        del arg93_1
        del arg94_1
        del arg95_1
        buf147 = reinterpret_tensor(buf123, (4096, 4096), (4096, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg96_1, (1024, 4096), (1, 1024), 0), out=buf147)
        del arg96_1
        buf148 = reinterpret_tensor(buf147, (32, 128, 4096), (524288, 4096, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf148, arg97_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg97_1
        buf149 = reinterpret_tensor(buf146, (4096, 1024), (1024, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg98_1, (4096, 1024), (1, 4096), 0), out=buf149)
        del arg98_1
        buf153 = reinterpret_tensor(buf124, (32, 128, 1024), (131072, 1024, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf142, buf149, arg99_1, arg100_1, arg101_1, buf153, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg100_1
        del arg101_1
        buf154 = reinterpret_tensor(buf117, (4096, 1024), (1024, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg102_1, (1024, 1024), (1, 1024), 0), out=buf154)
        del arg102_1
        buf155 = reinterpret_tensor(buf133, (4096, 1024), (1024, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg104_1, (1024, 1024), (1, 1024), 0), out=buf155)
        del arg104_1
        buf156 = reinterpret_tensor(buf132, (4096, 1024), (1024, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg106_1, (1024, 1024), (1, 1024), 0), out=buf156)
        del arg106_1
        buf157 = reinterpret_tensor(buf153, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf154, arg103_1, buf157, 4194304, grid=grid(4194304), stream=stream0)
        del arg103_1
        buf158 = reinterpret_tensor(buf154, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf155, arg105_1, buf158, 4194304, grid=grid(4194304), stream=stream0)
        del arg105_1
        buf159 = reinterpret_tensor(buf155, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf156, arg107_1, buf159, 4194304, grid=grid(4194304), stream=stream0)
        del arg107_1
        del buf156
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf160 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf157, buf158, buf159, None, False, scale=1.0)
        buf161 = buf160[0]
        del buf160
        buf165 = reinterpret_tensor(buf159, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf161, buf165, 4194304, grid=grid(4194304), stream=stream0)
        buf166 = reinterpret_tensor(buf161, (4096, 1024), (1024, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg108_1, (1024, 1024), (1, 1024), 0), out=buf166)
        del arg108_1
        buf167 = reinterpret_tensor(buf166, (32, 128, 1024), (131072, 1024, 1), 0); del buf166  # reuse
        buf171 = reinterpret_tensor(buf165, (32, 128, 1024), (131072, 1024, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_58, hidden_states_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf167, buf142, buf149, arg99_1, arg109_1, arg110_1, arg111_1, buf171, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg109_1
        del arg110_1
        del arg111_1
        del arg99_1
        buf172 = reinterpret_tensor(buf148, (4096, 4096), (4096, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg112_1, (1024, 4096), (1, 1024), 0), out=buf172)
        del arg112_1
        buf173 = reinterpret_tensor(buf172, (32, 128, 4096), (524288, 4096, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_60], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf173, arg113_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg113_1
        buf174 = reinterpret_tensor(buf171, (4096, 1024), (1024, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg114_1, (4096, 1024), (1, 4096), 0), out=buf174)
        del arg114_1
        buf178 = reinterpret_tensor(buf149, (32, 128, 1024), (131072, 1024, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf167, buf174, arg115_1, arg116_1, arg117_1, buf178, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg116_1
        del arg117_1
        buf179 = reinterpret_tensor(buf142, (4096, 1024), (1024, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg118_1, (1024, 1024), (1, 1024), 0), out=buf179)
        del arg118_1
        buf180 = reinterpret_tensor(buf158, (4096, 1024), (1024, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg120_1, (1024, 1024), (1, 1024), 0), out=buf180)
        del arg120_1
        buf181 = reinterpret_tensor(buf157, (4096, 1024), (1024, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg122_1, (1024, 1024), (1, 1024), 0), out=buf181)
        del arg122_1
        buf182 = reinterpret_tensor(buf178, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf179, arg119_1, buf182, 4194304, grid=grid(4194304), stream=stream0)
        del arg119_1
        buf183 = reinterpret_tensor(buf179, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf180, arg121_1, buf183, 4194304, grid=grid(4194304), stream=stream0)
        del arg121_1
        buf184 = reinterpret_tensor(buf180, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf181, arg123_1, buf184, 4194304, grid=grid(4194304), stream=stream0)
        del arg123_1
        del buf181
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf185 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf182, buf183, buf184, None, False, scale=1.0)
        buf186 = buf185[0]
        del buf185
        buf190 = reinterpret_tensor(buf184, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf186, buf190, 4194304, grid=grid(4194304), stream=stream0)
        buf191 = reinterpret_tensor(buf186, (4096, 1024), (1024, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg124_1, (1024, 1024), (1, 1024), 0), out=buf191)
        del arg124_1
        buf192 = reinterpret_tensor(buf191, (32, 128, 1024), (131072, 1024, 1), 0); del buf191  # reuse
        buf196 = reinterpret_tensor(buf190, (32, 128, 1024), (131072, 1024, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_67, hidden_states_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf192, buf167, buf174, arg115_1, arg125_1, arg126_1, arg127_1, buf196, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg115_1
        del arg125_1
        del arg126_1
        del arg127_1
        buf197 = reinterpret_tensor(buf173, (4096, 4096), (4096, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg128_1, (1024, 4096), (1, 1024), 0), out=buf197)
        del arg128_1
        buf198 = reinterpret_tensor(buf197, (32, 128, 4096), (524288, 4096, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf198, arg129_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg129_1
        buf199 = reinterpret_tensor(buf196, (4096, 1024), (1024, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg130_1, (4096, 1024), (1, 4096), 0), out=buf199)
        del arg130_1
        buf203 = reinterpret_tensor(buf174, (32, 128, 1024), (131072, 1024, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf192, buf199, arg131_1, arg132_1, arg133_1, buf203, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg132_1
        del arg133_1
        buf204 = reinterpret_tensor(buf167, (4096, 1024), (1024, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg134_1, (1024, 1024), (1, 1024), 0), out=buf204)
        del arg134_1
        buf205 = reinterpret_tensor(buf183, (4096, 1024), (1024, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg136_1, (1024, 1024), (1, 1024), 0), out=buf205)
        del arg136_1
        buf206 = reinterpret_tensor(buf182, (4096, 1024), (1024, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg138_1, (1024, 1024), (1, 1024), 0), out=buf206)
        del arg138_1
        buf207 = reinterpret_tensor(buf203, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf204, arg135_1, buf207, 4194304, grid=grid(4194304), stream=stream0)
        del arg135_1
        buf208 = reinterpret_tensor(buf204, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf205, arg137_1, buf208, 4194304, grid=grid(4194304), stream=stream0)
        del arg137_1
        buf209 = reinterpret_tensor(buf205, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf206, arg139_1, buf209, 4194304, grid=grid(4194304), stream=stream0)
        del arg139_1
        del buf206
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf210 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf207, buf208, buf209, None, False, scale=1.0)
        buf211 = buf210[0]
        del buf210
        buf215 = reinterpret_tensor(buf209, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf211, buf215, 4194304, grid=grid(4194304), stream=stream0)
        buf216 = reinterpret_tensor(buf211, (4096, 1024), (1024, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg140_1, (1024, 1024), (1, 1024), 0), out=buf216)
        del arg140_1
        buf217 = reinterpret_tensor(buf216, (32, 128, 1024), (131072, 1024, 1), 0); del buf216  # reuse
        buf221 = reinterpret_tensor(buf215, (32, 128, 1024), (131072, 1024, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_76, hidden_states_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf217, buf192, buf199, arg131_1, arg141_1, arg142_1, arg143_1, buf221, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg131_1
        del arg141_1
        del arg142_1
        del arg143_1
        buf222 = reinterpret_tensor(buf198, (4096, 4096), (4096, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg144_1, (1024, 4096), (1, 1024), 0), out=buf222)
        del arg144_1
        buf223 = reinterpret_tensor(buf222, (32, 128, 4096), (524288, 4096, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf223, arg145_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg145_1
        buf224 = reinterpret_tensor(buf221, (4096, 1024), (1024, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg146_1, (4096, 1024), (1, 4096), 0), out=buf224)
        del arg146_1
        buf228 = reinterpret_tensor(buf199, (32, 128, 1024), (131072, 1024, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82, hidden_states_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf217, buf224, arg147_1, arg148_1, arg149_1, buf228, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg148_1
        del arg149_1
        buf229 = reinterpret_tensor(buf192, (4096, 1024), (1024, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg150_1, (1024, 1024), (1, 1024), 0), out=buf229)
        del arg150_1
        buf230 = reinterpret_tensor(buf208, (4096, 1024), (1024, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg152_1, (1024, 1024), (1, 1024), 0), out=buf230)
        del arg152_1
        buf231 = reinterpret_tensor(buf207, (4096, 1024), (1024, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg154_1, (1024, 1024), (1, 1024), 0), out=buf231)
        del arg154_1
        buf232 = reinterpret_tensor(buf228, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf229, arg151_1, buf232, 4194304, grid=grid(4194304), stream=stream0)
        del arg151_1
        buf233 = reinterpret_tensor(buf229, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf230, arg153_1, buf233, 4194304, grid=grid(4194304), stream=stream0)
        del arg153_1
        buf234 = reinterpret_tensor(buf230, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf231, arg155_1, buf234, 4194304, grid=grid(4194304), stream=stream0)
        del arg155_1
        del buf231
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf235 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf232, buf233, buf234, None, False, scale=1.0)
        buf236 = buf235[0]
        del buf235
        buf240 = reinterpret_tensor(buf234, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf236, buf240, 4194304, grid=grid(4194304), stream=stream0)
        buf241 = reinterpret_tensor(buf236, (4096, 1024), (1024, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg156_1, (1024, 1024), (1, 1024), 0), out=buf241)
        del arg156_1
        buf242 = reinterpret_tensor(buf241, (32, 128, 1024), (131072, 1024, 1), 0); del buf241  # reuse
        buf246 = reinterpret_tensor(buf240, (32, 128, 1024), (131072, 1024, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82, hidden_states_85, hidden_states_86], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf242, buf217, buf224, arg147_1, arg157_1, arg158_1, arg159_1, buf246, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg147_1
        del arg157_1
        del arg158_1
        del arg159_1
        buf247 = reinterpret_tensor(buf223, (4096, 4096), (4096, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf246, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg160_1, (1024, 4096), (1, 1024), 0), out=buf247)
        del arg160_1
        buf248 = reinterpret_tensor(buf247, (32, 128, 4096), (524288, 4096, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf248, arg161_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg161_1
        buf249 = reinterpret_tensor(buf246, (4096, 1024), (1024, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg162_1, (4096, 1024), (1, 4096), 0), out=buf249)
        del arg162_1
        buf253 = reinterpret_tensor(buf224, (32, 128, 1024), (131072, 1024, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_92], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf242, buf249, arg163_1, arg164_1, arg165_1, buf253, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg164_1
        del arg165_1
        buf254 = reinterpret_tensor(buf217, (4096, 1024), (1024, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg166_1, (1024, 1024), (1, 1024), 0), out=buf254)
        del arg166_1
        buf255 = reinterpret_tensor(buf233, (4096, 1024), (1024, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg168_1, (1024, 1024), (1, 1024), 0), out=buf255)
        del arg168_1
        buf256 = reinterpret_tensor(buf232, (4096, 1024), (1024, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg170_1, (1024, 1024), (1, 1024), 0), out=buf256)
        del arg170_1
        buf257 = reinterpret_tensor(buf253, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf254, arg167_1, buf257, 4194304, grid=grid(4194304), stream=stream0)
        del arg167_1
        buf258 = reinterpret_tensor(buf254, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf255, arg169_1, buf258, 4194304, grid=grid(4194304), stream=stream0)
        del arg169_1
        buf259 = reinterpret_tensor(buf255, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf256, arg171_1, buf259, 4194304, grid=grid(4194304), stream=stream0)
        del arg171_1
        del buf256
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf260 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf257, buf258, buf259, None, False, scale=1.0)
        buf261 = buf260[0]
        del buf260
        buf265 = reinterpret_tensor(buf259, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf261, buf265, 4194304, grid=grid(4194304), stream=stream0)
        buf266 = reinterpret_tensor(buf261, (4096, 1024), (1024, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf265, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg172_1, (1024, 1024), (1, 1024), 0), out=buf266)
        del arg172_1
        buf267 = reinterpret_tensor(buf266, (32, 128, 1024), (131072, 1024, 1), 0); del buf266  # reuse
        buf271 = reinterpret_tensor(buf265, (32, 128, 1024), (131072, 1024, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_94, hidden_states_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf267, buf242, buf249, arg163_1, arg173_1, arg174_1, arg175_1, buf271, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg163_1
        del arg173_1
        del arg174_1
        del arg175_1
        buf272 = reinterpret_tensor(buf248, (4096, 4096), (4096, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg176_1, (1024, 4096), (1, 1024), 0), out=buf272)
        del arg176_1
        buf273 = reinterpret_tensor(buf272, (32, 128, 4096), (524288, 4096, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf273, arg177_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg177_1
        buf274 = reinterpret_tensor(buf271, (4096, 1024), (1024, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf273, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg178_1, (4096, 1024), (1, 4096), 0), out=buf274)
        del arg178_1
        buf278 = reinterpret_tensor(buf249, (32, 128, 1024), (131072, 1024, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf267, buf274, arg179_1, arg180_1, arg181_1, buf278, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg180_1
        del arg181_1
        buf279 = reinterpret_tensor(buf242, (4096, 1024), (1024, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg182_1, (1024, 1024), (1, 1024), 0), out=buf279)
        del arg182_1
        buf280 = reinterpret_tensor(buf258, (4096, 1024), (1024, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg184_1, (1024, 1024), (1, 1024), 0), out=buf280)
        del arg184_1
        buf281 = reinterpret_tensor(buf257, (4096, 1024), (1024, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg186_1, (1024, 1024), (1, 1024), 0), out=buf281)
        del arg186_1
        buf282 = reinterpret_tensor(buf278, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf279, arg183_1, buf282, 4194304, grid=grid(4194304), stream=stream0)
        del arg183_1
        buf283 = reinterpret_tensor(buf279, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf280, arg185_1, buf283, 4194304, grid=grid(4194304), stream=stream0)
        del arg185_1
        buf284 = reinterpret_tensor(buf280, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf281, arg187_1, buf284, 4194304, grid=grid(4194304), stream=stream0)
        del arg187_1
        del buf281
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf285 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf282, buf283, buf284, None, False, scale=1.0)
        buf286 = buf285[0]
        del buf285
        buf290 = reinterpret_tensor(buf284, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf286, buf290, 4194304, grid=grid(4194304), stream=stream0)
        buf291 = reinterpret_tensor(buf286, (4096, 1024), (1024, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf290, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg188_1, (1024, 1024), (1, 1024), 0), out=buf291)
        del arg188_1
        buf292 = reinterpret_tensor(buf291, (32, 128, 1024), (131072, 1024, 1), 0); del buf291  # reuse
        buf296 = reinterpret_tensor(buf290, (32, 128, 1024), (131072, 1024, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_103, hidden_states_104], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf292, buf267, buf274, arg179_1, arg189_1, arg190_1, arg191_1, buf296, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg179_1
        del arg189_1
        del arg190_1
        del arg191_1
        buf297 = reinterpret_tensor(buf273, (4096, 4096), (4096, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf296, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg192_1, (1024, 4096), (1, 1024), 0), out=buf297)
        del arg192_1
        buf298 = reinterpret_tensor(buf297, (32, 128, 4096), (524288, 4096, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_105], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf298, arg193_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg193_1
        buf299 = reinterpret_tensor(buf296, (4096, 1024), (1024, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg194_1, (4096, 1024), (1, 4096), 0), out=buf299)
        del arg194_1
        buf326 = reinterpret_tensor(buf274, (32, 128, 1024), (131072, 1024, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_109, hidden_states_110], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf292, buf299, arg195_1, arg196_1, arg197_1, buf326, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg195_1
        del arg196_1
        del arg197_1
        buf306 = reinterpret_tensor(buf299, (32, 128, 1024), (131072, 1024, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [embedding_2, inputs_embeds_1, positions_1, positions_2, hidden_states_111, hidden_states_113], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
        triton_red_fused_add_arange_embedding_mul_native_layer_norm_0.run(arg1_1, arg2_1, arg198_1, arg199_1, arg200_1, buf306, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg199_1
        del arg200_1
        buf307 = reinterpret_tensor(buf292, (4096, 1024), (1024, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf306, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg201_1, (1024, 1024), (1, 1024), 0), out=buf307)
        del arg201_1
        buf308 = reinterpret_tensor(buf267, (4096, 1024), (1024, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf306, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg203_1, (1024, 1024), (1, 1024), 0), out=buf308)
        del arg203_1
        buf309 = reinterpret_tensor(buf283, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf307, arg202_1, buf309, 4194304, grid=grid(4194304), stream=stream0)
        del arg202_1
        buf310 = reinterpret_tensor(buf307, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf308, arg204_1, buf310, 4194304, grid=grid(4194304), stream=stream0)
        del arg204_1
        buf311 = empty_strided_cuda((512, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf309, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf310, (512, 64, 128), (8192, 1, 64), 0), out=buf311)
        buf315 = empty_strided_cuda((512, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf311, buf315, 65536, 128, grid=grid(65536), stream=stream0)
        buf314 = reinterpret_tensor(buf310, (4096, 1024), (1024, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf306, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg205_1, (1024, 1024), (1, 1024), 0), out=buf314)
        del arg205_1
        buf316 = reinterpret_tensor(buf306, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf314, arg206_1, buf316, 4194304, grid=grid(4194304), stream=stream0)
        del arg206_1
        buf317 = reinterpret_tensor(buf314, (512, 128, 64), (8192, 64, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_27, attn_output_60], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf315, reinterpret_tensor(buf316, (512, 128, 64), (8192, 64, 1), 0), out=buf317)
        buf318 = reinterpret_tensor(buf316, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [attn_output_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf317, buf318, 4194304, grid=grid(4194304), stream=stream0)
        buf319 = reinterpret_tensor(buf317, (4096, 1024), (1024, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf318, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg207_1, (1024, 1024), (1, 1024), 0), out=buf319)
        del arg207_1
        buf320 = reinterpret_tensor(buf319, (32, 128, 1024), (131072, 1024, 1), 0); del buf319  # reuse
        buf324 = reinterpret_tensor(buf318, (32, 128, 1024), (131072, 1024, 1), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [embedding_2, inputs_embeds_1, positions_1, positions_2, hidden_states_111, hidden_states_115, hidden_states_116], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
        triton_per_fused_add_arange_embedding_mul_native_layer_norm_4.run(buf320, arg1_1, arg2_1, arg198_1, arg208_1, arg209_1, arg210_1, buf324, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg198_1
        del arg1_1
        del arg208_1
        del arg209_1
        del arg210_1
        buf325 = reinterpret_tensor(buf309, (4096, 1024), (1024, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg211_1, (1024, 1024), (1, 1024), 0), out=buf325)
        del arg211_1
        buf327 = reinterpret_tensor(buf324, (4096, 1024), (1024, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg213_1, (1024, 1024), (1, 1024), 0), out=buf327)
        del arg213_1
        buf328 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg215_1, (1024, 1024), (1, 1024), 0), out=buf328)
        del arg215_1
        buf329 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf325, arg212_1, buf329, 4194304, grid=grid(4194304), stream=stream0)
        del arg212_1
        buf330 = reinterpret_tensor(buf325, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf327, arg214_1, buf330, 4194304, grid=grid(4194304), stream=stream0)
        del arg214_1
        buf331 = reinterpret_tensor(buf327, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf328, arg216_1, buf331, 4194304, grid=grid(4194304), stream=stream0)
        del arg216_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf332 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf329, buf330, buf331, None, False, scale=1.0)
        buf333 = buf332[0]
        del buf332
        buf337 = reinterpret_tensor(buf331, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [attn_output_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf333, buf337, 4194304, grid=grid(4194304), stream=stream0)
        buf338 = reinterpret_tensor(buf333, (4096, 1024), (1024, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf337, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg217_1, (1024, 1024), (1, 1024), 0), out=buf338)
        del arg217_1
        buf342 = reinterpret_tensor(buf337, (32, 128, 1024), (131072, 1024, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_118, hidden_states_119], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf320, buf338, arg218_1, arg219_1, arg220_1, buf342, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg219_1
        del arg220_1
        buf343 = reinterpret_tensor(buf298, (4096, 4096), (4096, 1), 0); del buf298  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf342, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg221_1, (1024, 4096), (1, 1024), 0), out=buf343)
        del arg221_1
        buf344 = reinterpret_tensor(buf343, (32, 128, 4096), (524288, 4096, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_120], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf344, arg222_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg222_1
        buf345 = reinterpret_tensor(buf342, (4096, 1024), (1024, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf344, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg223_1, (4096, 1024), (1, 4096), 0), out=buf345)
        del arg223_1
        buf346 = reinterpret_tensor(buf345, (32, 128, 1024), (131072, 1024, 1), 0); del buf345  # reuse
        buf350 = reinterpret_tensor(buf330, (32, 128, 1024), (131072, 1024, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_118, hidden_states_124, hidden_states_125], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf346, buf320, buf338, arg218_1, arg224_1, arg225_1, arg226_1, buf350, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg218_1
        del arg224_1
        del arg225_1
        del arg226_1
        buf351 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf350, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg227_1, (1024, 1024), (1, 1024), 0), out=buf351)
        del arg227_1
        buf352 = reinterpret_tensor(buf320, (4096, 1024), (1024, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf350, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg229_1, (1024, 1024), (1, 1024), 0), out=buf352)
        del arg229_1
        buf353 = reinterpret_tensor(buf329, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf351, arg228_1, buf353, 4194304, grid=grid(4194304), stream=stream0)
        del arg228_1
        buf354 = reinterpret_tensor(buf351, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf352, arg230_1, buf354, 4194304, grid=grid(4194304), stream=stream0)
        del arg230_1
        buf355 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf353, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf354, (512, 64, 128), (8192, 1, 64), 0), out=buf355)
        buf359 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_33], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf355, buf359, 65536, 128, grid=grid(65536), stream=stream0)
        buf358 = reinterpret_tensor(buf354, (4096, 1024), (1024, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf350, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg231_1, (1024, 1024), (1, 1024), 0), out=buf358)
        del arg231_1
        buf360 = reinterpret_tensor(buf350, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf350  # reuse
        # Topologically Sorted Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf358, arg232_1, buf360, 4194304, grid=grid(4194304), stream=stream0)
        del arg232_1
        buf361 = reinterpret_tensor(buf358, (512, 128, 64), (8192, 64, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_33, attn_output_70], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf359, reinterpret_tensor(buf360, (512, 128, 64), (8192, 64, 1), 0), out=buf361)
        buf362 = reinterpret_tensor(buf360, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [attn_output_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf361, buf362, 4194304, grid=grid(4194304), stream=stream0)
        buf363 = reinterpret_tensor(buf361, (4096, 1024), (1024, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf362, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg233_1, (1024, 1024), (1, 1024), 0), out=buf363)
        del arg233_1
        buf367 = reinterpret_tensor(buf362, (32, 128, 1024), (131072, 1024, 1), 0); del buf362  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_127, hidden_states_128], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf346, buf363, arg234_1, arg235_1, arg236_1, buf367, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg235_1
        del arg236_1
        buf368 = reinterpret_tensor(buf353, (4096, 1024), (1024, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf367, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg237_1, (1024, 1024), (1, 1024), 0), out=buf368)
        del arg237_1
        buf369 = reinterpret_tensor(buf367, (4096, 1024), (1024, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg239_1, (1024, 1024), (1, 1024), 0), out=buf369)
        del arg239_1
        buf370 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg241_1, (1024, 1024), (1, 1024), 0), out=buf370)
        del arg241_1
        buf371 = reinterpret_tensor(buf328, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf368, arg238_1, buf371, 4194304, grid=grid(4194304), stream=stream0)
        del arg238_1
        buf372 = reinterpret_tensor(buf368, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf368  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf369, arg240_1, buf372, 4194304, grid=grid(4194304), stream=stream0)
        del arg240_1
        buf373 = reinterpret_tensor(buf369, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf369  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf370, arg242_1, buf373, 4194304, grid=grid(4194304), stream=stream0)
        del arg242_1
        del buf370
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf374 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf371, buf372, buf373, None, False, scale=1.0)
        buf375 = buf374[0]
        del buf374
        buf379 = reinterpret_tensor(buf373, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [attn_output_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf375, buf379, 4194304, grid=grid(4194304), stream=stream0)
        buf380 = reinterpret_tensor(buf375, (4096, 1024), (1024, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf379, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg243_1, (1024, 1024), (1, 1024), 0), out=buf380)
        del arg243_1
        buf381 = reinterpret_tensor(buf380, (32, 128, 1024), (131072, 1024, 1), 0); del buf380  # reuse
        buf385 = reinterpret_tensor(buf379, (32, 128, 1024), (131072, 1024, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_127, hidden_states_130, hidden_states_131], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf381, buf346, buf363, arg234_1, arg244_1, arg245_1, arg246_1, buf385, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg234_1
        del arg244_1
        del arg245_1
        del arg246_1
        buf386 = reinterpret_tensor(buf344, (4096, 4096), (4096, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf385, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg247_1, (1024, 4096), (1, 1024), 0), out=buf386)
        del arg247_1
        buf387 = reinterpret_tensor(buf386, (32, 128, 4096), (524288, 4096, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_132], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf387, arg248_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg248_1
        buf388 = reinterpret_tensor(buf385, (4096, 1024), (1024, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf387, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg249_1, (4096, 1024), (1, 4096), 0), out=buf388)
        del arg249_1
        buf392 = reinterpret_tensor(buf363, (32, 128, 1024), (131072, 1024, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_137], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf381, buf388, arg250_1, arg251_1, arg252_1, buf392, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg251_1
        del arg252_1
        buf393 = reinterpret_tensor(buf346, (4096, 1024), (1024, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg253_1, (1024, 1024), (1, 1024), 0), out=buf393)
        del arg253_1
        buf394 = reinterpret_tensor(buf372, (4096, 1024), (1024, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg255_1, (1024, 1024), (1, 1024), 0), out=buf394)
        del arg255_1
        buf395 = reinterpret_tensor(buf371, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf371  # reuse
        # Topologically Sorted Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf393, arg254_1, buf395, 4194304, grid=grid(4194304), stream=stream0)
        del arg254_1
        buf396 = reinterpret_tensor(buf393, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf394, arg256_1, buf396, 4194304, grid=grid(4194304), stream=stream0)
        del arg256_1
        del buf394
        buf397 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf395, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf396, (512, 64, 128), (8192, 1, 64), 0), out=buf397)
        buf401 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf397, buf401, 65536, 128, grid=grid(65536), stream=stream0)
        buf400 = reinterpret_tensor(buf396, (4096, 1024), (1024, 1), 0); del buf396  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg257_1, (1024, 1024), (1, 1024), 0), out=buf400)
        del arg257_1
        buf402 = reinterpret_tensor(buf392, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf400, arg258_1, buf402, 4194304, grid=grid(4194304), stream=stream0)
        del arg258_1
        buf403 = reinterpret_tensor(buf400, (512, 128, 64), (8192, 64, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_39, attn_output_80], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf401, reinterpret_tensor(buf402, (512, 128, 64), (8192, 64, 1), 0), out=buf403)
        buf404 = reinterpret_tensor(buf402, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf402  # reuse
        # Topologically Sorted Source Nodes: [attn_output_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf403, buf404, 4194304, grid=grid(4194304), stream=stream0)
        buf405 = reinterpret_tensor(buf403, (4096, 1024), (1024, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg259_1, (1024, 1024), (1, 1024), 0), out=buf405)
        del arg259_1
        buf406 = reinterpret_tensor(buf405, (32, 128, 1024), (131072, 1024, 1), 0); del buf405  # reuse
        buf410 = reinterpret_tensor(buf404, (32, 128, 1024), (131072, 1024, 1), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_139, hidden_states_140], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf406, buf381, buf388, arg250_1, arg260_1, arg261_1, arg262_1, buf410, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg250_1
        del arg260_1
        del arg261_1
        del arg262_1
        buf411 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf410, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg263_1, (1024, 1024), (1, 1024), 0), out=buf411)
        del arg263_1
        buf412 = reinterpret_tensor(buf410, (4096, 1024), (1024, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg265_1, (1024, 1024), (1, 1024), 0), out=buf412)
        del arg265_1
        buf413 = reinterpret_tensor(buf381, (4096, 1024), (1024, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg267_1, (1024, 1024), (1, 1024), 0), out=buf413)
        del arg267_1
        buf414 = reinterpret_tensor(buf395, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf411, arg264_1, buf414, 4194304, grid=grid(4194304), stream=stream0)
        del arg264_1
        buf415 = reinterpret_tensor(buf411, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf412, arg266_1, buf415, 4194304, grid=grid(4194304), stream=stream0)
        del arg266_1
        buf416 = reinterpret_tensor(buf412, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf413, arg268_1, buf416, 4194304, grid=grid(4194304), stream=stream0)
        del arg268_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf417 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf414, buf415, buf416, None, False, scale=1.0)
        buf418 = buf417[0]
        del buf417
        buf422 = reinterpret_tensor(buf416, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [attn_output_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf418, buf422, 4194304, grid=grid(4194304), stream=stream0)
        buf423 = reinterpret_tensor(buf418, (4096, 1024), (1024, 1), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf422, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg269_1, (1024, 1024), (1, 1024), 0), out=buf423)
        del arg269_1
        buf427 = reinterpret_tensor(buf422, (32, 128, 1024), (131072, 1024, 1), 0); del buf422  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_142, hidden_states_143], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf406, buf423, arg270_1, arg271_1, arg272_1, buf427, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg271_1
        del arg272_1
        buf428 = reinterpret_tensor(buf387, (4096, 4096), (4096, 1), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf427, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg273_1, (1024, 4096), (1, 1024), 0), out=buf428)
        del arg273_1
        buf429 = reinterpret_tensor(buf428, (32, 128, 4096), (524288, 4096, 1), 0); del buf428  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_144], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf429, arg274_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg274_1
        buf430 = reinterpret_tensor(buf427, (4096, 1024), (1024, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf429, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg275_1, (4096, 1024), (1, 4096), 0), out=buf430)
        del arg275_1
        buf431 = reinterpret_tensor(buf430, (32, 128, 1024), (131072, 1024, 1), 0); del buf430  # reuse
        buf435 = reinterpret_tensor(buf415, (32, 128, 1024), (131072, 1024, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_142, hidden_states_148, hidden_states_149], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf431, buf406, buf423, arg270_1, arg276_1, arg277_1, arg278_1, buf435, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg270_1
        del arg276_1
        del arg277_1
        del arg278_1
        buf436 = buf423; del buf423  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg279_1, (1024, 1024), (1, 1024), 0), out=buf436)
        del arg279_1
        buf437 = reinterpret_tensor(buf406, (4096, 1024), (1024, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg281_1, (1024, 1024), (1, 1024), 0), out=buf437)
        del arg281_1
        buf438 = reinterpret_tensor(buf414, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [contiguous_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf436, arg280_1, buf438, 4194304, grid=grid(4194304), stream=stream0)
        del arg280_1
        buf439 = reinterpret_tensor(buf436, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [key_states_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf437, arg282_1, buf439, 4194304, grid=grid(4194304), stream=stream0)
        del arg282_1
        buf440 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf438, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf439, (512, 64, 128), (8192, 1, 64), 0), out=buf440)
        buf444 = buf397; del buf397  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_45], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf440, buf444, 65536, 128, grid=grid(65536), stream=stream0)
        buf443 = reinterpret_tensor(buf439, (4096, 1024), (1024, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg283_1, (1024, 1024), (1, 1024), 0), out=buf443)
        del arg283_1
        buf445 = reinterpret_tensor(buf435, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf435  # reuse
        # Topologically Sorted Source Nodes: [value_states_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf443, arg284_1, buf445, 4194304, grid=grid(4194304), stream=stream0)
        del arg284_1
        buf446 = reinterpret_tensor(buf443, (512, 128, 64), (8192, 64, 1), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_45, attn_output_90], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf444, reinterpret_tensor(buf445, (512, 128, 64), (8192, 64, 1), 0), out=buf446)
        buf447 = reinterpret_tensor(buf445, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf445  # reuse
        # Topologically Sorted Source Nodes: [attn_output_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf446, buf447, 4194304, grid=grid(4194304), stream=stream0)
        buf448 = reinterpret_tensor(buf446, (4096, 1024), (1024, 1), 0); del buf446  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf447, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg285_1, (1024, 1024), (1, 1024), 0), out=buf448)
        del arg285_1
        buf452 = reinterpret_tensor(buf447, (32, 128, 1024), (131072, 1024, 1), 0); del buf447  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_151, hidden_states_152], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf431, buf448, arg286_1, arg287_1, arg288_1, buf452, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg287_1
        del arg288_1
        buf453 = reinterpret_tensor(buf438, (4096, 1024), (1024, 1), 0); del buf438  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf452, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg289_1, (1024, 1024), (1, 1024), 0), out=buf453)
        del arg289_1
        buf454 = reinterpret_tensor(buf452, (4096, 1024), (1024, 1), 0); del buf452  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg291_1, (1024, 1024), (1, 1024), 0), out=buf454)
        del arg291_1
        buf455 = buf437; del buf437  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg293_1, (1024, 1024), (1, 1024), 0), out=buf455)
        del arg293_1
        buf456 = reinterpret_tensor(buf413, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf453, arg290_1, buf456, 4194304, grid=grid(4194304), stream=stream0)
        del arg290_1
        buf457 = reinterpret_tensor(buf453, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf454, arg292_1, buf457, 4194304, grid=grid(4194304), stream=stream0)
        del arg292_1
        buf458 = reinterpret_tensor(buf454, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf455, arg294_1, buf458, 4194304, grid=grid(4194304), stream=stream0)
        del arg294_1
        del buf455
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf459 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf456, buf457, buf458, None, False, scale=1.0)
        buf460 = buf459[0]
        del buf459
        buf464 = reinterpret_tensor(buf458, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf458  # reuse
        # Topologically Sorted Source Nodes: [attn_output_98], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf460, buf464, 4194304, grid=grid(4194304), stream=stream0)
        buf465 = reinterpret_tensor(buf460, (4096, 1024), (1024, 1), 0); del buf460  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf464, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg295_1, (1024, 1024), (1, 1024), 0), out=buf465)
        del arg295_1
        buf466 = reinterpret_tensor(buf465, (32, 128, 1024), (131072, 1024, 1), 0); del buf465  # reuse
        buf470 = reinterpret_tensor(buf464, (32, 128, 1024), (131072, 1024, 1), 0); del buf464  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_151, hidden_states_154, hidden_states_155], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf466, buf431, buf448, arg286_1, arg296_1, arg297_1, arg298_1, buf470, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg286_1
        del arg296_1
        del arg297_1
        del arg298_1
        buf471 = reinterpret_tensor(buf429, (4096, 4096), (4096, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf470, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg299_1, (1024, 4096), (1, 1024), 0), out=buf471)
        del arg299_1
        buf472 = reinterpret_tensor(buf471, (32, 128, 4096), (524288, 4096, 1), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_156], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf472, arg300_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg300_1
        buf473 = reinterpret_tensor(buf470, (4096, 1024), (1024, 1), 0); del buf470  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf472, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg301_1, (4096, 1024), (1, 4096), 0), out=buf473)
        del arg301_1
        buf477 = reinterpret_tensor(buf448, (32, 128, 1024), (131072, 1024, 1), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_160, hidden_states_161], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf466, buf473, arg302_1, arg303_1, arg304_1, buf477, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg303_1
        del arg304_1
        buf478 = reinterpret_tensor(buf431, (4096, 1024), (1024, 1), 0); del buf431  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf477, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg305_1, (1024, 1024), (1, 1024), 0), out=buf478)
        del arg305_1
        buf479 = reinterpret_tensor(buf457, (4096, 1024), (1024, 1), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf477, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg307_1, (1024, 1024), (1, 1024), 0), out=buf479)
        del arg307_1
        buf480 = reinterpret_tensor(buf456, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf456  # reuse
        # Topologically Sorted Source Nodes: [contiguous_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf478, arg306_1, buf480, 4194304, grid=grid(4194304), stream=stream0)
        del arg306_1
        buf481 = reinterpret_tensor(buf478, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [key_states_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf479, arg308_1, buf481, 4194304, grid=grid(4194304), stream=stream0)
        del arg308_1
        del buf479
        buf482 = buf444; del buf444  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf480, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf481, (512, 64, 128), (8192, 1, 64), 0), out=buf482)
        buf486 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_51], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf482, buf486, 65536, 128, grid=grid(65536), stream=stream0)
        buf485 = reinterpret_tensor(buf481, (4096, 1024), (1024, 1), 0); del buf481  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf477, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg309_1, (1024, 1024), (1, 1024), 0), out=buf485)
        del arg309_1
        buf487 = reinterpret_tensor(buf477, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf477  # reuse
        # Topologically Sorted Source Nodes: [value_states_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf485, arg310_1, buf487, 4194304, grid=grid(4194304), stream=stream0)
        del arg310_1
        buf488 = reinterpret_tensor(buf485, (512, 128, 64), (8192, 64, 1), 0); del buf485  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_51, attn_output_100], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf486, reinterpret_tensor(buf487, (512, 128, 64), (8192, 64, 1), 0), out=buf488)
        buf489 = reinterpret_tensor(buf487, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf487  # reuse
        # Topologically Sorted Source Nodes: [attn_output_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf488, buf489, 4194304, grid=grid(4194304), stream=stream0)
        buf490 = reinterpret_tensor(buf488, (4096, 1024), (1024, 1), 0); del buf488  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf489, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg311_1, (1024, 1024), (1, 1024), 0), out=buf490)
        del arg311_1
        buf491 = reinterpret_tensor(buf490, (32, 128, 1024), (131072, 1024, 1), 0); del buf490  # reuse
        buf495 = reinterpret_tensor(buf489, (32, 128, 1024), (131072, 1024, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_160, hidden_states_163, hidden_states_164], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf491, buf466, buf473, arg302_1, arg312_1, arg313_1, arg314_1, buf495, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg302_1
        del arg312_1
        del arg313_1
        del arg314_1
        buf496 = buf473; del buf473  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf495, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg315_1, (1024, 1024), (1, 1024), 0), out=buf496)
        del arg315_1
        buf497 = reinterpret_tensor(buf495, (4096, 1024), (1024, 1), 0); del buf495  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg317_1, (1024, 1024), (1, 1024), 0), out=buf497)
        del arg317_1
        buf498 = reinterpret_tensor(buf466, (4096, 1024), (1024, 1), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg319_1, (1024, 1024), (1, 1024), 0), out=buf498)
        del arg319_1
        buf499 = reinterpret_tensor(buf480, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf496, arg316_1, buf499, 4194304, grid=grid(4194304), stream=stream0)
        del arg316_1
        buf500 = reinterpret_tensor(buf496, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf496  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf497, arg318_1, buf500, 4194304, grid=grid(4194304), stream=stream0)
        del arg318_1
        buf501 = reinterpret_tensor(buf497, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf497  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf498, arg320_1, buf501, 4194304, grid=grid(4194304), stream=stream0)
        del arg320_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf502 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf499, buf500, buf501, None, False, scale=1.0)
        buf503 = buf502[0]
        del buf502
        buf507 = reinterpret_tensor(buf501, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf501  # reuse
        # Topologically Sorted Source Nodes: [attn_output_108], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf503, buf507, 4194304, grid=grid(4194304), stream=stream0)
        buf508 = reinterpret_tensor(buf503, (4096, 1024), (1024, 1), 0); del buf503  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf507, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg321_1, (1024, 1024), (1, 1024), 0), out=buf508)
        del arg321_1
        buf512 = reinterpret_tensor(buf507, (32, 128, 1024), (131072, 1024, 1), 0); del buf507  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_166, hidden_states_167], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf491, buf508, arg322_1, arg323_1, arg324_1, buf512, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg323_1
        del arg324_1
        buf513 = reinterpret_tensor(buf472, (4096, 4096), (4096, 1), 0); del buf472  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf512, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg325_1, (1024, 4096), (1, 1024), 0), out=buf513)
        del arg325_1
        buf514 = reinterpret_tensor(buf513, (32, 128, 4096), (524288, 4096, 1), 0); del buf513  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_168], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf514, arg326_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg326_1
        buf515 = reinterpret_tensor(buf512, (4096, 1024), (1024, 1), 0); del buf512  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf514, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg327_1, (4096, 1024), (1, 4096), 0), out=buf515)
        del arg327_1
        buf516 = reinterpret_tensor(buf515, (32, 128, 1024), (131072, 1024, 1), 0); del buf515  # reuse
        buf520 = reinterpret_tensor(buf500, (32, 128, 1024), (131072, 1024, 1), 0); del buf500  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_166, hidden_states_172, hidden_states_173], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf516, buf491, buf508, arg322_1, arg328_1, arg329_1, arg330_1, buf520, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg322_1
        del arg328_1
        del arg329_1
        del arg330_1
        buf521 = buf508; del buf508  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf520, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg331_1, (1024, 1024), (1, 1024), 0), out=buf521)
        del arg331_1
        buf522 = reinterpret_tensor(buf491, (4096, 1024), (1024, 1), 0); del buf491  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf520, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg333_1, (1024, 1024), (1, 1024), 0), out=buf522)
        del arg333_1
        buf523 = reinterpret_tensor(buf499, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf499  # reuse
        # Topologically Sorted Source Nodes: [contiguous_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf521, arg332_1, buf523, 4194304, grid=grid(4194304), stream=stream0)
        del arg332_1
        buf524 = reinterpret_tensor(buf521, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf521  # reuse
        # Topologically Sorted Source Nodes: [key_states_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf522, arg334_1, buf524, 4194304, grid=grid(4194304), stream=stream0)
        del arg334_1
        buf525 = buf486; del buf486  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf523, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf524, (512, 64, 128), (8192, 1, 64), 0), out=buf525)
        buf529 = buf482; del buf482  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_57], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf525, buf529, 65536, 128, grid=grid(65536), stream=stream0)
        buf528 = reinterpret_tensor(buf524, (4096, 1024), (1024, 1), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf520, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg335_1, (1024, 1024), (1, 1024), 0), out=buf528)
        del arg335_1
        buf530 = reinterpret_tensor(buf520, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [value_states_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf528, arg336_1, buf530, 4194304, grid=grid(4194304), stream=stream0)
        del arg336_1
        buf531 = reinterpret_tensor(buf528, (512, 128, 64), (8192, 64, 1), 0); del buf528  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_57, attn_output_110], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf529, reinterpret_tensor(buf530, (512, 128, 64), (8192, 64, 1), 0), out=buf531)
        buf532 = reinterpret_tensor(buf530, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [attn_output_113], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf531, buf532, 4194304, grid=grid(4194304), stream=stream0)
        buf533 = reinterpret_tensor(buf531, (4096, 1024), (1024, 1), 0); del buf531  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf532, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg337_1, (1024, 1024), (1, 1024), 0), out=buf533)
        del arg337_1
        buf537 = reinterpret_tensor(buf532, (32, 128, 1024), (131072, 1024, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_175, hidden_states_176], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf516, buf533, arg338_1, arg339_1, arg340_1, buf537, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg339_1
        del arg340_1
        buf538 = reinterpret_tensor(buf523, (4096, 1024), (1024, 1), 0); del buf523  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf537, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg341_1, (1024, 1024), (1, 1024), 0), out=buf538)
        del arg341_1
        buf539 = reinterpret_tensor(buf537, (4096, 1024), (1024, 1), 0); del buf537  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg343_1, (1024, 1024), (1, 1024), 0), out=buf539)
        del arg343_1
        buf540 = buf522; del buf522  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg345_1, (1024, 1024), (1, 1024), 0), out=buf540)
        del arg345_1
        buf541 = reinterpret_tensor(buf498, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf498  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf538, arg342_1, buf541, 4194304, grid=grid(4194304), stream=stream0)
        del arg342_1
        buf542 = reinterpret_tensor(buf538, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf538  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf539, arg344_1, buf542, 4194304, grid=grid(4194304), stream=stream0)
        del arg344_1
        buf543 = reinterpret_tensor(buf539, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf539  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf540, arg346_1, buf543, 4194304, grid=grid(4194304), stream=stream0)
        del arg346_1
        del buf540
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf544 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf541, buf542, buf543, None, False, scale=1.0)
        buf545 = buf544[0]
        del buf544
        buf549 = reinterpret_tensor(buf543, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [attn_output_118], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf545, buf549, 4194304, grid=grid(4194304), stream=stream0)
        buf550 = reinterpret_tensor(buf545, (4096, 1024), (1024, 1), 0); del buf545  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf549, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg347_1, (1024, 1024), (1, 1024), 0), out=buf550)
        del arg347_1
        buf551 = reinterpret_tensor(buf550, (32, 128, 1024), (131072, 1024, 1), 0); del buf550  # reuse
        buf555 = reinterpret_tensor(buf549, (32, 128, 1024), (131072, 1024, 1), 0); del buf549  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_175, hidden_states_178, hidden_states_179], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf551, buf516, buf533, arg338_1, arg348_1, arg349_1, arg350_1, buf555, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg338_1
        del arg348_1
        del arg349_1
        del arg350_1
        buf556 = reinterpret_tensor(buf514, (4096, 4096), (4096, 1), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf555, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg351_1, (1024, 4096), (1, 1024), 0), out=buf556)
        del arg351_1
        buf557 = reinterpret_tensor(buf556, (32, 128, 4096), (524288, 4096, 1), 0); del buf556  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_180], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf557, arg352_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg352_1
        buf558 = reinterpret_tensor(buf555, (4096, 1024), (1024, 1), 0); del buf555  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf557, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg353_1, (4096, 1024), (1, 4096), 0), out=buf558)
        del arg353_1
        buf562 = reinterpret_tensor(buf533, (32, 128, 1024), (131072, 1024, 1), 0); del buf533  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_184, hidden_states_185], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf551, buf558, arg354_1, arg355_1, arg356_1, buf562, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg355_1
        del arg356_1
        buf563 = reinterpret_tensor(buf516, (4096, 1024), (1024, 1), 0); del buf516  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf562, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg357_1, (1024, 1024), (1, 1024), 0), out=buf563)
        del arg357_1
        buf564 = reinterpret_tensor(buf542, (4096, 1024), (1024, 1), 0); del buf542  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf562, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg359_1, (1024, 1024), (1, 1024), 0), out=buf564)
        del arg359_1
        buf565 = reinterpret_tensor(buf541, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf541  # reuse
        # Topologically Sorted Source Nodes: [contiguous_74], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf563, arg358_1, buf565, 4194304, grid=grid(4194304), stream=stream0)
        del arg358_1
        buf566 = reinterpret_tensor(buf563, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf563  # reuse
        # Topologically Sorted Source Nodes: [key_states_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf564, arg360_1, buf566, 4194304, grid=grid(4194304), stream=stream0)
        del arg360_1
        del buf564
        buf567 = buf529; del buf529  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf565, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf566, (512, 64, 128), (8192, 1, 64), 0), out=buf567)
        buf571 = buf525; del buf525  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_63], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf567, buf571, 65536, 128, grid=grid(65536), stream=stream0)
        buf570 = reinterpret_tensor(buf566, (4096, 1024), (1024, 1), 0); del buf566  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf562, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg361_1, (1024, 1024), (1, 1024), 0), out=buf570)
        del arg361_1
        buf572 = reinterpret_tensor(buf562, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf562  # reuse
        # Topologically Sorted Source Nodes: [value_states_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf570, arg362_1, buf572, 4194304, grid=grid(4194304), stream=stream0)
        del arg362_1
        buf573 = reinterpret_tensor(buf570, (512, 128, 64), (8192, 64, 1), 0); del buf570  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_63, attn_output_120], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf571, reinterpret_tensor(buf572, (512, 128, 64), (8192, 64, 1), 0), out=buf573)
        buf574 = reinterpret_tensor(buf572, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf572  # reuse
        # Topologically Sorted Source Nodes: [attn_output_123], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf573, buf574, 4194304, grid=grid(4194304), stream=stream0)
        buf575 = reinterpret_tensor(buf573, (4096, 1024), (1024, 1), 0); del buf573  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf574, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg363_1, (1024, 1024), (1, 1024), 0), out=buf575)
        del arg363_1
        buf576 = reinterpret_tensor(buf575, (32, 128, 1024), (131072, 1024, 1), 0); del buf575  # reuse
        buf580 = reinterpret_tensor(buf574, (32, 128, 1024), (131072, 1024, 1), 0); del buf574  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_184, hidden_states_187, hidden_states_188], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf576, buf551, buf558, arg354_1, arg364_1, arg365_1, arg366_1, buf580, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg354_1
        del arg364_1
        del arg365_1
        del arg366_1
        buf581 = buf558; del buf558  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf580, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg367_1, (1024, 1024), (1, 1024), 0), out=buf581)
        del arg367_1
        buf582 = reinterpret_tensor(buf580, (4096, 1024), (1024, 1), 0); del buf580  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg369_1, (1024, 1024), (1, 1024), 0), out=buf582)
        del arg369_1
        buf583 = reinterpret_tensor(buf551, (4096, 1024), (1024, 1), 0); del buf551  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg371_1, (1024, 1024), (1, 1024), 0), out=buf583)
        del arg371_1
        buf584 = reinterpret_tensor(buf565, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf565  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf581, arg368_1, buf584, 4194304, grid=grid(4194304), stream=stream0)
        del arg368_1
        buf585 = reinterpret_tensor(buf581, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf581  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf582, arg370_1, buf585, 4194304, grid=grid(4194304), stream=stream0)
        del arg370_1
        buf586 = reinterpret_tensor(buf582, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf582  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf583, arg372_1, buf586, 4194304, grid=grid(4194304), stream=stream0)
        del arg372_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf587 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf584, buf585, buf586, None, False, scale=1.0)
        buf588 = buf587[0]
        del buf587
        buf592 = reinterpret_tensor(buf586, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf586  # reuse
        # Topologically Sorted Source Nodes: [attn_output_128], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf588, buf592, 4194304, grid=grid(4194304), stream=stream0)
        buf593 = reinterpret_tensor(buf588, (4096, 1024), (1024, 1), 0); del buf588  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf592, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg373_1, (1024, 1024), (1, 1024), 0), out=buf593)
        del arg373_1
        buf597 = reinterpret_tensor(buf592, (32, 128, 1024), (131072, 1024, 1), 0); del buf592  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_190, hidden_states_191], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf576, buf593, arg374_1, arg375_1, arg376_1, buf597, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg375_1
        del arg376_1
        buf598 = reinterpret_tensor(buf557, (4096, 4096), (4096, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf597, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg377_1, (1024, 4096), (1, 1024), 0), out=buf598)
        del arg377_1
        buf599 = reinterpret_tensor(buf598, (32, 128, 4096), (524288, 4096, 1), 0); del buf598  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_192], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf599, arg378_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg378_1
        buf600 = reinterpret_tensor(buf597, (4096, 1024), (1024, 1), 0); del buf597  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf599, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg379_1, (4096, 1024), (1, 4096), 0), out=buf600)
        del arg379_1
        buf601 = reinterpret_tensor(buf600, (32, 128, 1024), (131072, 1024, 1), 0); del buf600  # reuse
        buf605 = reinterpret_tensor(buf585, (32, 128, 1024), (131072, 1024, 1), 0); del buf585  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_190, hidden_states_196, hidden_states_197], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf601, buf576, buf593, arg374_1, arg380_1, arg381_1, arg382_1, buf605, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg374_1
        del arg380_1
        del arg381_1
        del arg382_1
        buf606 = buf593; del buf593  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf605, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg383_1, (1024, 1024), (1, 1024), 0), out=buf606)
        del arg383_1
        buf607 = reinterpret_tensor(buf576, (4096, 1024), (1024, 1), 0); del buf576  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf605, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg385_1, (1024, 1024), (1, 1024), 0), out=buf607)
        del arg385_1
        buf608 = reinterpret_tensor(buf584, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf584  # reuse
        # Topologically Sorted Source Nodes: [contiguous_80], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf606, arg384_1, buf608, 4194304, grid=grid(4194304), stream=stream0)
        del arg384_1
        buf609 = reinterpret_tensor(buf606, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf606  # reuse
        # Topologically Sorted Source Nodes: [key_states_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf607, arg386_1, buf609, 4194304, grid=grid(4194304), stream=stream0)
        del arg386_1
        buf610 = buf571; del buf571  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf608, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf609, (512, 64, 128), (8192, 1, 64), 0), out=buf610)
        buf614 = buf567; del buf567  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_69], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf610, buf614, 65536, 128, grid=grid(65536), stream=stream0)
        buf613 = reinterpret_tensor(buf609, (4096, 1024), (1024, 1), 0); del buf609  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf605, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg387_1, (1024, 1024), (1, 1024), 0), out=buf613)
        del arg387_1
        buf615 = reinterpret_tensor(buf605, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [value_states_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf613, arg388_1, buf615, 4194304, grid=grid(4194304), stream=stream0)
        del arg388_1
        buf616 = reinterpret_tensor(buf613, (512, 128, 64), (8192, 64, 1), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_69, attn_output_130], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf614, reinterpret_tensor(buf615, (512, 128, 64), (8192, 64, 1), 0), out=buf616)
        buf617 = reinterpret_tensor(buf615, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf615  # reuse
        # Topologically Sorted Source Nodes: [attn_output_133], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf616, buf617, 4194304, grid=grid(4194304), stream=stream0)
        buf618 = reinterpret_tensor(buf616, (4096, 1024), (1024, 1), 0); del buf616  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf617, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg389_1, (1024, 1024), (1, 1024), 0), out=buf618)
        del arg389_1
        buf622 = reinterpret_tensor(buf617, (32, 128, 1024), (131072, 1024, 1), 0); del buf617  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_199, hidden_states_200], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf601, buf618, arg390_1, arg391_1, arg392_1, buf622, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg391_1
        del arg392_1
        buf623 = reinterpret_tensor(buf608, (4096, 1024), (1024, 1), 0); del buf608  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf622, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg393_1, (1024, 1024), (1, 1024), 0), out=buf623)
        del arg393_1
        buf624 = reinterpret_tensor(buf622, (4096, 1024), (1024, 1), 0); del buf622  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg395_1, (1024, 1024), (1, 1024), 0), out=buf624)
        del arg395_1
        buf625 = buf607; del buf607  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg397_1, (1024, 1024), (1, 1024), 0), out=buf625)
        del arg397_1
        buf626 = reinterpret_tensor(buf583, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf623, arg394_1, buf626, 4194304, grid=grid(4194304), stream=stream0)
        del arg394_1
        buf627 = reinterpret_tensor(buf623, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf623  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf624, arg396_1, buf627, 4194304, grid=grid(4194304), stream=stream0)
        del arg396_1
        buf628 = reinterpret_tensor(buf624, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf624  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf625, arg398_1, buf628, 4194304, grid=grid(4194304), stream=stream0)
        del arg398_1
        del buf625
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf629 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf626, buf627, buf628, None, False, scale=1.0)
        buf630 = buf629[0]
        del buf629
        buf634 = reinterpret_tensor(buf628, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf628  # reuse
        # Topologically Sorted Source Nodes: [attn_output_138], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf630, buf634, 4194304, grid=grid(4194304), stream=stream0)
        buf635 = reinterpret_tensor(buf630, (4096, 1024), (1024, 1), 0); del buf630  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf634, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg399_1, (1024, 1024), (1, 1024), 0), out=buf635)
        del arg399_1
        buf636 = reinterpret_tensor(buf635, (32, 128, 1024), (131072, 1024, 1), 0); del buf635  # reuse
        buf640 = reinterpret_tensor(buf634, (32, 128, 1024), (131072, 1024, 1), 0); del buf634  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_199, hidden_states_202, hidden_states_203], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf636, buf601, buf618, arg390_1, arg400_1, arg401_1, arg402_1, buf640, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg390_1
        del arg400_1
        del arg401_1
        del arg402_1
        buf641 = reinterpret_tensor(buf599, (4096, 4096), (4096, 1), 0); del buf599  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf640, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg403_1, (1024, 4096), (1, 1024), 0), out=buf641)
        del arg403_1
        buf642 = reinterpret_tensor(buf641, (32, 128, 4096), (524288, 4096, 1), 0); del buf641  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_204], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf642, arg404_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg404_1
        buf643 = reinterpret_tensor(buf640, (4096, 1024), (1024, 1), 0); del buf640  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf642, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg405_1, (4096, 1024), (1, 4096), 0), out=buf643)
        del arg405_1
        buf647 = reinterpret_tensor(buf618, (32, 128, 1024), (131072, 1024, 1), 0); del buf618  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_208, hidden_states_209], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf636, buf643, arg406_1, arg407_1, arg408_1, buf647, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg407_1
        del arg408_1
        buf648 = reinterpret_tensor(buf601, (4096, 1024), (1024, 1), 0); del buf601  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf647, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg409_1, (1024, 1024), (1, 1024), 0), out=buf648)
        del arg409_1
        buf649 = reinterpret_tensor(buf627, (4096, 1024), (1024, 1), 0); del buf627  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf647, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg411_1, (1024, 1024), (1, 1024), 0), out=buf649)
        del arg411_1
        buf650 = reinterpret_tensor(buf626, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf626  # reuse
        # Topologically Sorted Source Nodes: [contiguous_86], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf648, arg410_1, buf650, 4194304, grid=grid(4194304), stream=stream0)
        del arg410_1
        buf651 = reinterpret_tensor(buf648, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf648  # reuse
        # Topologically Sorted Source Nodes: [key_states_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf649, arg412_1, buf651, 4194304, grid=grid(4194304), stream=stream0)
        del arg412_1
        del buf649
        buf652 = buf614; del buf614  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_72], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf650, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf651, (512, 64, 128), (8192, 1, 64), 0), out=buf652)
        buf656 = buf610; del buf610  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_75], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf652, buf656, 65536, 128, grid=grid(65536), stream=stream0)
        buf655 = reinterpret_tensor(buf651, (4096, 1024), (1024, 1), 0); del buf651  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf647, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg413_1, (1024, 1024), (1, 1024), 0), out=buf655)
        del arg413_1
        buf657 = reinterpret_tensor(buf647, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf647  # reuse
        # Topologically Sorted Source Nodes: [value_states_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf655, arg414_1, buf657, 4194304, grid=grid(4194304), stream=stream0)
        del arg414_1
        buf658 = reinterpret_tensor(buf655, (512, 128, 64), (8192, 64, 1), 0); del buf655  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_75, attn_output_140], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf656, reinterpret_tensor(buf657, (512, 128, 64), (8192, 64, 1), 0), out=buf658)
        buf659 = reinterpret_tensor(buf657, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf657  # reuse
        # Topologically Sorted Source Nodes: [attn_output_143], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf658, buf659, 4194304, grid=grid(4194304), stream=stream0)
        buf660 = reinterpret_tensor(buf658, (4096, 1024), (1024, 1), 0); del buf658  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf659, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg415_1, (1024, 1024), (1, 1024), 0), out=buf660)
        del arg415_1
        buf661 = reinterpret_tensor(buf660, (32, 128, 1024), (131072, 1024, 1), 0); del buf660  # reuse
        buf665 = reinterpret_tensor(buf659, (32, 128, 1024), (131072, 1024, 1), 0); del buf659  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_208, hidden_states_211, hidden_states_212], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf661, buf636, buf643, arg406_1, arg416_1, arg417_1, arg418_1, buf665, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg406_1
        del arg416_1
        del arg417_1
        del arg418_1
        buf666 = buf643; del buf643  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf665, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg419_1, (1024, 1024), (1, 1024), 0), out=buf666)
        del arg419_1
        buf667 = reinterpret_tensor(buf665, (4096, 1024), (1024, 1), 0); del buf665  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg421_1, (1024, 1024), (1, 1024), 0), out=buf667)
        del arg421_1
        buf668 = reinterpret_tensor(buf636, (4096, 1024), (1024, 1), 0); del buf636  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg423_1, (1024, 1024), (1, 1024), 0), out=buf668)
        del arg423_1
        buf669 = reinterpret_tensor(buf650, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf650  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf666, arg420_1, buf669, 4194304, grid=grid(4194304), stream=stream0)
        del arg420_1
        buf670 = reinterpret_tensor(buf666, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf666  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf667, arg422_1, buf670, 4194304, grid=grid(4194304), stream=stream0)
        del arg422_1
        buf671 = reinterpret_tensor(buf667, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf667  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf668, arg424_1, buf671, 4194304, grid=grid(4194304), stream=stream0)
        del arg424_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf672 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf669, buf670, buf671, None, False, scale=1.0)
        buf673 = buf672[0]
        del buf672
        buf677 = reinterpret_tensor(buf671, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf671  # reuse
        # Topologically Sorted Source Nodes: [attn_output_148], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf673, buf677, 4194304, grid=grid(4194304), stream=stream0)
        buf678 = reinterpret_tensor(buf673, (4096, 1024), (1024, 1), 0); del buf673  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf677, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg425_1, (1024, 1024), (1, 1024), 0), out=buf678)
        del arg425_1
        buf682 = reinterpret_tensor(buf677, (32, 128, 1024), (131072, 1024, 1), 0); del buf677  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_214, hidden_states_215], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf661, buf678, arg426_1, arg427_1, arg428_1, buf682, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg427_1
        del arg428_1
        buf683 = reinterpret_tensor(buf642, (4096, 4096), (4096, 1), 0); del buf642  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf682, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg429_1, (1024, 4096), (1, 1024), 0), out=buf683)
        del arg429_1
        buf684 = reinterpret_tensor(buf683, (32, 128, 4096), (524288, 4096, 1), 0); del buf683  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_216], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf684, arg430_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg430_1
        buf685 = reinterpret_tensor(buf682, (4096, 1024), (1024, 1), 0); del buf682  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf684, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg431_1, (4096, 1024), (1, 4096), 0), out=buf685)
        del arg431_1
        buf686 = reinterpret_tensor(buf685, (32, 128, 1024), (131072, 1024, 1), 0); del buf685  # reuse
        buf690 = reinterpret_tensor(buf670, (32, 128, 1024), (131072, 1024, 1), 0); del buf670  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_214, hidden_states_220, hidden_states_221], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf686, buf661, buf678, arg426_1, arg432_1, arg433_1, arg434_1, buf690, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg426_1
        del arg432_1
        del arg433_1
        del arg434_1
        buf691 = buf678; del buf678  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf690, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg435_1, (1024, 1024), (1, 1024), 0), out=buf691)
        del arg435_1
        buf692 = reinterpret_tensor(buf661, (4096, 1024), (1024, 1), 0); del buf661  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf690, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg437_1, (1024, 1024), (1, 1024), 0), out=buf692)
        del arg437_1
        buf693 = reinterpret_tensor(buf669, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf669  # reuse
        # Topologically Sorted Source Nodes: [contiguous_92], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf691, arg436_1, buf693, 4194304, grid=grid(4194304), stream=stream0)
        del arg436_1
        buf694 = reinterpret_tensor(buf691, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf691  # reuse
        # Topologically Sorted Source Nodes: [key_states_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf692, arg438_1, buf694, 4194304, grid=grid(4194304), stream=stream0)
        del arg438_1
        buf695 = buf656; del buf656  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_78], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf693, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf694, (512, 64, 128), (8192, 1, 64), 0), out=buf695)
        buf699 = buf652; del buf652  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_81], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf695, buf699, 65536, 128, grid=grid(65536), stream=stream0)
        buf698 = reinterpret_tensor(buf694, (4096, 1024), (1024, 1), 0); del buf694  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf690, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg439_1, (1024, 1024), (1, 1024), 0), out=buf698)
        del arg439_1
        buf700 = reinterpret_tensor(buf690, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf690  # reuse
        # Topologically Sorted Source Nodes: [value_states_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf698, arg440_1, buf700, 4194304, grid=grid(4194304), stream=stream0)
        del arg440_1
        buf701 = reinterpret_tensor(buf698, (512, 128, 64), (8192, 64, 1), 0); del buf698  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_81, attn_output_150], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf699, reinterpret_tensor(buf700, (512, 128, 64), (8192, 64, 1), 0), out=buf701)
        buf702 = reinterpret_tensor(buf700, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf700  # reuse
        # Topologically Sorted Source Nodes: [attn_output_153], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf701, buf702, 4194304, grid=grid(4194304), stream=stream0)
        buf703 = reinterpret_tensor(buf701, (4096, 1024), (1024, 1), 0); del buf701  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf702, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg441_1, (1024, 1024), (1, 1024), 0), out=buf703)
        del arg441_1
        buf707 = reinterpret_tensor(buf702, (32, 128, 1024), (131072, 1024, 1), 0); del buf702  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_223, hidden_states_224], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf686, buf703, arg442_1, arg443_1, arg444_1, buf707, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg443_1
        del arg444_1
        buf708 = reinterpret_tensor(buf693, (4096, 1024), (1024, 1), 0); del buf693  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf707, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg445_1, (1024, 1024), (1, 1024), 0), out=buf708)
        del arg445_1
        buf709 = reinterpret_tensor(buf707, (4096, 1024), (1024, 1), 0); del buf707  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg447_1, (1024, 1024), (1, 1024), 0), out=buf709)
        del arg447_1
        buf710 = buf692; del buf692  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg449_1, (1024, 1024), (1, 1024), 0), out=buf710)
        del arg449_1
        buf711 = reinterpret_tensor(buf668, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf668  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf708, arg446_1, buf711, 4194304, grid=grid(4194304), stream=stream0)
        del arg446_1
        buf712 = reinterpret_tensor(buf708, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf708  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf709, arg448_1, buf712, 4194304, grid=grid(4194304), stream=stream0)
        del arg448_1
        buf713 = reinterpret_tensor(buf709, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf709  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf710, arg450_1, buf713, 4194304, grid=grid(4194304), stream=stream0)
        del arg450_1
        del buf710
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf714 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf711, buf712, buf713, None, False, scale=1.0)
        buf715 = buf714[0]
        del buf714
        buf719 = reinterpret_tensor(buf713, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf713  # reuse
        # Topologically Sorted Source Nodes: [attn_output_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf715, buf719, 4194304, grid=grid(4194304), stream=stream0)
        buf720 = reinterpret_tensor(buf715, (4096, 1024), (1024, 1), 0); del buf715  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf719, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg451_1, (1024, 1024), (1, 1024), 0), out=buf720)
        del arg451_1
        buf721 = reinterpret_tensor(buf720, (32, 128, 1024), (131072, 1024, 1), 0); del buf720  # reuse
        buf725 = reinterpret_tensor(buf719, (32, 128, 1024), (131072, 1024, 1), 0); del buf719  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_223, hidden_states_226, hidden_states_227], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf721, buf686, buf703, arg442_1, arg452_1, arg453_1, arg454_1, buf725, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg442_1
        del arg452_1
        del arg453_1
        del arg454_1
        buf726 = reinterpret_tensor(buf684, (4096, 4096), (4096, 1), 0); del buf684  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf725, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg455_1, (1024, 4096), (1, 1024), 0), out=buf726)
        del arg455_1
        buf727 = reinterpret_tensor(buf726, (32, 128, 4096), (524288, 4096, 1), 0); del buf726  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_228], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf727, arg456_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg456_1
        buf728 = reinterpret_tensor(buf725, (4096, 1024), (1024, 1), 0); del buf725  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf727, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg457_1, (4096, 1024), (1, 4096), 0), out=buf728)
        del arg457_1
        buf732 = reinterpret_tensor(buf703, (32, 128, 1024), (131072, 1024, 1), 0); del buf703  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_232, hidden_states_233], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf721, buf728, arg458_1, arg459_1, arg460_1, buf732, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg459_1
        del arg460_1
        buf733 = reinterpret_tensor(buf686, (4096, 1024), (1024, 1), 0); del buf686  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf732, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg461_1, (1024, 1024), (1, 1024), 0), out=buf733)
        del arg461_1
        buf734 = reinterpret_tensor(buf712, (4096, 1024), (1024, 1), 0); del buf712  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf732, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg463_1, (1024, 1024), (1, 1024), 0), out=buf734)
        del arg463_1
        buf735 = reinterpret_tensor(buf711, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf711  # reuse
        # Topologically Sorted Source Nodes: [contiguous_98], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf733, arg462_1, buf735, 4194304, grid=grid(4194304), stream=stream0)
        del arg462_1
        buf736 = reinterpret_tensor(buf733, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf733  # reuse
        # Topologically Sorted Source Nodes: [key_states_64], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf734, arg464_1, buf736, 4194304, grid=grid(4194304), stream=stream0)
        del arg464_1
        del buf734
        buf737 = buf699; del buf699  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf735, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf736, (512, 64, 128), (8192, 1, 64), 0), out=buf737)
        buf741 = buf695; del buf695  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_87], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf737, buf741, 65536, 128, grid=grid(65536), stream=stream0)
        buf740 = reinterpret_tensor(buf736, (4096, 1024), (1024, 1), 0); del buf736  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf732, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg465_1, (1024, 1024), (1, 1024), 0), out=buf740)
        del arg465_1
        buf742 = reinterpret_tensor(buf732, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf732  # reuse
        # Topologically Sorted Source Nodes: [value_states_64], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf740, arg466_1, buf742, 4194304, grid=grid(4194304), stream=stream0)
        del arg466_1
        buf743 = reinterpret_tensor(buf740, (512, 128, 64), (8192, 64, 1), 0); del buf740  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_87, attn_output_160], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf741, reinterpret_tensor(buf742, (512, 128, 64), (8192, 64, 1), 0), out=buf743)
        buf744 = reinterpret_tensor(buf742, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf742  # reuse
        # Topologically Sorted Source Nodes: [attn_output_163], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf743, buf744, 4194304, grid=grid(4194304), stream=stream0)
        buf745 = reinterpret_tensor(buf743, (4096, 1024), (1024, 1), 0); del buf743  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf744, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg467_1, (1024, 1024), (1, 1024), 0), out=buf745)
        del arg467_1
        buf746 = reinterpret_tensor(buf745, (32, 128, 1024), (131072, 1024, 1), 0); del buf745  # reuse
        buf750 = reinterpret_tensor(buf744, (32, 128, 1024), (131072, 1024, 1), 0); del buf744  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_232, hidden_states_235, hidden_states_236], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf746, buf721, buf728, arg458_1, arg468_1, arg469_1, arg470_1, buf750, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg458_1
        del arg468_1
        del arg469_1
        del arg470_1
        buf751 = buf728; del buf728  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf750, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg471_1, (1024, 1024), (1, 1024), 0), out=buf751)
        del arg471_1
        buf752 = reinterpret_tensor(buf750, (4096, 1024), (1024, 1), 0); del buf750  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg473_1, (1024, 1024), (1, 1024), 0), out=buf752)
        del arg473_1
        buf753 = reinterpret_tensor(buf721, (4096, 1024), (1024, 1), 0); del buf721  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg475_1, (1024, 1024), (1, 1024), 0), out=buf753)
        del arg475_1
        buf754 = reinterpret_tensor(buf735, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf735  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf751, arg472_1, buf754, 4194304, grid=grid(4194304), stream=stream0)
        del arg472_1
        buf755 = reinterpret_tensor(buf751, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf751  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf752, arg474_1, buf755, 4194304, grid=grid(4194304), stream=stream0)
        del arg474_1
        buf756 = reinterpret_tensor(buf752, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf752  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf753, arg476_1, buf756, 4194304, grid=grid(4194304), stream=stream0)
        del arg476_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf757 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf754, buf755, buf756, None, False, scale=1.0)
        buf758 = buf757[0]
        del buf757
        buf762 = reinterpret_tensor(buf756, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf756  # reuse
        # Topologically Sorted Source Nodes: [attn_output_168], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf758, buf762, 4194304, grid=grid(4194304), stream=stream0)
        buf763 = reinterpret_tensor(buf758, (4096, 1024), (1024, 1), 0); del buf758  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf762, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg477_1, (1024, 1024), (1, 1024), 0), out=buf763)
        del arg477_1
        buf767 = reinterpret_tensor(buf762, (32, 128, 1024), (131072, 1024, 1), 0); del buf762  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_238, hidden_states_239], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf746, buf763, arg478_1, arg479_1, arg480_1, buf767, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg479_1
        del arg480_1
        buf768 = reinterpret_tensor(buf727, (4096, 4096), (4096, 1), 0); del buf727  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf767, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg481_1, (1024, 4096), (1, 1024), 0), out=buf768)
        del arg481_1
        buf769 = reinterpret_tensor(buf768, (32, 128, 4096), (524288, 4096, 1), 0); del buf768  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_240], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf769, arg482_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg482_1
        buf770 = reinterpret_tensor(buf767, (4096, 1024), (1024, 1), 0); del buf767  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf769, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg483_1, (4096, 1024), (1, 4096), 0), out=buf770)
        del arg483_1
        buf771 = reinterpret_tensor(buf770, (32, 128, 1024), (131072, 1024, 1), 0); del buf770  # reuse
        buf775 = reinterpret_tensor(buf755, (32, 128, 1024), (131072, 1024, 1), 0); del buf755  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_238, hidden_states_244, hidden_states_245], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf771, buf746, buf763, arg478_1, arg484_1, arg485_1, arg486_1, buf775, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg478_1
        del arg484_1
        del arg485_1
        del arg486_1
        buf776 = buf763; del buf763  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf775, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg487_1, (1024, 1024), (1, 1024), 0), out=buf776)
        del arg487_1
        buf777 = reinterpret_tensor(buf746, (4096, 1024), (1024, 1), 0); del buf746  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf775, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg489_1, (1024, 1024), (1, 1024), 0), out=buf777)
        del arg489_1
        buf778 = reinterpret_tensor(buf754, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf754  # reuse
        # Topologically Sorted Source Nodes: [contiguous_104], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf776, arg488_1, buf778, 4194304, grid=grid(4194304), stream=stream0)
        del arg488_1
        buf779 = reinterpret_tensor(buf776, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf776  # reuse
        # Topologically Sorted Source Nodes: [key_states_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf777, arg490_1, buf779, 4194304, grid=grid(4194304), stream=stream0)
        del arg490_1
        buf780 = buf741; del buf741  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf778, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf779, (512, 64, 128), (8192, 1, 64), 0), out=buf780)
        buf784 = buf737; del buf737  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_93], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf780, buf784, 65536, 128, grid=grid(65536), stream=stream0)
        del buf780
        buf783 = reinterpret_tensor(buf779, (4096, 1024), (1024, 1), 0); del buf779  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf775, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg491_1, (1024, 1024), (1, 1024), 0), out=buf783)
        del arg491_1
        buf785 = reinterpret_tensor(buf775, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf775  # reuse
        # Topologically Sorted Source Nodes: [value_states_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf783, arg492_1, buf785, 4194304, grid=grid(4194304), stream=stream0)
        del arg492_1
        buf786 = reinterpret_tensor(buf783, (512, 128, 64), (8192, 64, 1), 0); del buf783  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_93, attn_output_170], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf784, reinterpret_tensor(buf785, (512, 128, 64), (8192, 64, 1), 0), out=buf786)
        del buf784
        buf787 = reinterpret_tensor(buf785, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf785  # reuse
        # Topologically Sorted Source Nodes: [attn_output_173], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf786, buf787, 4194304, grid=grid(4194304), stream=stream0)
        buf788 = reinterpret_tensor(buf786, (4096, 1024), (1024, 1), 0); del buf786  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf787, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg493_1, (1024, 1024), (1, 1024), 0), out=buf788)
        del arg493_1
        buf792 = reinterpret_tensor(buf787, (32, 128, 1024), (131072, 1024, 1), 0); del buf787  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_247, hidden_states_248], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf771, buf788, arg494_1, arg495_1, arg496_1, buf792, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg495_1
        del arg496_1
        buf793 = reinterpret_tensor(buf778, (4096, 1024), (1024, 1), 0); del buf778  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf792, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg497_1, (1024, 1024), (1, 1024), 0), out=buf793)
        del arg497_1
        buf794 = reinterpret_tensor(buf792, (4096, 1024), (1024, 1), 0); del buf792  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg499_1, (1024, 1024), (1, 1024), 0), out=buf794)
        del arg499_1
        buf795 = buf777; del buf777  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg501_1, (1024, 1024), (1, 1024), 0), out=buf795)
        del arg501_1
        buf796 = reinterpret_tensor(buf753, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf753  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf793, arg498_1, buf796, 4194304, grid=grid(4194304), stream=stream0)
        del arg498_1
        buf797 = reinterpret_tensor(buf793, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf793  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf794, arg500_1, buf797, 4194304, grid=grid(4194304), stream=stream0)
        del arg500_1
        buf798 = reinterpret_tensor(buf794, (1, 512, 128, 64), (4194304, 8192, 64, 1), 0); del buf794  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf795, arg502_1, buf798, 4194304, grid=grid(4194304), stream=stream0)
        del arg502_1
        del buf795
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf799 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf796, buf797, buf798, None, False, scale=1.0)
        del buf796
        del buf797
        buf800 = buf799[0]
        del buf799
        buf804 = reinterpret_tensor(buf798, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf798  # reuse
        # Topologically Sorted Source Nodes: [attn_output_178], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf800, buf804, 4194304, grid=grid(4194304), stream=stream0)
        buf805 = reinterpret_tensor(buf800, (4096, 1024), (1024, 1), 0); del buf800  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf804, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg503_1, (1024, 1024), (1, 1024), 0), out=buf805)
        del arg503_1
        buf806 = reinterpret_tensor(buf805, (32, 128, 1024), (131072, 1024, 1), 0); del buf805  # reuse
        buf810 = reinterpret_tensor(buf804, (32, 128, 1024), (131072, 1024, 1), 0); del buf804  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_247, hidden_states_250, hidden_states_251], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf806, buf771, buf788, arg494_1, arg504_1, arg505_1, arg506_1, buf810, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg494_1
        del arg504_1
        del arg505_1
        del arg506_1
        del buf771
        buf811 = reinterpret_tensor(buf769, (4096, 4096), (4096, 1), 0); del buf769  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf810, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg507_1, (1024, 4096), (1, 1024), 0), out=buf811)
        del arg507_1
        buf812 = reinterpret_tensor(buf811, (32, 128, 4096), (524288, 4096, 1), 0); del buf811  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_252], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf812, arg508_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg508_1
        buf813 = reinterpret_tensor(buf810, (4096, 1024), (1024, 1), 0); del buf810  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf812, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg509_1, (4096, 1024), (1, 4096), 0), out=buf813)
        del arg509_1
        del buf812
        buf817 = reinterpret_tensor(buf788, (32, 128, 1024), (131072, 1024, 1), 0); del buf788  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_256, hidden_states_257], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf806, buf813, arg510_1, arg511_1, arg512_1, buf817, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg510_1
        del arg511_1
        del arg512_1
        del buf806
        del buf813
        buf818 = empty_strided_cuda((1024, 50268), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(arg2_1, buf818, 51474432, grid=grid(51474432), stream=stream0)
        del arg2_1
        buf819 = empty_strided_cuda((4096, 50268), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf817, (4096, 1024), (1024, 1), 0), buf818, out=buf819)
        del buf817
        del buf818
        buf820 = empty_strided_cuda((32, 128, 50265), (6433920, 50265, 1), torch.float32)
        buf821 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        buf822 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [lm_logits, masked_lm_loss], Original ATen: [aten.add, aten._log_softmax]
        triton_red_fused__log_softmax_add_13.run(buf819, arg513_1, buf820, buf821, buf822, 4096, 50265, grid=grid(4096), stream=stream0)
        del arg513_1
        del buf819
        buf823 = empty_strided_cuda((), (), torch.float32)
        buf825 = buf823; del buf823  # reuse
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_14.run(buf825, arg0_1, buf820, buf821, buf822, 1, 4096, grid=grid(1), stream=stream0)
        del arg0_1
        del buf821
        del buf822
    return (buf825, buf820, buf326, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
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
    arg198_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
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
    compiled_module_main('PegasusForConditionalGeneration', benchmark_compiled_module)
