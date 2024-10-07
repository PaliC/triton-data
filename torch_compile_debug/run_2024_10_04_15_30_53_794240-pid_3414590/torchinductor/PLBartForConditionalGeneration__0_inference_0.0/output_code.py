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


# kernel path: /tmp/torchinductor_sahanp/r7/cr7a4sctesp6paddcwbsoersptbnwdi57vc66x3nlk2g2ne673dx.py
# Topologically Sorted Source Nodes: [embedding, inputs_embeds, add, embed_pos, hidden_states, hidden_states_1], Original ATen: [aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add
#   embed_pos => embedding_1
#   embedding => embedding
#   hidden_states => add_1
#   hidden_states_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
#   inputs_embeds => mul
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %view, 1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, 27.712812921102035), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand, 2), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %add), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg4_1), kwargs = {})
#   %add_3 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg5_1), kwargs = {})
triton_red_fused_add_embedding_mul_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x0 = xindex % 1024
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr2 + (1536 + r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 50005, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert((0 <= tmp4) & (tmp4 < 50005), "index out of bounds: 0 <= tmp4 < 50005")
        tmp6 = tl.load(in_ptr1 + (r2 + (768*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 27.712812921102035
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
        tmp23 = tl.load(in_ptr2 + (1536 + r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.full([XBLOCK, RBLOCK], 50005, tl.int32)
        tmp16 = tmp0 + tmp15
        tmp17 = tmp0 < 0
        tmp18 = tl.where(tmp17, tmp16, tmp0)
        tl.device_assert((0 <= tmp18) & (tmp18 < 50005), "index out of bounds: 0 <= tmp18 < 50005")
        tmp20 = tl.load(in_ptr1 + (r2 + (768*tmp18)), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = 27.712812921102035
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tmp25 = tmp24 - tmp12
        tmp26 = 768.0
        tmp27 = tmp13 / tmp26
        tmp28 = 1e-05
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp33 = tmp31 * tmp32
        tmp35 = tmp33 + tmp34
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6x/c6xheeycue7j6nuc5wxy2zqr2su2fbd6ps77ugiqrohowr6xq36q.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %_scaled_dot_product_efficient_attention_default_11 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%unsqueeze_default_33, %unsqueeze_default_34, %unsqueeze_default_35, None, False), kwargs = {scale: 1.0})
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
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*(x2 % 12)) + (768*x1) + (786432*(x2 // 12))), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*(x2 % 12))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rb/crb3dqkh7u5zvm54pge34qduoe64dpk3whznbi26j4kkayvz7z7x.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %_scaled_dot_product_efficient_attention_default_11 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%unsqueeze_default_33, %unsqueeze_default_34, %unsqueeze_default_35, None, False), kwargs = {scale: 1.0})
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
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*(x2 % 12)) + (768*x1) + (786432*(x2 // 12))), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*(x2 % 12))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qm/cqmn5clr6lqacaflyv6jpnj5xaxoybzfyt2hxgvsxsfsojyuj7ka.py
# Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_output_3 => clone_7
# Graph fragment:
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
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
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 768
    x1 = (xindex // 768) % 1024
    x2 = (xindex // 786432)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*x2) + (3072*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ey/ceycu5d3andwifydbh37nqk7odoob6k3i5gzisqwgqb6vwq42s5n.py
# Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_5], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_4 => add_4
#   hidden_states_5 => add_5, add_6, mul_4, mul_5, rsqrt_1, sub_3, var_mean_1
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_16), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_3), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg14_1), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg15_1), kwargs = {})
triton_per_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_4', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
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
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bd/cbde7bk5lwpyl72wf3jaberwtbzd2ypjpuafqeogkls7fdvijr3v.py
# Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_6 => add_7, erf, mul_6, mul_7, mul_8
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_18, 0.5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_18, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_7,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %add_7), kwargs = {})
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
    xnumel = 12582912
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


# kernel path: /tmp/torchinductor_sahanp/2h/c2hfkjum6qhxdxznd4y4hnfy4n3wqiltwnkqziokq6wrrj3n3gkr.py
# Topologically Sorted Source Nodes: [eq, masked_fill_, ne, sum_1], Original ATen: [aten.eq, aten.masked_fill, aten.ne, aten.sum]
# Source node to ATen node mapping:
#   eq => eq
#   masked_fill_ => full_default, where
#   ne => ne
#   sum_1 => sum_1
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg0_1, -100), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=5] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %arg0_1), kwargs = {})
#   %ne : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%where, 1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%ne, [1]), kwargs = {})
triton_per_fused_eq_masked_fill_ne_sum_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_eq_masked_fill_ne_sum_6', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 4
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
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tl.where(tmp2, tmp3, tmp0)
    tmp5 = tmp4 != tmp3
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tl.store(out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7e/c7e25nyy3zgmgkq7snbltdgyp5acdhz2tvpyq7pfe64qewt7w3xg.py
# Topologically Sorted Source Nodes: [eq, masked_fill_, clone_1, setitem, setitem_1, embedding_2, inputs_embeds_1, add_15, positions_2, hidden_states_57, hidden_states_58], Original ATen: [aten.eq, aten.masked_fill, aten.clone, aten.copy, aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_15 => add_47
#   clone_1 => clone_1
#   embedding_2 => embedding_2
#   eq => eq
#   hidden_states_57 => add_48
#   hidden_states_58 => add_49, add_50, mul_52, mul_53, rsqrt_13, sub_20, var_mean_13
#   inputs_embeds_1 => mul_51
#   masked_fill_ => full_default, where
#   positions_2 => embedding_3
#   setitem => copy
#   setitem_1 => copy_1
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg0_1, -100), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=5] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %arg0_1), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_8, %clone_1), kwargs = {})
#   %slice_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%where, %copy, 1, 1, 9223372036854775807), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_1, %squeeze), kwargs = {})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%slice_scatter_default, %copy_1, 1, 0), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %select_scatter_default, 1), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding_2, 27.712812921102035), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_2, 2), kwargs = {})
#   %embedding_3 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg102_1, %add_47), kwargs = {})
#   %add_48 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %embedding_3), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_48, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_48, %getitem_27), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-05), kwargs = {})
#   %rsqrt_13 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_49,), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %rsqrt_13), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %arg103_1), kwargs = {})
#   %add_50 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %arg104_1), kwargs = {})
triton_red_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp3 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    x3 = xindex
    tmp21 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp37_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp37_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp37_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp34 = tl.load(in_ptr3 + (1536 + r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp3 - tmp4
        tmp6 = tl.full([XBLOCK, RBLOCK], 1024, tl.int32)
        tmp7 = tmp5 + tmp6
        tmp8 = tmp5 < 0
        tmp9 = tl.where(tmp8, tmp7, tmp5)
        tl.device_assert((0 <= tmp9) & (tmp9 < 1024), "index out of bounds: 0 <= tmp9 < 1024")
        tmp11 = tl.load(in_ptr1 + (tmp9 + (1024*x1)), None, eviction_policy='evict_last')
        tmp12 = tl.full([1, 1], -100, tl.int64)
        tmp13 = tmp11 == tmp12
        tmp14 = tl.where(tmp13, tmp4, tmp11)
        tmp15 = tmp0 >= tmp4
        tmp16 = tl.load(in_ptr1 + (tl.broadcast_to((-1) + x3, [XBLOCK, RBLOCK])), rmask & tmp15, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp16 == tmp12
        tmp18 = tl.where(tmp17, tmp4, tmp16)
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp15, tmp18, tmp19)
        tmp22 = tmp21 == tmp12
        tmp23 = tl.where(tmp22, tmp4, tmp21)
        tmp24 = tl.where(tmp15, tmp20, tmp23)
        tmp25 = tl.where(tmp2, tmp14, tmp24)
        tmp26 = tl.full([XBLOCK, RBLOCK], 50005, tl.int32)
        tmp27 = tmp25 + tmp26
        tmp28 = tmp25 < 0
        tmp29 = tl.where(tmp28, tmp27, tmp25)
        tl.device_assert(((0 <= tmp29) & (tmp29 < 50005)) | ~(rmask), "index out of bounds: 0 <= tmp29 < 50005")
        tmp31 = tl.load(in_ptr2 + (r2 + (768*tmp29)), rmask, eviction_policy='evict_first', other=0.0)
        tmp32 = 27.712812921102035
        tmp33 = tmp31 * tmp32
        tmp35 = tmp33 + tmp34
        tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
        tmp37_mean_next, tmp37_m2_next, tmp37_weight_next = triton_helpers.welford_reduce(
            tmp36, tmp37_mean, tmp37_m2, tmp37_weight, roffset == 0
        )
        tmp37_mean = tl.where(rmask, tmp37_mean_next, tmp37_mean)
        tmp37_m2 = tl.where(rmask, tmp37_m2_next, tmp37_m2)
        tmp37_weight = tl.where(rmask, tmp37_weight_next, tmp37_weight)
        tl.store(out_ptr0 + (r2 + (768*x3)), tmp35, rmask)
    tmp37_tmp, tmp38_tmp, tmp39_tmp = triton_helpers.welford(
        tmp37_mean, tmp37_m2, tmp37_weight, 1
    )
    tmp37 = tmp37_tmp[:, None]
    tmp38 = tmp38_tmp[:, None]
    tmp39 = tmp39_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp40 = tl.load(out_ptr0 + (r2 + (768*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp48 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp50 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tmp40 - tmp37
        tmp42 = 768.0
        tmp43 = tmp38 / tmp42
        tmp44 = 1e-05
        tmp45 = tmp43 + tmp44
        tmp46 = libdevice.rsqrt(tmp45)
        tmp47 = tmp41 * tmp46
        tmp49 = tmp47 * tmp48
        tmp51 = tmp49 + tmp50
        tl.store(out_ptr3 + (r2 + (768*x3)), tmp51, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tt/ctt2ztzmskdgnwznnjzmr5spkugc7sjnsqbedq34opx6j4rumlk3.py
# Topologically Sorted Source Nodes: [key_states_12], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   key_states_12 => clone_52
# Graph fragment:
#   %clone_52 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_68,), kwargs = {memory_format: torch.contiguous_format})
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
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536) % 12
    x3 = (xindex // 786432)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (786432*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xc/cxcovqkjsuuyitr7amrdjfdhoeytm2j7fverhqkhct5zne6fzugm.py
# Topologically Sorted Source Nodes: [contiguous_20], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_20 => clone_54
# Graph fragment:
#   %clone_54 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_71,), kwargs = {memory_format: torch.contiguous_format})
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
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536) % 12
    x3 = (xindex // 786432)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (786432*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/b5/cb55kfpnsivq3hgthxtj6i7qkazic2otu27ja4arieburbc2rgs7.py
# Topologically Sorted Source Nodes: [attn_weights_15], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_weights_15 => amax_6, div_6, exp_6, sub_21, sum_8
# Graph fragment:
#   %amax_6 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_136, [-1], True), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_136, %amax_6), kwargs = {})
#   %exp_6 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_21,), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_6, [-1], True), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_6, %sum_8), kwargs = {})
triton_per_fused__softmax_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_10', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 49152
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), None)
    tmp1 = r2
    tmp2 = 1 + x0
    tmp3 = tmp1 < tmp2
    tmp4 = 0.0
    tmp5 = -3.4028234663852886e+38
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 + tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp8, 0))
    tmp11 = tmp7 - tmp10
    tmp12 = tl_math.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp12 / tmp15
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/m5/cm54wbu4jchc3rarxyx5qrbymjb7gut4pukvindl4xpyet74anfm.py
# Topologically Sorted Source Nodes: [attn_output_33], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_output_33 => clone_56
# Graph fragment:
#   %clone_56 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_73,), kwargs = {memory_format: torch.contiguous_format})
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
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768) % 1024
    x3 = (xindex // 786432)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (65536*x1) + (786432*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/st/cst45oqncr3wuapu5mnf3rqun4v2tgybmw47wgfddfmhi3byszhu.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_186, %full_default_5], 1), kwargs = {})
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
    xnumel = 38406144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 50008
    x1 = (xindex // 50008)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 50005, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + (768*x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 50008, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x0 + (50016*x1)), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g2/cg2vcsf2wrszfgvnz5yh3argjiemixxibw34afabosyqathn5sou.py
# Topologically Sorted Source Nodes: [lm_logits_1, masked_lm_loss], Original ATen: [aten.add, aten._log_softmax]
# Source node to ATen node mapping:
#   lm_logits_1 => add_117
#   masked_lm_loss => amax_18, exp_18, sub_51, sum_20
# Graph fragment:
#   %add_117 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_352, %arg261_1), kwargs = {})
#   %amax_18 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_353, [1], True), kwargs = {})
#   %sub_51 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_353, %amax_18), kwargs = {})
#   %exp_18 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_51,), kwargs = {})
#   %sum_20 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_18, [1], True), kwargs = {})
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
    rnumel = 50005
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
        tmp0 = tl.load(in_ptr0 + (r1 + (50016*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tl.store(out_ptr0 + (r1 + (50005*x0)), tmp2, rmask)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp4, None)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(out_ptr0 + (r1 + (50005*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp6 - tmp4
        tmp8 = tl_math.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ah/cahees6ut2f7huqendee6unttukdtttvzgsbpr7nxfxdy4b43fkb.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   masked_lm_loss => convert_element_type, div_18, full_default_4, ne_2, ne_3, neg, sum_21, sum_22, where_3
# Graph fragment:
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_354, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_1,), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_2, %neg, %full_default_4), kwargs = {})
#   %sum_22 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_3,), kwargs = {})
#   %ne_3 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_354, -100), kwargs = {})
#   %sum_21 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_3,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_21, torch.float32), kwargs = {})
#   %div_18 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_22, %convert_element_type), kwargs = {})
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
        tmp5 = tl.full([XBLOCK, RBLOCK], 50005, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 50005)) | ~(rmask), "index out of bounds: 0 <= tmp8 < 50005")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (50005*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 1024), (1024, 1))
    assert_size_stride(arg1_1, (4, 1024), (1024, 1))
    assert_size_stride(arg2_1, (50005, 768), (768, 1))
    assert_size_stride(arg3_1, (1026, 768), (768, 1))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, 768), (768, 1))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, 768), (768, 1))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, 768), (768, 1))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, 768), (768, 1))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (3072, 768), (768, 1))
    assert_size_stride(arg17_1, (3072, ), (1, ))
    assert_size_stride(arg18_1, (768, 3072), (3072, 1))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, 768), (768, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, 768), (768, 1))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, 768), (768, 1))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, 768), (768, 1))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (3072, 768), (768, 1))
    assert_size_stride(arg33_1, (3072, ), (1, ))
    assert_size_stride(arg34_1, (768, 3072), (3072, 1))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, 768), (768, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, 768), (768, 1))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, 768), (768, 1))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, 768), (768, 1))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (3072, 768), (768, 1))
    assert_size_stride(arg49_1, (3072, ), (1, ))
    assert_size_stride(arg50_1, (768, 3072), (3072, 1))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, 768), (768, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, 768), (768, 1))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, 768), (768, 1))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (3072, 768), (768, 1))
    assert_size_stride(arg65_1, (3072, ), (1, ))
    assert_size_stride(arg66_1, (768, 3072), (3072, 1))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, 768), (768, 1))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, 768), (768, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, 768), (768, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, 768), (768, 1))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (3072, 768), (768, 1))
    assert_size_stride(arg81_1, (3072, ), (1, ))
    assert_size_stride(arg82_1, (768, 3072), (3072, 1))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, 768), (768, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, 768), (768, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, 768), (768, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, 768), (768, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (3072, 768), (768, 1))
    assert_size_stride(arg97_1, (3072, ), (1, ))
    assert_size_stride(arg98_1, (768, 3072), (3072, 1))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (1026, 768), (768, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, 768), (768, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, 768), (768, 1))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, 768), (768, 1))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, 768), (768, 1))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, 768), (768, 1))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, 768), (768, 1))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, 768), (768, 1))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, 768), (768, 1))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (3072, 768), (768, 1))
    assert_size_stride(arg126_1, (3072, ), (1, ))
    assert_size_stride(arg127_1, (768, 3072), (3072, 1))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, 768), (768, 1))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, 768), (768, 1))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, 768), (768, 1))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, 768), (768, 1))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, 768), (768, 1))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, 768), (768, 1))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (768, 768), (768, 1))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, 768), (768, 1))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (3072, 768), (768, 1))
    assert_size_stride(arg152_1, (3072, ), (1, ))
    assert_size_stride(arg153_1, (768, 3072), (3072, 1))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, 768), (768, 1))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (768, 768), (768, 1))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, 768), (768, 1))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, 768), (768, 1))
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
    assert_size_stride(arg193_1, (768, 768), (768, 1))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (768, 768), (768, 1))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, 768), (768, 1))
    assert_size_stride(arg198_1, (768, ), (1, ))
    assert_size_stride(arg199_1, (768, 768), (768, 1))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (768, ), (1, ))
    assert_size_stride(arg203_1, (3072, 768), (768, 1))
    assert_size_stride(arg204_1, (3072, ), (1, ))
    assert_size_stride(arg205_1, (768, 3072), (3072, 1))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (768, ), (1, ))
    assert_size_stride(arg209_1, (768, 768), (768, 1))
    assert_size_stride(arg210_1, (768, ), (1, ))
    assert_size_stride(arg211_1, (768, 768), (768, 1))
    assert_size_stride(arg212_1, (768, ), (1, ))
    assert_size_stride(arg213_1, (768, 768), (768, 1))
    assert_size_stride(arg214_1, (768, ), (1, ))
    assert_size_stride(arg215_1, (768, 768), (768, 1))
    assert_size_stride(arg216_1, (768, ), (1, ))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (768, ), (1, ))
    assert_size_stride(arg219_1, (768, 768), (768, 1))
    assert_size_stride(arg220_1, (768, ), (1, ))
    assert_size_stride(arg221_1, (768, 768), (768, 1))
    assert_size_stride(arg222_1, (768, ), (1, ))
    assert_size_stride(arg223_1, (768, 768), (768, 1))
    assert_size_stride(arg224_1, (768, ), (1, ))
    assert_size_stride(arg225_1, (768, 768), (768, 1))
    assert_size_stride(arg226_1, (768, ), (1, ))
    assert_size_stride(arg227_1, (768, ), (1, ))
    assert_size_stride(arg228_1, (768, ), (1, ))
    assert_size_stride(arg229_1, (3072, 768), (768, 1))
    assert_size_stride(arg230_1, (3072, ), (1, ))
    assert_size_stride(arg231_1, (768, 3072), (3072, 1))
    assert_size_stride(arg232_1, (768, ), (1, ))
    assert_size_stride(arg233_1, (768, ), (1, ))
    assert_size_stride(arg234_1, (768, ), (1, ))
    assert_size_stride(arg235_1, (768, 768), (768, 1))
    assert_size_stride(arg236_1, (768, ), (1, ))
    assert_size_stride(arg237_1, (768, 768), (768, 1))
    assert_size_stride(arg238_1, (768, ), (1, ))
    assert_size_stride(arg239_1, (768, 768), (768, 1))
    assert_size_stride(arg240_1, (768, ), (1, ))
    assert_size_stride(arg241_1, (768, 768), (768, 1))
    assert_size_stride(arg242_1, (768, ), (1, ))
    assert_size_stride(arg243_1, (768, ), (1, ))
    assert_size_stride(arg244_1, (768, ), (1, ))
    assert_size_stride(arg245_1, (768, 768), (768, 1))
    assert_size_stride(arg246_1, (768, ), (1, ))
    assert_size_stride(arg247_1, (768, 768), (768, 1))
    assert_size_stride(arg248_1, (768, ), (1, ))
    assert_size_stride(arg249_1, (768, 768), (768, 1))
    assert_size_stride(arg250_1, (768, ), (1, ))
    assert_size_stride(arg251_1, (768, 768), (768, 1))
    assert_size_stride(arg252_1, (768, ), (1, ))
    assert_size_stride(arg253_1, (768, ), (1, ))
    assert_size_stride(arg254_1, (768, ), (1, ))
    assert_size_stride(arg255_1, (3072, 768), (768, 1))
    assert_size_stride(arg256_1, (3072, ), (1, ))
    assert_size_stride(arg257_1, (768, 3072), (3072, 1))
    assert_size_stride(arg258_1, (768, ), (1, ))
    assert_size_stride(arg259_1, (768, ), (1, ))
    assert_size_stride(arg260_1, (768, ), (1, ))
    assert_size_stride(arg261_1, (1, 50005), (50005, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((4, 1024, 768), (786432, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, add, embed_pos, hidden_states, hidden_states_1], Original ATen: [aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, buf3, 4096, 768, grid=grid(4096), stream=stream0)
        del arg1_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((4096, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (4096, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 768), (1, 768), 0), out=buf4)
        del arg6_1
        buf5 = empty_strided_cuda((4096, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (4096, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0), out=buf5)
        del arg8_1
        buf6 = empty_strided_cuda((4096, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (4096, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 768), (1, 768), 0), out=buf6)
        del arg10_1
        buf7 = empty_strided_cuda((1, 48, 1024, 64), (3145728, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf4, arg7_1, buf7, 3145728, grid=grid(3145728), stream=stream0)
        del arg7_1
        buf8 = reinterpret_tensor(buf4, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf5, arg9_1, buf8, 3145728, grid=grid(3145728), stream=stream0)
        del arg9_1
        buf9 = reinterpret_tensor(buf5, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf6, arg11_1, buf9, 3145728, grid=grid(3145728), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf7, buf8, buf9, None, False, scale=1.0)
        buf11 = buf10[0]
        del buf10
        buf15 = reinterpret_tensor(buf9, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf11, buf15, 3145728, grid=grid(3145728), stream=stream0)
        buf16 = reinterpret_tensor(buf11, (4096, 768), (768, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (4096, 768), (768, 1), 0), reinterpret_tensor(arg12_1, (768, 768), (1, 768), 0), out=buf16)
        del arg12_1
        buf20 = reinterpret_tensor(buf15, (4, 1024, 768), (786432, 768, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf3, buf16, arg13_1, arg14_1, arg15_1, buf20, 4096, 768, grid=grid(4096), stream=stream0)
        del arg13_1
        del arg14_1
        del arg15_1
        buf21 = empty_strided_cuda((4096, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (4096, 768), (768, 1), 0), reinterpret_tensor(arg16_1, (768, 3072), (1, 768), 0), out=buf21)
        del arg16_1
        buf22 = reinterpret_tensor(buf21, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf22, arg17_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg17_1
        buf23 = reinterpret_tensor(buf3, (4096, 768), (768, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg18_1, (3072, 768), (1, 3072), 0), out=buf23)
        del arg18_1
        buf27 = reinterpret_tensor(buf16, (4, 1024, 768), (786432, 768, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf20, buf23, arg19_1, arg20_1, arg21_1, buf27, 4096, 768, grid=grid(4096), stream=stream0)
        del arg19_1
        del arg20_1
        del arg21_1
        buf28 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (4096, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 768), (1, 768), 0), out=buf28)
        del arg22_1
        buf29 = reinterpret_tensor(buf20, (4096, 768), (768, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (4096, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 768), (1, 768), 0), out=buf29)
        del arg24_1
        buf30 = reinterpret_tensor(buf8, (4096, 768), (768, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (4096, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 768), (1, 768), 0), out=buf30)
        del arg26_1
        buf31 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf28, arg23_1, buf31, 3145728, grid=grid(3145728), stream=stream0)
        del arg23_1
        buf32 = reinterpret_tensor(buf28, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf29, arg25_1, buf32, 3145728, grid=grid(3145728), stream=stream0)
        del arg25_1
        buf33 = reinterpret_tensor(buf29, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf30, arg27_1, buf33, 3145728, grid=grid(3145728), stream=stream0)
        del arg27_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf34 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf31, buf32, buf33, None, False, scale=1.0)
        buf35 = buf34[0]
        del buf34
        buf39 = reinterpret_tensor(buf33, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf35, buf39, 3145728, grid=grid(3145728), stream=stream0)
        buf40 = reinterpret_tensor(buf35, (4096, 768), (768, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (4096, 768), (768, 1), 0), reinterpret_tensor(arg28_1, (768, 768), (1, 768), 0), out=buf40)
        del arg28_1
        buf44 = reinterpret_tensor(buf39, (4, 1024, 768), (786432, 768, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf27, buf40, arg29_1, arg30_1, arg31_1, buf44, 4096, 768, grid=grid(4096), stream=stream0)
        del arg29_1
        del arg30_1
        del arg31_1
        buf45 = reinterpret_tensor(buf22, (4096, 3072), (3072, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (4096, 768), (768, 1), 0), reinterpret_tensor(arg32_1, (768, 3072), (1, 768), 0), out=buf45)
        del arg32_1
        buf46 = reinterpret_tensor(buf45, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf46, arg33_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg33_1
        buf47 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg34_1, (3072, 768), (1, 3072), 0), out=buf47)
        del arg34_1
        buf51 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf44, buf47, arg35_1, arg36_1, arg37_1, buf51, 4096, 768, grid=grid(4096), stream=stream0)
        del arg35_1
        del arg36_1
        del arg37_1
        buf52 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (4096, 768), (768, 1), 0), reinterpret_tensor(arg38_1, (768, 768), (1, 768), 0), out=buf52)
        del arg38_1
        buf53 = reinterpret_tensor(buf44, (4096, 768), (768, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (4096, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 768), (1, 768), 0), out=buf53)
        del arg40_1
        buf54 = reinterpret_tensor(buf32, (4096, 768), (768, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (4096, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 768), (1, 768), 0), out=buf54)
        del arg42_1
        buf55 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf52, arg39_1, buf55, 3145728, grid=grid(3145728), stream=stream0)
        del arg39_1
        buf56 = reinterpret_tensor(buf52, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf53, arg41_1, buf56, 3145728, grid=grid(3145728), stream=stream0)
        del arg41_1
        buf57 = reinterpret_tensor(buf53, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf54, arg43_1, buf57, 3145728, grid=grid(3145728), stream=stream0)
        del arg43_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf58 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf55, buf56, buf57, None, False, scale=1.0)
        buf59 = buf58[0]
        del buf58
        buf63 = reinterpret_tensor(buf57, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf59, buf63, 3145728, grid=grid(3145728), stream=stream0)
        buf64 = reinterpret_tensor(buf59, (4096, 768), (768, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (4096, 768), (768, 1), 0), reinterpret_tensor(arg44_1, (768, 768), (1, 768), 0), out=buf64)
        del arg44_1
        buf68 = reinterpret_tensor(buf63, (4, 1024, 768), (786432, 768, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_22, hidden_states_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf51, buf64, arg45_1, arg46_1, arg47_1, buf68, 4096, 768, grid=grid(4096), stream=stream0)
        del arg45_1
        del arg46_1
        del arg47_1
        buf69 = reinterpret_tensor(buf46, (4096, 3072), (3072, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (4096, 768), (768, 1), 0), reinterpret_tensor(arg48_1, (768, 3072), (1, 768), 0), out=buf69)
        del arg48_1
        buf70 = reinterpret_tensor(buf69, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf70, arg49_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg49_1
        buf71 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg50_1, (3072, 768), (1, 3072), 0), out=buf71)
        del arg50_1
        buf75 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf68, buf71, arg51_1, arg52_1, arg53_1, buf75, 4096, 768, grid=grid(4096), stream=stream0)
        del arg51_1
        del arg52_1
        del arg53_1
        buf76 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (4096, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 768), (1, 768), 0), out=buf76)
        del arg54_1
        buf77 = reinterpret_tensor(buf68, (4096, 768), (768, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (4096, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), out=buf77)
        del arg56_1
        buf78 = reinterpret_tensor(buf56, (4096, 768), (768, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (4096, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 768), (1, 768), 0), out=buf78)
        del arg58_1
        buf79 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf76, arg55_1, buf79, 3145728, grid=grid(3145728), stream=stream0)
        del arg55_1
        buf80 = reinterpret_tensor(buf76, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf77, arg57_1, buf80, 3145728, grid=grid(3145728), stream=stream0)
        del arg57_1
        buf81 = reinterpret_tensor(buf77, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf78, arg59_1, buf81, 3145728, grid=grid(3145728), stream=stream0)
        del arg59_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf82 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf79, buf80, buf81, None, False, scale=1.0)
        buf83 = buf82[0]
        del buf82
        buf87 = reinterpret_tensor(buf81, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf83, buf87, 3145728, grid=grid(3145728), stream=stream0)
        buf88 = reinterpret_tensor(buf83, (4096, 768), (768, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (4096, 768), (768, 1), 0), reinterpret_tensor(arg60_1, (768, 768), (1, 768), 0), out=buf88)
        del arg60_1
        buf92 = reinterpret_tensor(buf87, (4, 1024, 768), (786432, 768, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf75, buf88, arg61_1, arg62_1, arg63_1, buf92, 4096, 768, grid=grid(4096), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        buf93 = reinterpret_tensor(buf70, (4096, 3072), (3072, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (4096, 768), (768, 1), 0), reinterpret_tensor(arg64_1, (768, 3072), (1, 768), 0), out=buf93)
        del arg64_1
        buf94 = reinterpret_tensor(buf93, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_33], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf94, arg65_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg65_1
        buf95 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg66_1, (3072, 768), (1, 3072), 0), out=buf95)
        del arg66_1
        buf99 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf92, buf95, arg67_1, arg68_1, arg69_1, buf99, 4096, 768, grid=grid(4096), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        buf100 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (4096, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 768), (1, 768), 0), out=buf100)
        del arg70_1
        buf101 = reinterpret_tensor(buf92, (4096, 768), (768, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (4096, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 768), (1, 768), 0), out=buf101)
        del arg72_1
        buf102 = reinterpret_tensor(buf80, (4096, 768), (768, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (4096, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), out=buf102)
        del arg74_1
        buf103 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf100, arg71_1, buf103, 3145728, grid=grid(3145728), stream=stream0)
        del arg71_1
        buf104 = reinterpret_tensor(buf100, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf101, arg73_1, buf104, 3145728, grid=grid(3145728), stream=stream0)
        del arg73_1
        buf105 = reinterpret_tensor(buf101, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf102, arg75_1, buf105, 3145728, grid=grid(3145728), stream=stream0)
        del arg75_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf106 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf103, buf104, buf105, None, False, scale=1.0)
        buf107 = buf106[0]
        del buf106
        buf111 = reinterpret_tensor(buf105, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf107, buf111, 3145728, grid=grid(3145728), stream=stream0)
        buf112 = reinterpret_tensor(buf107, (4096, 768), (768, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (4096, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 768), (1, 768), 0), out=buf112)
        del arg76_1
        buf116 = reinterpret_tensor(buf111, (4, 1024, 768), (786432, 768, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf99, buf112, arg77_1, arg78_1, arg79_1, buf116, 4096, 768, grid=grid(4096), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        buf117 = reinterpret_tensor(buf94, (4096, 3072), (3072, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (4096, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 3072), (1, 768), 0), out=buf117)
        del arg80_1
        buf118 = reinterpret_tensor(buf117, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf118, arg81_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg81_1
        buf119 = reinterpret_tensor(buf99, (4096, 768), (768, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg82_1, (3072, 768), (1, 3072), 0), out=buf119)
        del arg82_1
        buf123 = reinterpret_tensor(buf112, (4, 1024, 768), (786432, 768, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf116, buf119, arg83_1, arg84_1, arg85_1, buf123, 4096, 768, grid=grid(4096), stream=stream0)
        del arg83_1
        del arg84_1
        del arg85_1
        buf124 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (4096, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), out=buf124)
        del arg86_1
        buf125 = reinterpret_tensor(buf116, (4096, 768), (768, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (4096, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), out=buf125)
        del arg88_1
        buf126 = reinterpret_tensor(buf104, (4096, 768), (768, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (4096, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), out=buf126)
        del arg90_1
        buf127 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf124, arg87_1, buf127, 3145728, grid=grid(3145728), stream=stream0)
        del arg87_1
        buf128 = reinterpret_tensor(buf124, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf125, arg89_1, buf128, 3145728, grid=grid(3145728), stream=stream0)
        del arg89_1
        buf129 = reinterpret_tensor(buf125, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf126, arg91_1, buf129, 3145728, grid=grid(3145728), stream=stream0)
        del arg91_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf130 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf127, buf128, buf129, None, False, scale=1.0)
        buf131 = buf130[0]
        del buf130
        buf135 = reinterpret_tensor(buf129, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf131, buf135, 3145728, grid=grid(3145728), stream=stream0)
        buf136 = reinterpret_tensor(buf131, (4096, 768), (768, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (4096, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 768), (1, 768), 0), out=buf136)
        del arg92_1
        buf140 = reinterpret_tensor(buf135, (4, 1024, 768), (786432, 768, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf123, buf136, arg93_1, arg94_1, arg95_1, buf140, 4096, 768, grid=grid(4096), stream=stream0)
        del arg93_1
        del arg94_1
        del arg95_1
        buf141 = reinterpret_tensor(buf118, (4096, 3072), (3072, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (4096, 768), (768, 1), 0), reinterpret_tensor(arg96_1, (768, 3072), (1, 768), 0), out=buf141)
        del arg96_1
        buf142 = reinterpret_tensor(buf141, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf142, arg97_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg97_1
        buf143 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg98_1, (3072, 768), (1, 3072), 0), out=buf143)
        del arg98_1
        buf171 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf140, buf143, arg99_1, arg100_1, arg101_1, buf171, 4096, 768, grid=grid(4096), stream=stream0)
        del arg100_1
        del arg101_1
        del arg99_1
        buf147 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [eq, masked_fill_, ne, sum_1], Original ATen: [aten.eq, aten.masked_fill, aten.ne, aten.sum]
        triton_per_fused_eq_masked_fill_ne_sum_6.run(arg0_1, buf147, 4, 1024, grid=grid(4), stream=stream0)
        buf148 = reinterpret_tensor(buf143, (4, 1024, 768), (786432, 768, 1), 0); del buf143  # reuse
        buf152 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [eq, masked_fill_, clone_1, setitem, setitem_1, embedding_2, inputs_embeds_1, add_15, positions_2, hidden_states_57, hidden_states_58], Original ATen: [aten.eq, aten.masked_fill, aten.clone, aten.copy, aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_7.run(buf147, arg0_1, arg2_1, arg102_1, arg103_1, arg104_1, buf148, buf152, 4096, 768, grid=grid(4096), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del buf147
        buf153 = reinterpret_tensor(buf148, (4096, 768), (768, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf152, (4096, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), out=buf153)
        del arg105_1
        buf154 = reinterpret_tensor(buf128, (4096, 768), (768, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf152, (4096, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 768), (1, 768), 0), out=buf154)
        del arg107_1
        buf155 = reinterpret_tensor(buf127, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [key_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf154, arg108_1, buf155, 3145728, grid=grid(3145728), stream=stream0)
        del arg108_1
        buf156 = reinterpret_tensor(buf154, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf153, arg106_1, buf156, 3145728, grid=grid(3145728), stream=stream0)
        del arg106_1
        buf157 = empty_strided_cuda((48, 1024, 1024), (1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf156, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf155, (48, 64, 1024), (65536, 1, 64), 0), out=buf157)
        buf162 = empty_strided_cuda((48, 1024, 1024), (1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_weights_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf157, buf162, 49152, 1024, grid=grid(49152), stream=stream0)
        buf160 = reinterpret_tensor(buf156, (4096, 768), (768, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf152, (4096, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 768), (1, 768), 0), out=buf160)
        del arg109_1
        buf161 = reinterpret_tensor(buf153, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [value_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf160, arg110_1, buf161, 3145728, grid=grid(3145728), stream=stream0)
        del arg110_1
        buf163 = reinterpret_tensor(buf160, (48, 1024, 64), (65536, 64, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_15, attn_output_30], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf162, reinterpret_tensor(buf161, (48, 1024, 64), (65536, 64, 1), 0), out=buf163)
        buf164 = reinterpret_tensor(buf126, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf163, buf164, 3145728, grid=grid(3145728), stream=stream0)
        buf165 = reinterpret_tensor(buf163, (4096, 768), (768, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (4096, 768), (768, 1), 0), reinterpret_tensor(arg111_1, (768, 768), (1, 768), 0), out=buf165)
        del arg111_1
        buf169 = reinterpret_tensor(buf164, (4, 1024, 768), (786432, 768, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf152, buf165, arg112_1, arg113_1, arg114_1, buf169, 4096, 768, grid=grid(4096), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        buf170 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (4096, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 768), (1, 768), 0), out=buf170)
        del arg115_1
        buf172 = reinterpret_tensor(buf152, (4096, 768), (768, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 768), (1, 768), 0), out=buf172)
        del arg117_1
        buf173 = reinterpret_tensor(buf102, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [key_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf172, arg118_1, buf173, 3145728, grid=grid(3145728), stream=stream0)
        del arg118_1
        buf174 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 768), (768, 1), 0), reinterpret_tensor(arg119_1, (768, 768), (1, 768), 0), out=buf174)
        del arg119_1
        buf175 = reinterpret_tensor(buf78, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [value_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf174, arg120_1, buf175, 3145728, grid=grid(3145728), stream=stream0)
        del arg120_1
        buf176 = reinterpret_tensor(buf174, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf170, arg116_1, buf176, 3145728, grid=grid(3145728), stream=stream0)
        del arg116_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf177 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf176, reinterpret_tensor(buf173, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0), reinterpret_tensor(buf175, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0), None, False, scale=1.0)
        buf178 = buf177[0]
        del buf177
        buf182 = reinterpret_tensor(buf176, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf178, buf182, 3145728, grid=grid(3145728), stream=stream0)
        buf183 = reinterpret_tensor(buf178, (4096, 768), (768, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (4096, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 768), (1, 768), 0), out=buf183)
        del arg121_1
        buf187 = reinterpret_tensor(buf182, (4, 1024, 768), (786432, 768, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf169, buf183, arg122_1, arg123_1, arg124_1, buf187, 4096, 768, grid=grid(4096), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        buf188 = reinterpret_tensor(buf142, (4096, 3072), (3072, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (4096, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 3072), (1, 768), 0), out=buf188)
        del arg125_1
        buf189 = reinterpret_tensor(buf188, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_66], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf189, arg126_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg126_1
        buf190 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg127_1, (3072, 768), (1, 3072), 0), out=buf190)
        del arg127_1
        buf194 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_70, hidden_states_71], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf187, buf190, arg128_1, arg129_1, arg130_1, buf194, 4096, 768, grid=grid(4096), stream=stream0)
        del arg128_1
        del arg129_1
        del arg130_1
        buf195 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (4096, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 768), (1, 768), 0), out=buf195)
        del arg131_1
        buf196 = reinterpret_tensor(buf187, (4096, 768), (768, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (4096, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 768), (1, 768), 0), out=buf196)
        del arg133_1
        buf197 = reinterpret_tensor(buf170, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf196, arg134_1, buf197, 3145728, grid=grid(3145728), stream=stream0)
        del arg134_1
        buf198 = reinterpret_tensor(buf196, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf195, arg132_1, buf198, 3145728, grid=grid(3145728), stream=stream0)
        del arg132_1
        buf199 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf197, (48, 64, 1024), (65536, 1, 64), 0), out=buf199)
        buf204 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_21], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf199, buf204, 49152, 1024, grid=grid(49152), stream=stream0)
        buf202 = reinterpret_tensor(buf198, (4096, 768), (768, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (4096, 768), (768, 1), 0), reinterpret_tensor(arg135_1, (768, 768), (1, 768), 0), out=buf202)
        del arg135_1
        buf203 = reinterpret_tensor(buf195, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf202, arg136_1, buf203, 3145728, grid=grid(3145728), stream=stream0)
        del arg136_1
        buf205 = reinterpret_tensor(buf202, (48, 1024, 64), (65536, 64, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_21, attn_output_40], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf204, reinterpret_tensor(buf203, (48, 1024, 64), (65536, 64, 1), 0), out=buf205)
        buf206 = reinterpret_tensor(buf54, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf205, buf206, 3145728, grid=grid(3145728), stream=stream0)
        buf207 = reinterpret_tensor(buf205, (4096, 768), (768, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (4096, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), out=buf207)
        del arg137_1
        buf211 = reinterpret_tensor(buf206, (4, 1024, 768), (786432, 768, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf194, buf207, arg138_1, arg139_1, arg140_1, buf211, 4096, 768, grid=grid(4096), stream=stream0)
        del arg138_1
        del arg139_1
        del arg140_1
        buf212 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (4096, 768), (768, 1), 0), reinterpret_tensor(arg141_1, (768, 768), (1, 768), 0), out=buf212)
        del arg141_1
        buf213 = reinterpret_tensor(buf194, (4096, 768), (768, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 768), (1, 768), 0), out=buf213)
        del arg143_1
        buf214 = reinterpret_tensor(buf30, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [key_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf213, arg144_1, buf214, 3145728, grid=grid(3145728), stream=stream0)
        del arg144_1
        buf215 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 768), (768, 1), 0), reinterpret_tensor(arg145_1, (768, 768), (1, 768), 0), out=buf215)
        del arg145_1
        buf216 = reinterpret_tensor(buf6, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [value_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf215, arg146_1, buf216, 3145728, grid=grid(3145728), stream=stream0)
        del arg146_1
        buf217 = reinterpret_tensor(buf215, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf212, arg142_1, buf217, 3145728, grid=grid(3145728), stream=stream0)
        del arg142_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf218 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf217, reinterpret_tensor(buf214, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0), reinterpret_tensor(buf216, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0), None, False, scale=1.0)
        buf219 = buf218[0]
        del buf218
        buf223 = reinterpret_tensor(buf217, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf219, buf223, 3145728, grid=grid(3145728), stream=stream0)
        buf224 = reinterpret_tensor(buf219, (4096, 768), (768, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (4096, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 768), (1, 768), 0), out=buf224)
        del arg147_1
        buf228 = reinterpret_tensor(buf223, (4, 1024, 768), (786432, 768, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76, hidden_states_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf211, buf224, arg148_1, arg149_1, arg150_1, buf228, 4096, 768, grid=grid(4096), stream=stream0)
        del arg148_1
        del arg149_1
        del arg150_1
        buf229 = reinterpret_tensor(buf189, (4096, 3072), (3072, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (4096, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 3072), (1, 768), 0), out=buf229)
        del arg151_1
        buf230 = reinterpret_tensor(buf229, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf230, arg152_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg152_1
        buf231 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg153_1, (3072, 768), (1, 3072), 0), out=buf231)
        del arg153_1
        buf235 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82, hidden_states_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf228, buf231, arg154_1, arg155_1, arg156_1, buf235, 4096, 768, grid=grid(4096), stream=stream0)
        del arg154_1
        del arg155_1
        del arg156_1
        buf236 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (4096, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 768), (1, 768), 0), out=buf236)
        del arg157_1
        buf237 = reinterpret_tensor(buf228, (4096, 768), (768, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (4096, 768), (768, 1), 0), reinterpret_tensor(arg159_1, (768, 768), (1, 768), 0), out=buf237)
        del arg159_1
        buf238 = reinterpret_tensor(buf212, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf237, arg160_1, buf238, 3145728, grid=grid(3145728), stream=stream0)
        del arg160_1
        buf239 = reinterpret_tensor(buf237, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf236, arg158_1, buf239, 3145728, grid=grid(3145728), stream=stream0)
        del arg158_1
        buf240 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf239, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf238, (48, 64, 1024), (65536, 1, 64), 0), out=buf240)
        buf245 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf240, buf245, 49152, 1024, grid=grid(49152), stream=stream0)
        buf243 = reinterpret_tensor(buf239, (4096, 768), (768, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (4096, 768), (768, 1), 0), reinterpret_tensor(arg161_1, (768, 768), (1, 768), 0), out=buf243)
        del arg161_1
        buf244 = reinterpret_tensor(buf236, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf243, arg162_1, buf244, 3145728, grid=grid(3145728), stream=stream0)
        del arg162_1
        buf246 = reinterpret_tensor(buf243, (48, 1024, 64), (65536, 64, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_27, attn_output_50], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf245, reinterpret_tensor(buf244, (48, 1024, 64), (65536, 64, 1), 0), out=buf246)
        buf247 = empty_strided_cuda((4, 1024, 12, 64), (786432, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf246, buf247, 3145728, grid=grid(3145728), stream=stream0)
        buf248 = reinterpret_tensor(buf246, (4096, 768), (768, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (4096, 768), (768, 1), 0), reinterpret_tensor(arg163_1, (768, 768), (1, 768), 0), out=buf248)
        del arg163_1
        buf252 = reinterpret_tensor(buf247, (4, 1024, 768), (786432, 768, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf235, buf248, arg164_1, arg165_1, arg166_1, buf252, 4096, 768, grid=grid(4096), stream=stream0)
        del arg164_1
        del arg165_1
        del arg166_1
        buf253 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf252, (4096, 768), (768, 1), 0), reinterpret_tensor(arg167_1, (768, 768), (1, 768), 0), out=buf253)
        del arg167_1
        buf254 = reinterpret_tensor(buf235, (4096, 768), (768, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 768), (1, 768), 0), out=buf254)
        del arg169_1
        buf255 = empty_strided_cuda((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf254, arg170_1, buf255, 3145728, grid=grid(3145728), stream=stream0)
        del arg170_1
        buf256 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 768), (1, 768), 0), out=buf256)
        del arg171_1
        buf257 = empty_strided_cuda((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [value_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf256, arg172_1, buf257, 3145728, grid=grid(3145728), stream=stream0)
        del arg172_1
        buf258 = reinterpret_tensor(buf256, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf253, arg168_1, buf258, 3145728, grid=grid(3145728), stream=stream0)
        del arg168_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf259 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf258, reinterpret_tensor(buf255, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0), reinterpret_tensor(buf257, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0), None, False, scale=1.0)
        buf260 = buf259[0]
        del buf259
        buf264 = reinterpret_tensor(buf258, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf260, buf264, 3145728, grid=grid(3145728), stream=stream0)
        buf265 = reinterpret_tensor(buf260, (4096, 768), (768, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf264, (4096, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 768), (1, 768), 0), out=buf265)
        del arg173_1
        buf269 = reinterpret_tensor(buf264, (4, 1024, 768), (786432, 768, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_89], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf252, buf265, arg174_1, arg175_1, arg176_1, buf269, 4096, 768, grid=grid(4096), stream=stream0)
        del arg174_1
        del arg175_1
        del arg176_1
        buf270 = reinterpret_tensor(buf230, (4096, 3072), (3072, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (4096, 768), (768, 1), 0), reinterpret_tensor(arg177_1, (768, 3072), (1, 768), 0), out=buf270)
        del arg177_1
        buf271 = reinterpret_tensor(buf270, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_90], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf271, arg178_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg178_1
        buf272 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg179_1, (3072, 768), (1, 3072), 0), out=buf272)
        del arg179_1
        buf276 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf269, buf272, arg180_1, arg181_1, arg182_1, buf276, 4096, 768, grid=grid(4096), stream=stream0)
        del arg180_1
        del arg181_1
        del arg182_1
        buf277 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (4096, 768), (768, 1), 0), reinterpret_tensor(arg183_1, (768, 768), (1, 768), 0), out=buf277)
        del arg183_1
        buf278 = reinterpret_tensor(buf269, (4096, 768), (768, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (4096, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), out=buf278)
        del arg185_1
        buf279 = reinterpret_tensor(buf253, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf278, arg186_1, buf279, 3145728, grid=grid(3145728), stream=stream0)
        del arg186_1
        buf280 = reinterpret_tensor(buf278, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf277, arg184_1, buf280, 3145728, grid=grid(3145728), stream=stream0)
        del arg184_1
        buf281 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf280, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf279, (48, 64, 1024), (65536, 1, 64), 0), out=buf281)
        buf286 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_33], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf281, buf286, 49152, 1024, grid=grid(49152), stream=stream0)
        buf284 = reinterpret_tensor(buf280, (4096, 768), (768, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (4096, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 768), (1, 768), 0), out=buf284)
        del arg187_1
        buf285 = reinterpret_tensor(buf277, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf284, arg188_1, buf285, 3145728, grid=grid(3145728), stream=stream0)
        del arg188_1
        buf287 = reinterpret_tensor(buf284, (48, 1024, 64), (65536, 64, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_33, attn_output_60], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf286, reinterpret_tensor(buf285, (48, 1024, 64), (65536, 64, 1), 0), out=buf287)
        buf288 = empty_strided_cuda((4, 1024, 12, 64), (786432, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf287, buf288, 3145728, grid=grid(3145728), stream=stream0)
        buf289 = reinterpret_tensor(buf287, (4096, 768), (768, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf288, (4096, 768), (768, 1), 0), reinterpret_tensor(arg189_1, (768, 768), (1, 768), 0), out=buf289)
        del arg189_1
        buf293 = reinterpret_tensor(buf288, (4, 1024, 768), (786432, 768, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_97, hidden_states_98], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf276, buf289, arg190_1, arg191_1, arg192_1, buf293, 4096, 768, grid=grid(4096), stream=stream0)
        del arg190_1
        del arg191_1
        del arg192_1
        buf294 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf293, (4096, 768), (768, 1), 0), reinterpret_tensor(arg193_1, (768, 768), (1, 768), 0), out=buf294)
        del arg193_1
        buf295 = reinterpret_tensor(buf276, (4096, 768), (768, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 768), (768, 1), 0), reinterpret_tensor(arg195_1, (768, 768), (1, 768), 0), out=buf295)
        del arg195_1
        buf296 = empty_strided_cuda((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf295, arg196_1, buf296, 3145728, grid=grid(3145728), stream=stream0)
        del arg196_1
        buf297 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 768), (768, 1), 0), reinterpret_tensor(arg197_1, (768, 768), (1, 768), 0), out=buf297)
        del arg197_1
        buf298 = empty_strided_cuda((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [value_states_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf297, arg198_1, buf298, 3145728, grid=grid(3145728), stream=stream0)
        del arg198_1
        buf299 = reinterpret_tensor(buf297, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf294, arg194_1, buf299, 3145728, grid=grid(3145728), stream=stream0)
        del arg194_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf300 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf299, reinterpret_tensor(buf296, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0), reinterpret_tensor(buf298, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0), None, False, scale=1.0)
        buf301 = buf300[0]
        del buf300
        buf305 = reinterpret_tensor(buf299, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [attn_output_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf301, buf305, 3145728, grid=grid(3145728), stream=stream0)
        buf306 = reinterpret_tensor(buf301, (4096, 768), (768, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf305, (4096, 768), (768, 1), 0), reinterpret_tensor(arg199_1, (768, 768), (1, 768), 0), out=buf306)
        del arg199_1
        buf310 = reinterpret_tensor(buf305, (4, 1024, 768), (786432, 768, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf293, buf306, arg200_1, arg201_1, arg202_1, buf310, 4096, 768, grid=grid(4096), stream=stream0)
        del arg200_1
        del arg201_1
        del arg202_1
        buf311 = reinterpret_tensor(buf271, (4096, 3072), (3072, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (4096, 768), (768, 1), 0), reinterpret_tensor(arg203_1, (768, 3072), (1, 768), 0), out=buf311)
        del arg203_1
        buf312 = reinterpret_tensor(buf311, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_102], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf312, arg204_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg204_1
        buf313 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg205_1, (3072, 768), (1, 3072), 0), out=buf313)
        del arg205_1
        buf317 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_106, hidden_states_107], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf310, buf313, arg206_1, arg207_1, arg208_1, buf317, 4096, 768, grid=grid(4096), stream=stream0)
        del arg206_1
        del arg207_1
        del arg208_1
        buf318 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (4096, 768), (768, 1), 0), reinterpret_tensor(arg209_1, (768, 768), (1, 768), 0), out=buf318)
        del arg209_1
        buf319 = reinterpret_tensor(buf310, (4096, 768), (768, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (4096, 768), (768, 1), 0), reinterpret_tensor(arg211_1, (768, 768), (1, 768), 0), out=buf319)
        del arg211_1
        buf320 = reinterpret_tensor(buf294, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf319, arg212_1, buf320, 3145728, grid=grid(3145728), stream=stream0)
        del arg212_1
        buf321 = reinterpret_tensor(buf319, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf318, arg210_1, buf321, 3145728, grid=grid(3145728), stream=stream0)
        del arg210_1
        buf322 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf321, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf320, (48, 64, 1024), (65536, 1, 64), 0), out=buf322)
        buf327 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf322, buf327, 49152, 1024, grid=grid(49152), stream=stream0)
        buf325 = reinterpret_tensor(buf321, (4096, 768), (768, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (4096, 768), (768, 1), 0), reinterpret_tensor(arg213_1, (768, 768), (1, 768), 0), out=buf325)
        del arg213_1
        buf326 = reinterpret_tensor(buf318, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf325, arg214_1, buf326, 3145728, grid=grid(3145728), stream=stream0)
        del arg214_1
        buf328 = reinterpret_tensor(buf325, (48, 1024, 64), (65536, 64, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_39, attn_output_70], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf327, reinterpret_tensor(buf326, (48, 1024, 64), (65536, 64, 1), 0), out=buf328)
        buf329 = empty_strided_cuda((4, 1024, 12, 64), (786432, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf328, buf329, 3145728, grid=grid(3145728), stream=stream0)
        buf330 = reinterpret_tensor(buf328, (4096, 768), (768, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf329, (4096, 768), (768, 1), 0), reinterpret_tensor(arg215_1, (768, 768), (1, 768), 0), out=buf330)
        del arg215_1
        buf334 = reinterpret_tensor(buf329, (4, 1024, 768), (786432, 768, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_109, hidden_states_110], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf317, buf330, arg216_1, arg217_1, arg218_1, buf334, 4096, 768, grid=grid(4096), stream=stream0)
        del arg216_1
        del arg217_1
        del arg218_1
        buf335 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (4096, 768), (768, 1), 0), reinterpret_tensor(arg219_1, (768, 768), (1, 768), 0), out=buf335)
        del arg219_1
        buf336 = reinterpret_tensor(buf317, (4096, 768), (768, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 768), (768, 1), 0), reinterpret_tensor(arg221_1, (768, 768), (1, 768), 0), out=buf336)
        del arg221_1
        buf337 = empty_strided_cuda((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf336, arg222_1, buf337, 3145728, grid=grid(3145728), stream=stream0)
        del arg222_1
        buf338 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 768), (768, 1), 0), reinterpret_tensor(arg223_1, (768, 768), (1, 768), 0), out=buf338)
        del arg223_1
        buf339 = empty_strided_cuda((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [value_states_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf338, arg224_1, buf339, 3145728, grid=grid(3145728), stream=stream0)
        del arg224_1
        buf340 = reinterpret_tensor(buf338, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf335, arg220_1, buf340, 3145728, grid=grid(3145728), stream=stream0)
        del arg220_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf341 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf340, reinterpret_tensor(buf337, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0), reinterpret_tensor(buf339, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0), None, False, scale=1.0)
        buf342 = buf341[0]
        del buf341
        buf346 = reinterpret_tensor(buf340, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [attn_output_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf342, buf346, 3145728, grid=grid(3145728), stream=stream0)
        buf347 = reinterpret_tensor(buf342, (4096, 768), (768, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (4096, 768), (768, 1), 0), reinterpret_tensor(arg225_1, (768, 768), (1, 768), 0), out=buf347)
        del arg225_1
        buf351 = reinterpret_tensor(buf346, (4, 1024, 768), (786432, 768, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_112, hidden_states_113], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf334, buf347, arg226_1, arg227_1, arg228_1, buf351, 4096, 768, grid=grid(4096), stream=stream0)
        del arg226_1
        del arg227_1
        del arg228_1
        buf352 = reinterpret_tensor(buf312, (4096, 3072), (3072, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf351, (4096, 768), (768, 1), 0), reinterpret_tensor(arg229_1, (768, 3072), (1, 768), 0), out=buf352)
        del arg229_1
        buf353 = reinterpret_tensor(buf352, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_114], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf353, arg230_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg230_1
        buf354 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg231_1, (3072, 768), (1, 3072), 0), out=buf354)
        del arg231_1
        buf358 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_118, hidden_states_119], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf351, buf354, arg232_1, arg233_1, arg234_1, buf358, 4096, 768, grid=grid(4096), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        buf359 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (4096, 768), (768, 1), 0), reinterpret_tensor(arg235_1, (768, 768), (1, 768), 0), out=buf359)
        del arg235_1
        buf360 = reinterpret_tensor(buf351, (4096, 768), (768, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (4096, 768), (768, 1), 0), reinterpret_tensor(arg237_1, (768, 768), (1, 768), 0), out=buf360)
        del arg237_1
        buf361 = reinterpret_tensor(buf335, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf360, arg238_1, buf361, 3145728, grid=grid(3145728), stream=stream0)
        del arg238_1
        buf362 = reinterpret_tensor(buf360, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf359, arg236_1, buf362, 3145728, grid=grid(3145728), stream=stream0)
        del arg236_1
        buf363 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf361, (48, 64, 1024), (65536, 1, 64), 0), out=buf363)
        buf368 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_45], Original ATen: [aten._softmax]
        triton_per_fused__softmax_10.run(buf363, buf368, 49152, 1024, grid=grid(49152), stream=stream0)
        del buf363
        buf366 = reinterpret_tensor(buf362, (4096, 768), (768, 1), 0); del buf362  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (4096, 768), (768, 1), 0), reinterpret_tensor(arg239_1, (768, 768), (1, 768), 0), out=buf366)
        del arg239_1
        buf367 = reinterpret_tensor(buf359, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf366, arg240_1, buf367, 3145728, grid=grid(3145728), stream=stream0)
        del arg240_1
        buf369 = reinterpret_tensor(buf366, (48, 1024, 64), (65536, 64, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_45, attn_output_80], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf368, reinterpret_tensor(buf367, (48, 1024, 64), (65536, 64, 1), 0), out=buf369)
        del buf368
        buf370 = empty_strided_cuda((4, 1024, 12, 64), (786432, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf369, buf370, 3145728, grid=grid(3145728), stream=stream0)
        buf371 = reinterpret_tensor(buf369, (4096, 768), (768, 1), 0); del buf369  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf370, (4096, 768), (768, 1), 0), reinterpret_tensor(arg241_1, (768, 768), (1, 768), 0), out=buf371)
        del arg241_1
        buf375 = reinterpret_tensor(buf370, (4, 1024, 768), (786432, 768, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_121, hidden_states_122], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf358, buf371, arg242_1, arg243_1, arg244_1, buf375, 4096, 768, grid=grid(4096), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        buf376 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf375, (4096, 768), (768, 1), 0), reinterpret_tensor(arg245_1, (768, 768), (1, 768), 0), out=buf376)
        del arg245_1
        buf377 = reinterpret_tensor(buf358, (4096, 768), (768, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 768), (768, 1), 0), reinterpret_tensor(arg247_1, (768, 768), (1, 768), 0), out=buf377)
        del arg247_1
        buf378 = empty_strided_cuda((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf377, arg248_1, buf378, 3145728, grid=grid(3145728), stream=stream0)
        del arg248_1
        buf379 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 768), (768, 1), 0), reinterpret_tensor(arg249_1, (768, 768), (1, 768), 0), out=buf379)
        del arg249_1
        buf380 = empty_strided_cuda((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [value_states_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf379, arg250_1, buf380, 3145728, grid=grid(3145728), stream=stream0)
        del arg250_1
        buf381 = reinterpret_tensor(buf379, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf376, arg246_1, buf381, 3145728, grid=grid(3145728), stream=stream0)
        del arg246_1
        del buf376
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf382 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf381, reinterpret_tensor(buf378, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0), reinterpret_tensor(buf380, (1, 48, 1024, 64), (3145728, 65536, 64, 1), 0), None, False, scale=1.0)
        buf383 = buf382[0]
        del buf382
        buf387 = reinterpret_tensor(buf381, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [attn_output_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf383, buf387, 3145728, grid=grid(3145728), stream=stream0)
        buf388 = reinterpret_tensor(buf383, (4096, 768), (768, 1), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf387, (4096, 768), (768, 1), 0), reinterpret_tensor(arg251_1, (768, 768), (1, 768), 0), out=buf388)
        del arg251_1
        buf392 = reinterpret_tensor(buf387, (4, 1024, 768), (786432, 768, 1), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_124, hidden_states_125], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf375, buf388, arg252_1, arg253_1, arg254_1, buf392, 4096, 768, grid=grid(4096), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        buf393 = reinterpret_tensor(buf353, (4096, 3072), (3072, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (4096, 768), (768, 1), 0), reinterpret_tensor(arg255_1, (768, 3072), (1, 768), 0), out=buf393)
        del arg255_1
        buf394 = reinterpret_tensor(buf393, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_126], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf394, arg256_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg256_1
        buf395 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg257_1, (3072, 768), (1, 3072), 0), out=buf395)
        del arg257_1
        del buf394
        buf399 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_130, hidden_states_131], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf392, buf395, arg258_1, arg259_1, arg260_1, buf399, 4096, 768, grid=grid(4096), stream=stream0)
        del arg258_1
        del arg259_1
        del arg260_1
        del buf392
        del buf395
        buf400 = empty_strided_cuda((768, 50008), (50016, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_12.run(arg2_1, buf400, 38406144, grid=grid(38406144), stream=stream0)
        del arg2_1
        buf401 = empty_strided_cuda((4096, 50008), (50016, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf399, (4096, 768), (768, 1), 0), buf400, out=buf401)
        del buf399
        del buf400
        buf402 = empty_strided_cuda((4, 1024, 50005), (51205120, 50005, 1), torch.float32)
        buf403 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        buf404 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [lm_logits_1, masked_lm_loss], Original ATen: [aten.add, aten._log_softmax]
        triton_red_fused__log_softmax_add_13.run(buf401, arg261_1, buf402, buf403, buf404, 4096, 50005, grid=grid(4096), stream=stream0)
        del arg261_1
        del buf401
        buf405 = empty_strided_cuda((), (), torch.float32)
        buf407 = buf405; del buf405  # reuse
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_14.run(buf407, arg0_1, buf402, buf403, buf404, 1, 4096, grid=grid(1), stream=stream0)
        del arg0_1
        del buf403
        del buf404
    return (buf407, buf402, buf155, buf161, buf173, buf175, buf197, buf203, buf214, buf216, buf238, buf244, buf255, buf257, buf279, buf285, buf296, buf298, buf320, buf326, buf337, buf339, buf361, buf367, buf378, buf380, buf171, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((50005, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1026, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1026, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
    arg193_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1, 50005), (50005, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('PLBartForConditionalGeneration', benchmark_compiled_module)
