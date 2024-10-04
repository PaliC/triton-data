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


# kernel path: /tmp/torchinductor_sahanp/ys/cys4v4paqdfcjixhu2yrwwh4qlem2blldgk4gckmegqh3qq4fwzk.py
# Topologically Sorted Source Nodes: [embedding, inputs_embeds, positions, embed_pos, hidden_states, hidden_states_1], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embed_pos => embedding_1
#   embedding => embedding
#   hidden_states => add
#   hidden_states_1 => add_1, add_2, mul_1, mul_2, rsqrt, sub, var_mean
#   inputs_embeds => mul
#   positions => iota
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %view, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, 1.0), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %iota), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg4_1), kwargs = {})
#   %add_2 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg5_1), kwargs = {})
triton_red_fused_add_arange_embedding_mul_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_mul_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
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
        tmp9 = tl.load(in_ptr2 + (r2 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert((0 <= tmp4) & (tmp4 < 50265), "index out of bounds: 0 <= tmp4 < 50265")
        tmp6 = tl.load(in_ptr1 + (r2 + (512*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp23 = tl.load(in_ptr2 + (r2 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp16 = tmp0 + tmp15
        tmp17 = tmp0 < 0
        tmp18 = tl.where(tmp17, tmp16, tmp0)
        tl.device_assert((0 <= tmp18) & (tmp18 < 50265), "index out of bounds: 0 <= tmp18 < 50265")
        tmp20 = tl.load(in_ptr1 + (r2 + (512*tmp18)), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = 1.0
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tmp25 = tmp24 - tmp12
        tmp26 = 512.0
        tmp27 = tmp13 / tmp26
        tmp28 = 1e-05
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp33 = tmp31 * tmp32
        tmp35 = tmp33 + tmp34
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cp/ccpjnqugv3g5lbsghdhpc7neecbqj76mv6oen2lwtaqri7u6tsjy.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %_scaled_dot_product_efficient_attention_default_15 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%unsqueeze_default_45, %unsqueeze_default_46, %unsqueeze_default_47, None, False), kwargs = {scale: 1.0})
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
    x0 = xindex % 32
    x1 = (xindex // 32) % 128
    x2 = (xindex // 4096)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*(x2 % 16)) + (512*x1) + (65536*(x2 // 16))), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*(x2 % 16))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dp/cdpn2tf47i3xwqsbwhwn7on62xffaind3jmitixpqxrdxhunr5xr.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %_scaled_dot_product_efficient_attention_default_15 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%unsqueeze_default_45, %unsqueeze_default_46, %unsqueeze_default_47, None, False), kwargs = {scale: 1.0})
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
    x0 = xindex % 32
    x1 = (xindex // 32) % 128
    x2 = (xindex // 4096)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*(x2 % 16)) + (512*x1) + (65536*(x2 // 16))), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*(x2 % 16))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ba/cbahbinm53y3up25umpbprqox7ayfiloo5i6wyyqiatnplixahwl.py
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
    x0 = xindex % 512
    x1 = (xindex // 512) % 128
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x2) + (32768*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7p/c7pfryuriygsjsglaneew544ldjk4k6lfcvhoi2axph7b5krwkn4.py
# Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_5], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_4 => add_3
#   hidden_states_5 => add_4, add_5, mul_4, mul_5, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_16), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_3), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg14_1), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg15_1), kwargs = {})
triton_per_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_4', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 8192
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
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 512, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp4 - tmp12
    tmp19 = 512.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/m2/cm2gbmkufnkkblluvc6ribgkuynmdp4f3ky5asejtiloocwuogsc.py
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
    x0 = xindex % 2048
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


# kernel path: /tmp/torchinductor_sahanp/io/cio5uwrldbzsg6flr5alicbgmyvevbypyogfhygy66aidk3hjuh6.py
# Topologically Sorted Source Nodes: [embedding_2, inputs_embeds_1, inputs_embeds_2, positions_1, positions_2, hidden_states_75], Original ATen: [aten.embedding, aten.mul, aten.native_layer_norm, aten.arange, aten.add]
# Source node to ATen node mapping:
#   embedding_2 => embedding_2
#   hidden_states_75 => add_62
#   inputs_embeds_1 => mul_67
#   inputs_embeds_2 => add_60, add_61, mul_68, mul_69, rsqrt_17, sub_25, var_mean_17
#   positions_1 => iota_2
#   positions_2 => embedding_3
# Graph fragment:
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %view_161, 0), kwargs = {})
#   %mul_67 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding_2, 1.0), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_67, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_67, %getitem_35), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_34, 1e-05), kwargs = {})
#   %rsqrt_17 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_60,), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt_17), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_68, %arg135_1), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_69, %arg136_1), kwargs = {})
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %embedding_3 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg134_1, %iota_2), kwargs = {})
#   %add_62 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_61, %embedding_3), kwargs = {})
triton_red_fused_add_arange_embedding_mul_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_mul_native_layer_norm_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert((0 <= tmp4) & (tmp4 < 50265), "index out of bounds: 0 <= tmp4 < 50265")
        tmp6 = tl.load(in_ptr1 + (r1 + (512*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(rmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask, tmp10_weight_next, tmp10_weight)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    x2 = xindex % 128
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr4 + (r1 + (512*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full([XBLOCK, RBLOCK], 50265, tl.int32)
        tmp14 = tmp0 + tmp13
        tmp15 = tmp0 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp0)
        tl.device_assert((0 <= tmp16) & (tmp16 < 50265), "index out of bounds: 0 <= tmp16 < 50265")
        tmp18 = tl.load(in_ptr1 + (r1 + (512*tmp16)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = 1.0
        tmp20 = tmp18 * tmp19
        tmp21 = tmp20 - tmp10
        tmp22 = 512.0
        tmp23 = tmp11 / tmp22
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = libdevice.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tmp33 = tmp31 + tmp32
        tl.store(out_ptr2 + (r1 + (512*x0)), tmp33, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/om/com5nnegqq7dsj3crk2qhm4ttgsjl6u32zgb52fb4eluageifmft.py
# Topologically Sorted Source Nodes: [contiguous_26], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_26 => clone_68
# Graph fragment:
#   %clone_68 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_93,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 128
    x2 = (xindex // 4096) % 16
    x3 = (xindex // 65536)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (512*x1) + (65536*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wz/cwzaj7k7l3gminugcsucd3xnqivwcj2qwnn2oee653fmxkolblxn.py
# Topologically Sorted Source Nodes: [key_states_16], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   key_states_16 => clone_66
# Graph fragment:
#   %clone_66 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_90,), kwargs = {memory_format: torch.contiguous_format})
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
    x0 = xindex % 32
    x1 = (xindex // 32) % 128
    x2 = (xindex // 4096) % 16
    x3 = (xindex // 65536)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (512*x1) + (65536*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lu/clupus6xykl2uq647aklnfnrrgk55oic2osvrg5jtpjeuymifxfm.py
# Topologically Sorted Source Nodes: [attn_weights_19], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_weights_19 => amax_8, div_8, exp_8, sub_26, sum_9
# Graph fragment:
#   %amax_8 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_176, [-1], True), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_176, %amax_8), kwargs = {})
#   %exp_8 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_26,), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_8, [-1], True), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_8, %sum_9), kwargs = {})
triton_per_fused__softmax_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
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


# kernel path: /tmp/torchinductor_sahanp/v7/cv72fgwr6vdsinsblpwrkctbf3zhnovu6xhzzacbqvqs7hszxkt5.py
# Topologically Sorted Source Nodes: [attn_output_43], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_output_43 => clone_70
# Graph fragment:
#   %clone_70 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_95,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 16
    x2 = (xindex // 512) % 128
    x3 = (xindex // 65536)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (4096*x1) + (65536*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qb/cqbaj3ectqkkwtvzqqkiy664x7udvcgo3eq6ddm4sdwo4b6kidz4.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_248, %full_default_4], 1), kwargs = {})
triton_poi_fused_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25737216
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
    tmp5 = tl.load(in_ptr0 + (x1 + (512*x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 50268, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x0 + (50272*x1)), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2u/c2uidlwnzqkkeokbvs6yk2gakp6ddd2wtmwkwpvfi2qf4douk2hz.py
# Topologically Sorted Source Nodes: [lm_logits, masked_lm_loss], Original ATen: [aten.add, aten._log_softmax]
# Source node to ATen node mapping:
#   lm_logits => add_151
#   masked_lm_loss => amax_24, exp_24, sub_66, sum_25
# Graph fragment:
#   %add_151 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_468, %arg345_1), kwargs = {})
#   %amax_24 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_469, [1], True), kwargs = {})
#   %sub_66 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_469, %amax_24), kwargs = {})
#   %exp_24 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_66,), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [1], True), kwargs = {})
triton_red_fused__log_softmax_add_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_add_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
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


# kernel path: /tmp/torchinductor_sahanp/ki/ckibdy5nl63kcfewnx4zapps3mdwne35edqjwrx3qtcy6yyzvgo3.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   masked_lm_loss => convert_element_type, div_24, full_default_3, ne_1, ne_2, neg, sum_26, sum_27, where_2
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_470, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_3), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_2,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_470, -100), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_26, torch.float32), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_27, %convert_element_type), kwargs = {})
triton_red_fused_nll_loss_forward_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_13', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 128), (128, 1))
    assert_size_stride(arg1_1, (64, 128), (128, 1))
    assert_size_stride(arg2_1, (50265, 512), (512, 1))
    assert_size_stride(arg3_1, (512, 512), (512, 1))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, ), (1, ))
    assert_size_stride(arg6_1, (512, 512), (512, 1))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (512, 512), (512, 1))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (512, 512), (512, 1))
    assert_size_stride(arg11_1, (512, ), (1, ))
    assert_size_stride(arg12_1, (512, 512), (512, 1))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, ), (1, ))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (2048, 512), (512, 1))
    assert_size_stride(arg17_1, (2048, ), (1, ))
    assert_size_stride(arg18_1, (512, 2048), (2048, 1))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (512, ), (1, ))
    assert_size_stride(arg22_1, (512, 512), (512, 1))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, 512), (512, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, 512), (512, 1))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (512, 512), (512, 1))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (2048, 512), (512, 1))
    assert_size_stride(arg33_1, (2048, ), (1, ))
    assert_size_stride(arg34_1, (512, 2048), (2048, 1))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (512, 512), (512, 1))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (512, 512), (512, 1))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, 512), (512, 1))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, 512), (512, 1))
    assert_size_stride(arg45_1, (512, ), (1, ))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (2048, 512), (512, 1))
    assert_size_stride(arg49_1, (2048, ), (1, ))
    assert_size_stride(arg50_1, (512, 2048), (2048, 1))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (512, 512), (512, 1))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (512, 512), (512, 1))
    assert_size_stride(arg57_1, (512, ), (1, ))
    assert_size_stride(arg58_1, (512, 512), (512, 1))
    assert_size_stride(arg59_1, (512, ), (1, ))
    assert_size_stride(arg60_1, (512, 512), (512, 1))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (512, ), (1, ))
    assert_size_stride(arg64_1, (2048, 512), (512, 1))
    assert_size_stride(arg65_1, (2048, ), (1, ))
    assert_size_stride(arg66_1, (512, 2048), (2048, 1))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, 512), (512, 1))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (512, 512), (512, 1))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, 512), (512, 1))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (512, 512), (512, 1))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (2048, 512), (512, 1))
    assert_size_stride(arg81_1, (2048, ), (1, ))
    assert_size_stride(arg82_1, (512, 2048), (2048, 1))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, ), (1, ))
    assert_size_stride(arg86_1, (512, 512), (512, 1))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (512, 512), (512, 1))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (512, 512), (512, 1))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (512, 512), (512, 1))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (512, ), (1, ))
    assert_size_stride(arg95_1, (512, ), (1, ))
    assert_size_stride(arg96_1, (2048, 512), (512, 1))
    assert_size_stride(arg97_1, (2048, ), (1, ))
    assert_size_stride(arg98_1, (512, 2048), (2048, 1))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (512, ), (1, ))
    assert_size_stride(arg101_1, (512, ), (1, ))
    assert_size_stride(arg102_1, (512, 512), (512, 1))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, 512), (512, 1))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (512, 512), (512, 1))
    assert_size_stride(arg107_1, (512, ), (1, ))
    assert_size_stride(arg108_1, (512, 512), (512, 1))
    assert_size_stride(arg109_1, (512, ), (1, ))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (512, ), (1, ))
    assert_size_stride(arg112_1, (2048, 512), (512, 1))
    assert_size_stride(arg113_1, (2048, ), (1, ))
    assert_size_stride(arg114_1, (512, 2048), (2048, 1))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (512, ), (1, ))
    assert_size_stride(arg117_1, (512, ), (1, ))
    assert_size_stride(arg118_1, (512, 512), (512, 1))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (512, 512), (512, 1))
    assert_size_stride(arg121_1, (512, ), (1, ))
    assert_size_stride(arg122_1, (512, 512), (512, 1))
    assert_size_stride(arg123_1, (512, ), (1, ))
    assert_size_stride(arg124_1, (512, 512), (512, 1))
    assert_size_stride(arg125_1, (512, ), (1, ))
    assert_size_stride(arg126_1, (512, ), (1, ))
    assert_size_stride(arg127_1, (512, ), (1, ))
    assert_size_stride(arg128_1, (2048, 512), (512, 1))
    assert_size_stride(arg129_1, (2048, ), (1, ))
    assert_size_stride(arg130_1, (512, 2048), (2048, 1))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (512, ), (1, ))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (512, 512), (512, 1))
    assert_size_stride(arg135_1, (512, ), (1, ))
    assert_size_stride(arg136_1, (512, ), (1, ))
    assert_size_stride(arg137_1, (512, 512), (512, 1))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (512, 512), (512, 1))
    assert_size_stride(arg140_1, (512, ), (1, ))
    assert_size_stride(arg141_1, (512, 512), (512, 1))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, 512), (512, 1))
    assert_size_stride(arg144_1, (512, ), (1, ))
    assert_size_stride(arg145_1, (512, ), (1, ))
    assert_size_stride(arg146_1, (512, ), (1, ))
    assert_size_stride(arg147_1, (512, 512), (512, 1))
    assert_size_stride(arg148_1, (512, ), (1, ))
    assert_size_stride(arg149_1, (512, 512), (512, 1))
    assert_size_stride(arg150_1, (512, ), (1, ))
    assert_size_stride(arg151_1, (512, 512), (512, 1))
    assert_size_stride(arg152_1, (512, ), (1, ))
    assert_size_stride(arg153_1, (512, 512), (512, 1))
    assert_size_stride(arg154_1, (512, ), (1, ))
    assert_size_stride(arg155_1, (512, ), (1, ))
    assert_size_stride(arg156_1, (512, ), (1, ))
    assert_size_stride(arg157_1, (2048, 512), (512, 1))
    assert_size_stride(arg158_1, (2048, ), (1, ))
    assert_size_stride(arg159_1, (512, 2048), (2048, 1))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (512, ), (1, ))
    assert_size_stride(arg162_1, (512, ), (1, ))
    assert_size_stride(arg163_1, (512, 512), (512, 1))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (512, 512), (512, 1))
    assert_size_stride(arg166_1, (512, ), (1, ))
    assert_size_stride(arg167_1, (512, 512), (512, 1))
    assert_size_stride(arg168_1, (512, ), (1, ))
    assert_size_stride(arg169_1, (512, 512), (512, 1))
    assert_size_stride(arg170_1, (512, ), (1, ))
    assert_size_stride(arg171_1, (512, ), (1, ))
    assert_size_stride(arg172_1, (512, ), (1, ))
    assert_size_stride(arg173_1, (512, 512), (512, 1))
    assert_size_stride(arg174_1, (512, ), (1, ))
    assert_size_stride(arg175_1, (512, 512), (512, 1))
    assert_size_stride(arg176_1, (512, ), (1, ))
    assert_size_stride(arg177_1, (512, 512), (512, 1))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (512, 512), (512, 1))
    assert_size_stride(arg180_1, (512, ), (1, ))
    assert_size_stride(arg181_1, (512, ), (1, ))
    assert_size_stride(arg182_1, (512, ), (1, ))
    assert_size_stride(arg183_1, (2048, 512), (512, 1))
    assert_size_stride(arg184_1, (2048, ), (1, ))
    assert_size_stride(arg185_1, (512, 2048), (2048, 1))
    assert_size_stride(arg186_1, (512, ), (1, ))
    assert_size_stride(arg187_1, (512, ), (1, ))
    assert_size_stride(arg188_1, (512, ), (1, ))
    assert_size_stride(arg189_1, (512, 512), (512, 1))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (512, 512), (512, 1))
    assert_size_stride(arg192_1, (512, ), (1, ))
    assert_size_stride(arg193_1, (512, 512), (512, 1))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (512, 512), (512, 1))
    assert_size_stride(arg196_1, (512, ), (1, ))
    assert_size_stride(arg197_1, (512, ), (1, ))
    assert_size_stride(arg198_1, (512, ), (1, ))
    assert_size_stride(arg199_1, (512, 512), (512, 1))
    assert_size_stride(arg200_1, (512, ), (1, ))
    assert_size_stride(arg201_1, (512, 512), (512, 1))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (512, 512), (512, 1))
    assert_size_stride(arg204_1, (512, ), (1, ))
    assert_size_stride(arg205_1, (512, 512), (512, 1))
    assert_size_stride(arg206_1, (512, ), (1, ))
    assert_size_stride(arg207_1, (512, ), (1, ))
    assert_size_stride(arg208_1, (512, ), (1, ))
    assert_size_stride(arg209_1, (2048, 512), (512, 1))
    assert_size_stride(arg210_1, (2048, ), (1, ))
    assert_size_stride(arg211_1, (512, 2048), (2048, 1))
    assert_size_stride(arg212_1, (512, ), (1, ))
    assert_size_stride(arg213_1, (512, ), (1, ))
    assert_size_stride(arg214_1, (512, ), (1, ))
    assert_size_stride(arg215_1, (512, 512), (512, 1))
    assert_size_stride(arg216_1, (512, ), (1, ))
    assert_size_stride(arg217_1, (512, 512), (512, 1))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (512, 512), (512, 1))
    assert_size_stride(arg220_1, (512, ), (1, ))
    assert_size_stride(arg221_1, (512, 512), (512, 1))
    assert_size_stride(arg222_1, (512, ), (1, ))
    assert_size_stride(arg223_1, (512, ), (1, ))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (512, 512), (512, 1))
    assert_size_stride(arg226_1, (512, ), (1, ))
    assert_size_stride(arg227_1, (512, 512), (512, 1))
    assert_size_stride(arg228_1, (512, ), (1, ))
    assert_size_stride(arg229_1, (512, 512), (512, 1))
    assert_size_stride(arg230_1, (512, ), (1, ))
    assert_size_stride(arg231_1, (512, 512), (512, 1))
    assert_size_stride(arg232_1, (512, ), (1, ))
    assert_size_stride(arg233_1, (512, ), (1, ))
    assert_size_stride(arg234_1, (512, ), (1, ))
    assert_size_stride(arg235_1, (2048, 512), (512, 1))
    assert_size_stride(arg236_1, (2048, ), (1, ))
    assert_size_stride(arg237_1, (512, 2048), (2048, 1))
    assert_size_stride(arg238_1, (512, ), (1, ))
    assert_size_stride(arg239_1, (512, ), (1, ))
    assert_size_stride(arg240_1, (512, ), (1, ))
    assert_size_stride(arg241_1, (512, 512), (512, 1))
    assert_size_stride(arg242_1, (512, ), (1, ))
    assert_size_stride(arg243_1, (512, 512), (512, 1))
    assert_size_stride(arg244_1, (512, ), (1, ))
    assert_size_stride(arg245_1, (512, 512), (512, 1))
    assert_size_stride(arg246_1, (512, ), (1, ))
    assert_size_stride(arg247_1, (512, 512), (512, 1))
    assert_size_stride(arg248_1, (512, ), (1, ))
    assert_size_stride(arg249_1, (512, ), (1, ))
    assert_size_stride(arg250_1, (512, ), (1, ))
    assert_size_stride(arg251_1, (512, 512), (512, 1))
    assert_size_stride(arg252_1, (512, ), (1, ))
    assert_size_stride(arg253_1, (512, 512), (512, 1))
    assert_size_stride(arg254_1, (512, ), (1, ))
    assert_size_stride(arg255_1, (512, 512), (512, 1))
    assert_size_stride(arg256_1, (512, ), (1, ))
    assert_size_stride(arg257_1, (512, 512), (512, 1))
    assert_size_stride(arg258_1, (512, ), (1, ))
    assert_size_stride(arg259_1, (512, ), (1, ))
    assert_size_stride(arg260_1, (512, ), (1, ))
    assert_size_stride(arg261_1, (2048, 512), (512, 1))
    assert_size_stride(arg262_1, (2048, ), (1, ))
    assert_size_stride(arg263_1, (512, 2048), (2048, 1))
    assert_size_stride(arg264_1, (512, ), (1, ))
    assert_size_stride(arg265_1, (512, ), (1, ))
    assert_size_stride(arg266_1, (512, ), (1, ))
    assert_size_stride(arg267_1, (512, 512), (512, 1))
    assert_size_stride(arg268_1, (512, ), (1, ))
    assert_size_stride(arg269_1, (512, 512), (512, 1))
    assert_size_stride(arg270_1, (512, ), (1, ))
    assert_size_stride(arg271_1, (512, 512), (512, 1))
    assert_size_stride(arg272_1, (512, ), (1, ))
    assert_size_stride(arg273_1, (512, 512), (512, 1))
    assert_size_stride(arg274_1, (512, ), (1, ))
    assert_size_stride(arg275_1, (512, ), (1, ))
    assert_size_stride(arg276_1, (512, ), (1, ))
    assert_size_stride(arg277_1, (512, 512), (512, 1))
    assert_size_stride(arg278_1, (512, ), (1, ))
    assert_size_stride(arg279_1, (512, 512), (512, 1))
    assert_size_stride(arg280_1, (512, ), (1, ))
    assert_size_stride(arg281_1, (512, 512), (512, 1))
    assert_size_stride(arg282_1, (512, ), (1, ))
    assert_size_stride(arg283_1, (512, 512), (512, 1))
    assert_size_stride(arg284_1, (512, ), (1, ))
    assert_size_stride(arg285_1, (512, ), (1, ))
    assert_size_stride(arg286_1, (512, ), (1, ))
    assert_size_stride(arg287_1, (2048, 512), (512, 1))
    assert_size_stride(arg288_1, (2048, ), (1, ))
    assert_size_stride(arg289_1, (512, 2048), (2048, 1))
    assert_size_stride(arg290_1, (512, ), (1, ))
    assert_size_stride(arg291_1, (512, ), (1, ))
    assert_size_stride(arg292_1, (512, ), (1, ))
    assert_size_stride(arg293_1, (512, 512), (512, 1))
    assert_size_stride(arg294_1, (512, ), (1, ))
    assert_size_stride(arg295_1, (512, 512), (512, 1))
    assert_size_stride(arg296_1, (512, ), (1, ))
    assert_size_stride(arg297_1, (512, 512), (512, 1))
    assert_size_stride(arg298_1, (512, ), (1, ))
    assert_size_stride(arg299_1, (512, 512), (512, 1))
    assert_size_stride(arg300_1, (512, ), (1, ))
    assert_size_stride(arg301_1, (512, ), (1, ))
    assert_size_stride(arg302_1, (512, ), (1, ))
    assert_size_stride(arg303_1, (512, 512), (512, 1))
    assert_size_stride(arg304_1, (512, ), (1, ))
    assert_size_stride(arg305_1, (512, 512), (512, 1))
    assert_size_stride(arg306_1, (512, ), (1, ))
    assert_size_stride(arg307_1, (512, 512), (512, 1))
    assert_size_stride(arg308_1, (512, ), (1, ))
    assert_size_stride(arg309_1, (512, 512), (512, 1))
    assert_size_stride(arg310_1, (512, ), (1, ))
    assert_size_stride(arg311_1, (512, ), (1, ))
    assert_size_stride(arg312_1, (512, ), (1, ))
    assert_size_stride(arg313_1, (2048, 512), (512, 1))
    assert_size_stride(arg314_1, (2048, ), (1, ))
    assert_size_stride(arg315_1, (512, 2048), (2048, 1))
    assert_size_stride(arg316_1, (512, ), (1, ))
    assert_size_stride(arg317_1, (512, ), (1, ))
    assert_size_stride(arg318_1, (512, ), (1, ))
    assert_size_stride(arg319_1, (512, 512), (512, 1))
    assert_size_stride(arg320_1, (512, ), (1, ))
    assert_size_stride(arg321_1, (512, 512), (512, 1))
    assert_size_stride(arg322_1, (512, ), (1, ))
    assert_size_stride(arg323_1, (512, 512), (512, 1))
    assert_size_stride(arg324_1, (512, ), (1, ))
    assert_size_stride(arg325_1, (512, 512), (512, 1))
    assert_size_stride(arg326_1, (512, ), (1, ))
    assert_size_stride(arg327_1, (512, ), (1, ))
    assert_size_stride(arg328_1, (512, ), (1, ))
    assert_size_stride(arg329_1, (512, 512), (512, 1))
    assert_size_stride(arg330_1, (512, ), (1, ))
    assert_size_stride(arg331_1, (512, 512), (512, 1))
    assert_size_stride(arg332_1, (512, ), (1, ))
    assert_size_stride(arg333_1, (512, 512), (512, 1))
    assert_size_stride(arg334_1, (512, ), (1, ))
    assert_size_stride(arg335_1, (512, 512), (512, 1))
    assert_size_stride(arg336_1, (512, ), (1, ))
    assert_size_stride(arg337_1, (512, ), (1, ))
    assert_size_stride(arg338_1, (512, ), (1, ))
    assert_size_stride(arg339_1, (2048, 512), (512, 1))
    assert_size_stride(arg340_1, (2048, ), (1, ))
    assert_size_stride(arg341_1, (512, 2048), (2048, 1))
    assert_size_stride(arg342_1, (512, ), (1, ))
    assert_size_stride(arg343_1, (512, ), (1, ))
    assert_size_stride(arg344_1, (512, ), (1, ))
    assert_size_stride(arg345_1, (1, 50265), (50265, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((64, 128, 512), (65536, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, positions, embed_pos, hidden_states, hidden_states_1], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_arange_embedding_mul_native_layer_norm_0.run(arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, buf3, 8192, 512, grid=grid(8192), stream=stream0)
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((8192, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (8192, 512), (512, 1), 0), reinterpret_tensor(arg6_1, (512, 512), (1, 512), 0), out=buf4)
        del arg6_1
        buf5 = empty_strided_cuda((8192, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (8192, 512), (512, 1), 0), reinterpret_tensor(arg8_1, (512, 512), (1, 512), 0), out=buf5)
        del arg8_1
        buf6 = empty_strided_cuda((8192, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (8192, 512), (512, 1), 0), reinterpret_tensor(arg10_1, (512, 512), (1, 512), 0), out=buf6)
        del arg10_1
        buf7 = empty_strided_cuda((1, 1024, 128, 32), (4194304, 4096, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf4, arg7_1, buf7, 4194304, grid=grid(4194304), stream=stream0)
        del arg7_1
        buf8 = reinterpret_tensor(buf4, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf5, arg9_1, buf8, 4194304, grid=grid(4194304), stream=stream0)
        del arg9_1
        buf9 = reinterpret_tensor(buf5, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf6, arg11_1, buf9, 4194304, grid=grid(4194304), stream=stream0)
        del arg11_1
        del buf6
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf7, buf8, buf9, None, False, scale=1.0)
        buf11 = buf10[0]
        del buf10
        buf15 = reinterpret_tensor(buf9, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf11, buf15, 4194304, grid=grid(4194304), stream=stream0)
        buf16 = reinterpret_tensor(buf11, (8192, 512), (512, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (8192, 512), (512, 1), 0), reinterpret_tensor(arg12_1, (512, 512), (1, 512), 0), out=buf16)
        del arg12_1
        buf20 = reinterpret_tensor(buf15, (64, 128, 512), (65536, 512, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf3, buf16, arg13_1, arg14_1, arg15_1, buf20, 8192, 512, grid=grid(8192), stream=stream0)
        del arg13_1
        del arg14_1
        del arg15_1
        buf21 = empty_strided_cuda((8192, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (8192, 512), (512, 1), 0), reinterpret_tensor(arg16_1, (512, 2048), (1, 512), 0), out=buf21)
        del arg16_1
        buf22 = reinterpret_tensor(buf21, (64, 128, 2048), (262144, 2048, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf22, arg17_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg17_1
        buf23 = reinterpret_tensor(buf3, (8192, 512), (512, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg18_1, (2048, 512), (1, 2048), 0), out=buf23)
        del arg18_1
        buf27 = reinterpret_tensor(buf16, (64, 128, 512), (65536, 512, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf20, buf23, arg19_1, arg20_1, arg21_1, buf27, 8192, 512, grid=grid(8192), stream=stream0)
        del arg19_1
        del arg20_1
        del arg21_1
        buf28 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (8192, 512), (512, 1), 0), reinterpret_tensor(arg22_1, (512, 512), (1, 512), 0), out=buf28)
        del arg22_1
        buf29 = reinterpret_tensor(buf20, (8192, 512), (512, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (8192, 512), (512, 1), 0), reinterpret_tensor(arg24_1, (512, 512), (1, 512), 0), out=buf29)
        del arg24_1
        buf30 = reinterpret_tensor(buf8, (8192, 512), (512, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (8192, 512), (512, 1), 0), reinterpret_tensor(arg26_1, (512, 512), (1, 512), 0), out=buf30)
        del arg26_1
        buf31 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf28, arg23_1, buf31, 4194304, grid=grid(4194304), stream=stream0)
        del arg23_1
        buf32 = reinterpret_tensor(buf28, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf29, arg25_1, buf32, 4194304, grid=grid(4194304), stream=stream0)
        del arg25_1
        buf33 = reinterpret_tensor(buf29, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf30, arg27_1, buf33, 4194304, grid=grid(4194304), stream=stream0)
        del arg27_1
        del buf30
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf34 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf31, buf32, buf33, None, False, scale=1.0)
        buf35 = buf34[0]
        del buf34
        buf39 = reinterpret_tensor(buf33, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf35, buf39, 4194304, grid=grid(4194304), stream=stream0)
        buf40 = reinterpret_tensor(buf35, (8192, 512), (512, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (8192, 512), (512, 1), 0), reinterpret_tensor(arg28_1, (512, 512), (1, 512), 0), out=buf40)
        del arg28_1
        buf44 = reinterpret_tensor(buf39, (64, 128, 512), (65536, 512, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf27, buf40, arg29_1, arg30_1, arg31_1, buf44, 8192, 512, grid=grid(8192), stream=stream0)
        del arg29_1
        del arg30_1
        del arg31_1
        buf45 = reinterpret_tensor(buf22, (8192, 2048), (2048, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (8192, 512), (512, 1), 0), reinterpret_tensor(arg32_1, (512, 2048), (1, 512), 0), out=buf45)
        del arg32_1
        buf46 = reinterpret_tensor(buf45, (64, 128, 2048), (262144, 2048, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf46, arg33_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg33_1
        buf47 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg34_1, (2048, 512), (1, 2048), 0), out=buf47)
        del arg34_1
        buf51 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf44, buf47, arg35_1, arg36_1, arg37_1, buf51, 8192, 512, grid=grid(8192), stream=stream0)
        del arg35_1
        del arg36_1
        del arg37_1
        buf52 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (8192, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 512), (1, 512), 0), out=buf52)
        del arg38_1
        buf53 = reinterpret_tensor(buf44, (8192, 512), (512, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (8192, 512), (512, 1), 0), reinterpret_tensor(arg40_1, (512, 512), (1, 512), 0), out=buf53)
        del arg40_1
        buf54 = reinterpret_tensor(buf32, (8192, 512), (512, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (8192, 512), (512, 1), 0), reinterpret_tensor(arg42_1, (512, 512), (1, 512), 0), out=buf54)
        del arg42_1
        buf55 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf52, arg39_1, buf55, 4194304, grid=grid(4194304), stream=stream0)
        del arg39_1
        buf56 = reinterpret_tensor(buf52, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf53, arg41_1, buf56, 4194304, grid=grid(4194304), stream=stream0)
        del arg41_1
        buf57 = reinterpret_tensor(buf53, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf54, arg43_1, buf57, 4194304, grid=grid(4194304), stream=stream0)
        del arg43_1
        del buf54
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf58 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf55, buf56, buf57, None, False, scale=1.0)
        buf59 = buf58[0]
        del buf58
        buf63 = reinterpret_tensor(buf57, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf59, buf63, 4194304, grid=grid(4194304), stream=stream0)
        buf64 = reinterpret_tensor(buf59, (8192, 512), (512, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (8192, 512), (512, 1), 0), reinterpret_tensor(arg44_1, (512, 512), (1, 512), 0), out=buf64)
        del arg44_1
        buf68 = reinterpret_tensor(buf63, (64, 128, 512), (65536, 512, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_22, hidden_states_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf51, buf64, arg45_1, arg46_1, arg47_1, buf68, 8192, 512, grid=grid(8192), stream=stream0)
        del arg45_1
        del arg46_1
        del arg47_1
        buf69 = reinterpret_tensor(buf46, (8192, 2048), (2048, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (8192, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 2048), (1, 512), 0), out=buf69)
        del arg48_1
        buf70 = reinterpret_tensor(buf69, (64, 128, 2048), (262144, 2048, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf70, arg49_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg49_1
        buf71 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg50_1, (2048, 512), (1, 2048), 0), out=buf71)
        del arg50_1
        buf75 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf68, buf71, arg51_1, arg52_1, arg53_1, buf75, 8192, 512, grid=grid(8192), stream=stream0)
        del arg51_1
        del arg52_1
        del arg53_1
        buf76 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (8192, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 512), (1, 512), 0), out=buf76)
        del arg54_1
        buf77 = reinterpret_tensor(buf68, (8192, 512), (512, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (8192, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 512), (1, 512), 0), out=buf77)
        del arg56_1
        buf78 = reinterpret_tensor(buf56, (8192, 512), (512, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (8192, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 512), (1, 512), 0), out=buf78)
        del arg58_1
        buf79 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf76, arg55_1, buf79, 4194304, grid=grid(4194304), stream=stream0)
        del arg55_1
        buf80 = reinterpret_tensor(buf76, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf77, arg57_1, buf80, 4194304, grid=grid(4194304), stream=stream0)
        del arg57_1
        buf81 = reinterpret_tensor(buf77, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf78, arg59_1, buf81, 4194304, grid=grid(4194304), stream=stream0)
        del arg59_1
        del buf78
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf82 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf79, buf80, buf81, None, False, scale=1.0)
        buf83 = buf82[0]
        del buf82
        buf87 = reinterpret_tensor(buf81, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf83, buf87, 4194304, grid=grid(4194304), stream=stream0)
        buf88 = reinterpret_tensor(buf83, (8192, 512), (512, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (8192, 512), (512, 1), 0), reinterpret_tensor(arg60_1, (512, 512), (1, 512), 0), out=buf88)
        del arg60_1
        buf92 = reinterpret_tensor(buf87, (64, 128, 512), (65536, 512, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf75, buf88, arg61_1, arg62_1, arg63_1, buf92, 8192, 512, grid=grid(8192), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        buf93 = reinterpret_tensor(buf70, (8192, 2048), (2048, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (8192, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 2048), (1, 512), 0), out=buf93)
        del arg64_1
        buf94 = reinterpret_tensor(buf93, (64, 128, 2048), (262144, 2048, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_33], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf94, arg65_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg65_1
        buf95 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg66_1, (2048, 512), (1, 2048), 0), out=buf95)
        del arg66_1
        buf99 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf92, buf95, arg67_1, arg68_1, arg69_1, buf99, 8192, 512, grid=grid(8192), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        buf100 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (8192, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 512), (1, 512), 0), out=buf100)
        del arg70_1
        buf101 = reinterpret_tensor(buf92, (8192, 512), (512, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (8192, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 512), (1, 512), 0), out=buf101)
        del arg72_1
        buf102 = reinterpret_tensor(buf80, (8192, 512), (512, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (8192, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 512), (1, 512), 0), out=buf102)
        del arg74_1
        buf103 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf100, arg71_1, buf103, 4194304, grid=grid(4194304), stream=stream0)
        del arg71_1
        buf104 = reinterpret_tensor(buf100, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf101, arg73_1, buf104, 4194304, grid=grid(4194304), stream=stream0)
        del arg73_1
        buf105 = reinterpret_tensor(buf101, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf102, arg75_1, buf105, 4194304, grid=grid(4194304), stream=stream0)
        del arg75_1
        del buf102
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf106 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf103, buf104, buf105, None, False, scale=1.0)
        buf107 = buf106[0]
        del buf106
        buf111 = reinterpret_tensor(buf105, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf107, buf111, 4194304, grid=grid(4194304), stream=stream0)
        buf112 = reinterpret_tensor(buf107, (8192, 512), (512, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (8192, 512), (512, 1), 0), reinterpret_tensor(arg76_1, (512, 512), (1, 512), 0), out=buf112)
        del arg76_1
        buf116 = reinterpret_tensor(buf111, (64, 128, 512), (65536, 512, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf99, buf112, arg77_1, arg78_1, arg79_1, buf116, 8192, 512, grid=grid(8192), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        buf117 = reinterpret_tensor(buf94, (8192, 2048), (2048, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (8192, 512), (512, 1), 0), reinterpret_tensor(arg80_1, (512, 2048), (1, 512), 0), out=buf117)
        del arg80_1
        buf118 = reinterpret_tensor(buf117, (64, 128, 2048), (262144, 2048, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf118, arg81_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg81_1
        buf119 = reinterpret_tensor(buf99, (8192, 512), (512, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg82_1, (2048, 512), (1, 2048), 0), out=buf119)
        del arg82_1
        buf123 = reinterpret_tensor(buf112, (64, 128, 512), (65536, 512, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf116, buf119, arg83_1, arg84_1, arg85_1, buf123, 8192, 512, grid=grid(8192), stream=stream0)
        del arg83_1
        del arg84_1
        del arg85_1
        buf124 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (8192, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 512), (1, 512), 0), out=buf124)
        del arg86_1
        buf125 = reinterpret_tensor(buf116, (8192, 512), (512, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (8192, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 512), (1, 512), 0), out=buf125)
        del arg88_1
        buf126 = reinterpret_tensor(buf104, (8192, 512), (512, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (8192, 512), (512, 1), 0), reinterpret_tensor(arg90_1, (512, 512), (1, 512), 0), out=buf126)
        del arg90_1
        buf127 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf124, arg87_1, buf127, 4194304, grid=grid(4194304), stream=stream0)
        del arg87_1
        buf128 = reinterpret_tensor(buf124, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf125, arg89_1, buf128, 4194304, grid=grid(4194304), stream=stream0)
        del arg89_1
        buf129 = reinterpret_tensor(buf125, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf126, arg91_1, buf129, 4194304, grid=grid(4194304), stream=stream0)
        del arg91_1
        del buf126
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf130 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf127, buf128, buf129, None, False, scale=1.0)
        buf131 = buf130[0]
        del buf130
        buf135 = reinterpret_tensor(buf129, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf131, buf135, 4194304, grid=grid(4194304), stream=stream0)
        buf136 = reinterpret_tensor(buf131, (8192, 512), (512, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (8192, 512), (512, 1), 0), reinterpret_tensor(arg92_1, (512, 512), (1, 512), 0), out=buf136)
        del arg92_1
        buf140 = reinterpret_tensor(buf135, (64, 128, 512), (65536, 512, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf123, buf136, arg93_1, arg94_1, arg95_1, buf140, 8192, 512, grid=grid(8192), stream=stream0)
        del arg93_1
        del arg94_1
        del arg95_1
        buf141 = reinterpret_tensor(buf118, (8192, 2048), (2048, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (8192, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 2048), (1, 512), 0), out=buf141)
        del arg96_1
        buf142 = reinterpret_tensor(buf141, (64, 128, 2048), (262144, 2048, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf142, arg97_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg97_1
        buf143 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg98_1, (2048, 512), (1, 2048), 0), out=buf143)
        del arg98_1
        buf147 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf140, buf143, arg99_1, arg100_1, arg101_1, buf147, 8192, 512, grid=grid(8192), stream=stream0)
        del arg100_1
        del arg101_1
        del arg99_1
        buf148 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (8192, 512), (512, 1), 0), reinterpret_tensor(arg102_1, (512, 512), (1, 512), 0), out=buf148)
        del arg102_1
        buf149 = reinterpret_tensor(buf140, (8192, 512), (512, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (8192, 512), (512, 1), 0), reinterpret_tensor(arg104_1, (512, 512), (1, 512), 0), out=buf149)
        del arg104_1
        buf150 = reinterpret_tensor(buf128, (8192, 512), (512, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (8192, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 512), (1, 512), 0), out=buf150)
        del arg106_1
        buf151 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf148, arg103_1, buf151, 4194304, grid=grid(4194304), stream=stream0)
        del arg103_1
        buf152 = reinterpret_tensor(buf148, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf149, arg105_1, buf152, 4194304, grid=grid(4194304), stream=stream0)
        del arg105_1
        buf153 = reinterpret_tensor(buf149, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf150, arg107_1, buf153, 4194304, grid=grid(4194304), stream=stream0)
        del arg107_1
        del buf150
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf154 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf151, buf152, buf153, None, False, scale=1.0)
        buf155 = buf154[0]
        del buf154
        buf159 = reinterpret_tensor(buf153, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf155, buf159, 4194304, grid=grid(4194304), stream=stream0)
        buf160 = reinterpret_tensor(buf155, (8192, 512), (512, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (8192, 512), (512, 1), 0), reinterpret_tensor(arg108_1, (512, 512), (1, 512), 0), out=buf160)
        del arg108_1
        buf164 = reinterpret_tensor(buf159, (64, 128, 512), (65536, 512, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf147, buf160, arg109_1, arg110_1, arg111_1, buf164, 8192, 512, grid=grid(8192), stream=stream0)
        del arg109_1
        del arg110_1
        del arg111_1
        buf165 = reinterpret_tensor(buf142, (8192, 2048), (2048, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (8192, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 2048), (1, 512), 0), out=buf165)
        del arg112_1
        buf166 = reinterpret_tensor(buf165, (64, 128, 2048), (262144, 2048, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_60], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf166, arg113_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg113_1
        buf167 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg114_1, (2048, 512), (1, 2048), 0), out=buf167)
        del arg114_1
        buf171 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf164, buf167, arg115_1, arg116_1, arg117_1, buf171, 8192, 512, grid=grid(8192), stream=stream0)
        del arg115_1
        del arg116_1
        del arg117_1
        buf172 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (8192, 512), (512, 1), 0), reinterpret_tensor(arg118_1, (512, 512), (1, 512), 0), out=buf172)
        del arg118_1
        buf173 = reinterpret_tensor(buf164, (8192, 512), (512, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (8192, 512), (512, 1), 0), reinterpret_tensor(arg120_1, (512, 512), (1, 512), 0), out=buf173)
        del arg120_1
        buf174 = reinterpret_tensor(buf152, (8192, 512), (512, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (8192, 512), (512, 1), 0), reinterpret_tensor(arg122_1, (512, 512), (1, 512), 0), out=buf174)
        del arg122_1
        buf175 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf172, arg119_1, buf175, 4194304, grid=grid(4194304), stream=stream0)
        del arg119_1
        buf176 = reinterpret_tensor(buf172, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf173, arg121_1, buf176, 4194304, grid=grid(4194304), stream=stream0)
        del arg121_1
        buf177 = reinterpret_tensor(buf173, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf174, arg123_1, buf177, 4194304, grid=grid(4194304), stream=stream0)
        del arg123_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf178 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf175, buf176, buf177, None, False, scale=1.0)
        buf179 = buf178[0]
        del buf178
        buf183 = reinterpret_tensor(buf177, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf179, buf183, 4194304, grid=grid(4194304), stream=stream0)
        buf184 = reinterpret_tensor(buf179, (8192, 512), (512, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf183, (8192, 512), (512, 1), 0), reinterpret_tensor(arg124_1, (512, 512), (1, 512), 0), out=buf184)
        del arg124_1
        buf188 = reinterpret_tensor(buf183, (64, 128, 512), (65536, 512, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf171, buf184, arg125_1, arg126_1, arg127_1, buf188, 8192, 512, grid=grid(8192), stream=stream0)
        del arg125_1
        del arg126_1
        del arg127_1
        buf189 = reinterpret_tensor(buf166, (8192, 2048), (2048, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (8192, 512), (512, 1), 0), reinterpret_tensor(arg128_1, (512, 2048), (1, 512), 0), out=buf189)
        del arg128_1
        buf190 = reinterpret_tensor(buf189, (64, 128, 2048), (262144, 2048, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf190, arg129_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg129_1
        buf191 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg130_1, (2048, 512), (1, 2048), 0), out=buf191)
        del arg130_1
        buf217 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf188, buf191, arg131_1, arg132_1, arg133_1, buf217, 8192, 512, grid=grid(8192), stream=stream0)
        del arg131_1
        del arg132_1
        del arg133_1
        buf198 = reinterpret_tensor(buf191, (64, 128, 512), (65536, 512, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [embedding_2, inputs_embeds_1, inputs_embeds_2, positions_1, positions_2, hidden_states_75], Original ATen: [aten.embedding, aten.mul, aten.native_layer_norm, aten.arange, aten.add]
        triton_red_fused_add_arange_embedding_mul_native_layer_norm_6.run(arg1_1, arg2_1, arg135_1, arg136_1, arg134_1, buf198, 8192, 512, grid=grid(8192), stream=stream0)
        del arg134_1
        del arg135_1
        del arg136_1
        del arg1_1
        buf199 = reinterpret_tensor(buf188, (8192, 512), (512, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (8192, 512), (512, 1), 0), reinterpret_tensor(arg137_1, (512, 512), (1, 512), 0), out=buf199)
        del arg137_1
        buf200 = reinterpret_tensor(buf176, (8192, 512), (512, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (8192, 512), (512, 1), 0), reinterpret_tensor(arg139_1, (512, 512), (1, 512), 0), out=buf200)
        del arg139_1
        buf201 = reinterpret_tensor(buf175, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf199, arg138_1, buf201, 4194304, grid=grid(4194304), stream=stream0)
        del arg138_1
        buf202 = reinterpret_tensor(buf199, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf200, arg140_1, buf202, 4194304, grid=grid(4194304), stream=stream0)
        del arg140_1
        buf203 = reinterpret_tensor(buf190, (1024, 128, 128), (16384, 128, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf201, (1024, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf202, (1024, 32, 128), (4096, 1, 32), 0), out=buf203)
        buf207 = empty_strided_cuda((1024, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_weights_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf203, buf207, 131072, 128, grid=grid(131072), stream=stream0)
        buf206 = reinterpret_tensor(buf202, (8192, 512), (512, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (8192, 512), (512, 1), 0), reinterpret_tensor(arg141_1, (512, 512), (1, 512), 0), out=buf206)
        del arg141_1
        buf208 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf206, arg142_1, buf208, 4194304, grid=grid(4194304), stream=stream0)
        del arg142_1
        buf209 = reinterpret_tensor(buf206, (1024, 128, 32), (4096, 32, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_19, attn_output_40], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf207, reinterpret_tensor(buf208, (1024, 128, 32), (4096, 32, 1), 0), out=buf209)
        buf210 = reinterpret_tensor(buf208, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf209, buf210, 4194304, grid=grid(4194304), stream=stream0)
        buf211 = reinterpret_tensor(buf209, (8192, 512), (512, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (8192, 512), (512, 1), 0), reinterpret_tensor(arg143_1, (512, 512), (1, 512), 0), out=buf211)
        del arg143_1
        buf215 = reinterpret_tensor(buf210, (64, 128, 512), (65536, 512, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf198, buf211, arg144_1, arg145_1, arg146_1, buf215, 8192, 512, grid=grid(8192), stream=stream0)
        del arg144_1
        del arg145_1
        del arg146_1
        buf216 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (8192, 512), (512, 1), 0), reinterpret_tensor(arg147_1, (512, 512), (1, 512), 0), out=buf216)
        del arg147_1
        buf218 = reinterpret_tensor(buf198, (8192, 512), (512, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg149_1, (512, 512), (1, 512), 0), out=buf218)
        del arg149_1
        buf219 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg151_1, (512, 512), (1, 512), 0), out=buf219)
        del arg151_1
        buf220 = reinterpret_tensor(buf174, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf216, arg148_1, buf220, 4194304, grid=grid(4194304), stream=stream0)
        del arg148_1
        buf221 = reinterpret_tensor(buf216, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf218, arg150_1, buf221, 4194304, grid=grid(4194304), stream=stream0)
        del arg150_1
        buf222 = reinterpret_tensor(buf218, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf219, arg152_1, buf222, 4194304, grid=grid(4194304), stream=stream0)
        del arg152_1
        del buf219
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf223 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf220, buf221, buf222, None, False, scale=1.0)
        buf224 = buf223[0]
        del buf223
        buf228 = reinterpret_tensor(buf222, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf224, buf228, 4194304, grid=grid(4194304), stream=stream0)
        buf229 = reinterpret_tensor(buf224, (8192, 512), (512, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (8192, 512), (512, 1), 0), reinterpret_tensor(arg153_1, (512, 512), (1, 512), 0), out=buf229)
        del arg153_1
        buf233 = reinterpret_tensor(buf228, (64, 128, 512), (65536, 512, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_81, hidden_states_82], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf215, buf229, arg154_1, arg155_1, arg156_1, buf233, 8192, 512, grid=grid(8192), stream=stream0)
        del arg154_1
        del arg155_1
        del arg156_1
        buf234 = reinterpret_tensor(buf207, (8192, 2048), (2048, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (8192, 512), (512, 1), 0), reinterpret_tensor(arg157_1, (512, 2048), (1, 512), 0), out=buf234)
        del arg157_1
        buf235 = reinterpret_tensor(buf234, (64, 128, 2048), (262144, 2048, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_83], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf235, arg158_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg158_1
        buf236 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg159_1, (2048, 512), (1, 2048), 0), out=buf236)
        del arg159_1
        buf240 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87, hidden_states_88], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf233, buf236, arg160_1, arg161_1, arg162_1, buf240, 8192, 512, grid=grid(8192), stream=stream0)
        del arg160_1
        del arg161_1
        del arg162_1
        buf241 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (8192, 512), (512, 1), 0), reinterpret_tensor(arg163_1, (512, 512), (1, 512), 0), out=buf241)
        del arg163_1
        buf242 = reinterpret_tensor(buf233, (8192, 512), (512, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (8192, 512), (512, 1), 0), reinterpret_tensor(arg165_1, (512, 512), (1, 512), 0), out=buf242)
        del arg165_1
        buf243 = reinterpret_tensor(buf221, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf241, arg164_1, buf243, 4194304, grid=grid(4194304), stream=stream0)
        del arg164_1
        buf244 = reinterpret_tensor(buf241, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf242, arg166_1, buf244, 4194304, grid=grid(4194304), stream=stream0)
        del arg166_1
        buf245 = reinterpret_tensor(buf235, (1024, 128, 128), (16384, 128, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf243, (1024, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf244, (1024, 32, 128), (4096, 1, 32), 0), out=buf245)
        buf249 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_25], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf245, buf249, 131072, 128, grid=grid(131072), stream=stream0)
        buf248 = reinterpret_tensor(buf244, (8192, 512), (512, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (8192, 512), (512, 1), 0), reinterpret_tensor(arg167_1, (512, 512), (1, 512), 0), out=buf248)
        del arg167_1
        buf250 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf248, arg168_1, buf250, 4194304, grid=grid(4194304), stream=stream0)
        del arg168_1
        buf251 = reinterpret_tensor(buf248, (1024, 128, 32), (4096, 32, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_25, attn_output_50], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf249, reinterpret_tensor(buf250, (1024, 128, 32), (4096, 32, 1), 0), out=buf251)
        buf252 = reinterpret_tensor(buf250, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf251, buf252, 4194304, grid=grid(4194304), stream=stream0)
        buf253 = reinterpret_tensor(buf251, (8192, 512), (512, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf252, (8192, 512), (512, 1), 0), reinterpret_tensor(arg169_1, (512, 512), (1, 512), 0), out=buf253)
        del arg169_1
        buf257 = reinterpret_tensor(buf252, (64, 128, 512), (65536, 512, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_90, hidden_states_91], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf240, buf253, arg170_1, arg171_1, arg172_1, buf257, 8192, 512, grid=grid(8192), stream=stream0)
        del arg170_1
        del arg171_1
        del arg172_1
        buf258 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (8192, 512), (512, 1), 0), reinterpret_tensor(arg173_1, (512, 512), (1, 512), 0), out=buf258)
        del arg173_1
        buf259 = reinterpret_tensor(buf240, (8192, 512), (512, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg175_1, (512, 512), (1, 512), 0), out=buf259)
        del arg175_1
        buf260 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg177_1, (512, 512), (1, 512), 0), out=buf260)
        del arg177_1
        buf261 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf258, arg174_1, buf261, 4194304, grid=grid(4194304), stream=stream0)
        del arg174_1
        buf262 = reinterpret_tensor(buf258, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf259, arg176_1, buf262, 4194304, grid=grid(4194304), stream=stream0)
        del arg176_1
        buf263 = reinterpret_tensor(buf259, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf260, arg178_1, buf263, 4194304, grid=grid(4194304), stream=stream0)
        del arg178_1
        del buf260
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf264 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf261, buf262, buf263, None, False, scale=1.0)
        buf265 = buf264[0]
        del buf264
        buf269 = reinterpret_tensor(buf263, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf265, buf269, 4194304, grid=grid(4194304), stream=stream0)
        buf270 = reinterpret_tensor(buf265, (8192, 512), (512, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (8192, 512), (512, 1), 0), reinterpret_tensor(arg179_1, (512, 512), (1, 512), 0), out=buf270)
        del arg179_1
        buf274 = reinterpret_tensor(buf269, (64, 128, 512), (65536, 512, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93, hidden_states_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf257, buf270, arg180_1, arg181_1, arg182_1, buf274, 8192, 512, grid=grid(8192), stream=stream0)
        del arg180_1
        del arg181_1
        del arg182_1
        buf275 = reinterpret_tensor(buf249, (8192, 2048), (2048, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (8192, 512), (512, 1), 0), reinterpret_tensor(arg183_1, (512, 2048), (1, 512), 0), out=buf275)
        del arg183_1
        buf276 = reinterpret_tensor(buf275, (64, 128, 2048), (262144, 2048, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_95], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf276, arg184_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg184_1
        buf277 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg185_1, (2048, 512), (1, 2048), 0), out=buf277)
        del arg185_1
        buf281 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_99, hidden_states_100], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf274, buf277, arg186_1, arg187_1, arg188_1, buf281, 8192, 512, grid=grid(8192), stream=stream0)
        del arg186_1
        del arg187_1
        del arg188_1
        buf282 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (8192, 512), (512, 1), 0), reinterpret_tensor(arg189_1, (512, 512), (1, 512), 0), out=buf282)
        del arg189_1
        buf283 = reinterpret_tensor(buf274, (8192, 512), (512, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (8192, 512), (512, 1), 0), reinterpret_tensor(arg191_1, (512, 512), (1, 512), 0), out=buf283)
        del arg191_1
        buf284 = reinterpret_tensor(buf262, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf282, arg190_1, buf284, 4194304, grid=grid(4194304), stream=stream0)
        del arg190_1
        buf285 = reinterpret_tensor(buf282, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf283, arg192_1, buf285, 4194304, grid=grid(4194304), stream=stream0)
        del arg192_1
        buf286 = reinterpret_tensor(buf276, (1024, 128, 128), (16384, 128, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf284, (1024, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf285, (1024, 32, 128), (4096, 1, 32), 0), out=buf286)
        buf290 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_31], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf286, buf290, 131072, 128, grid=grid(131072), stream=stream0)
        buf289 = reinterpret_tensor(buf285, (8192, 512), (512, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (8192, 512), (512, 1), 0), reinterpret_tensor(arg193_1, (512, 512), (1, 512), 0), out=buf289)
        del arg193_1
        buf291 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf289, arg194_1, buf291, 4194304, grid=grid(4194304), stream=stream0)
        del arg194_1
        buf292 = reinterpret_tensor(buf289, (1024, 128, 32), (4096, 32, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_31, attn_output_60], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf290, reinterpret_tensor(buf291, (1024, 128, 32), (4096, 32, 1), 0), out=buf292)
        buf293 = reinterpret_tensor(buf291, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [attn_output_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf292, buf293, 4194304, grid=grid(4194304), stream=stream0)
        buf294 = reinterpret_tensor(buf292, (8192, 512), (512, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf293, (8192, 512), (512, 1), 0), reinterpret_tensor(arg195_1, (512, 512), (1, 512), 0), out=buf294)
        del arg195_1
        buf298 = reinterpret_tensor(buf293, (64, 128, 512), (65536, 512, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_102, hidden_states_103], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf281, buf294, arg196_1, arg197_1, arg198_1, buf298, 8192, 512, grid=grid(8192), stream=stream0)
        del arg196_1
        del arg197_1
        del arg198_1
        buf299 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (8192, 512), (512, 1), 0), reinterpret_tensor(arg199_1, (512, 512), (1, 512), 0), out=buf299)
        del arg199_1
        buf300 = reinterpret_tensor(buf281, (8192, 512), (512, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg201_1, (512, 512), (1, 512), 0), out=buf300)
        del arg201_1
        buf301 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg203_1, (512, 512), (1, 512), 0), out=buf301)
        del arg203_1
        buf302 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf299, arg200_1, buf302, 4194304, grid=grid(4194304), stream=stream0)
        del arg200_1
        buf303 = reinterpret_tensor(buf299, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf300, arg202_1, buf303, 4194304, grid=grid(4194304), stream=stream0)
        del arg202_1
        buf304 = reinterpret_tensor(buf300, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf301, arg204_1, buf304, 4194304, grid=grid(4194304), stream=stream0)
        del arg204_1
        del buf301
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf305 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf302, buf303, buf304, None, False, scale=1.0)
        buf306 = buf305[0]
        del buf305
        buf310 = reinterpret_tensor(buf304, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [attn_output_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf306, buf310, 4194304, grid=grid(4194304), stream=stream0)
        buf311 = reinterpret_tensor(buf306, (8192, 512), (512, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (8192, 512), (512, 1), 0), reinterpret_tensor(arg205_1, (512, 512), (1, 512), 0), out=buf311)
        del arg205_1
        buf315 = reinterpret_tensor(buf310, (64, 128, 512), (65536, 512, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_105, hidden_states_106], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf298, buf311, arg206_1, arg207_1, arg208_1, buf315, 8192, 512, grid=grid(8192), stream=stream0)
        del arg206_1
        del arg207_1
        del arg208_1
        buf316 = reinterpret_tensor(buf290, (8192, 2048), (2048, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (8192, 512), (512, 1), 0), reinterpret_tensor(arg209_1, (512, 2048), (1, 512), 0), out=buf316)
        del arg209_1
        buf317 = reinterpret_tensor(buf316, (64, 128, 2048), (262144, 2048, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf317, arg210_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg210_1
        buf318 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg211_1, (2048, 512), (1, 2048), 0), out=buf318)
        del arg211_1
        buf322 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_112], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf315, buf318, arg212_1, arg213_1, arg214_1, buf322, 8192, 512, grid=grid(8192), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        buf323 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf322, (8192, 512), (512, 1), 0), reinterpret_tensor(arg215_1, (512, 512), (1, 512), 0), out=buf323)
        del arg215_1
        buf324 = reinterpret_tensor(buf315, (8192, 512), (512, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf322, (8192, 512), (512, 1), 0), reinterpret_tensor(arg217_1, (512, 512), (1, 512), 0), out=buf324)
        del arg217_1
        buf325 = reinterpret_tensor(buf303, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf323, arg216_1, buf325, 4194304, grid=grid(4194304), stream=stream0)
        del arg216_1
        buf326 = reinterpret_tensor(buf323, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf324, arg218_1, buf326, 4194304, grid=grid(4194304), stream=stream0)
        del arg218_1
        buf327 = reinterpret_tensor(buf317, (1024, 128, 128), (16384, 128, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (1024, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf326, (1024, 32, 128), (4096, 1, 32), 0), out=buf327)
        buf331 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_37], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf327, buf331, 131072, 128, grid=grid(131072), stream=stream0)
        buf330 = reinterpret_tensor(buf326, (8192, 512), (512, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf322, (8192, 512), (512, 1), 0), reinterpret_tensor(arg219_1, (512, 512), (1, 512), 0), out=buf330)
        del arg219_1
        buf332 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf330, arg220_1, buf332, 4194304, grid=grid(4194304), stream=stream0)
        del arg220_1
        buf333 = reinterpret_tensor(buf330, (1024, 128, 32), (4096, 32, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_37, attn_output_70], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf331, reinterpret_tensor(buf332, (1024, 128, 32), (4096, 32, 1), 0), out=buf333)
        buf334 = reinterpret_tensor(buf332, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [attn_output_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf333, buf334, 4194304, grid=grid(4194304), stream=stream0)
        buf335 = reinterpret_tensor(buf333, (8192, 512), (512, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (8192, 512), (512, 1), 0), reinterpret_tensor(arg221_1, (512, 512), (1, 512), 0), out=buf335)
        del arg221_1
        buf339 = reinterpret_tensor(buf334, (64, 128, 512), (65536, 512, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_114, hidden_states_115], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf322, buf335, arg222_1, arg223_1, arg224_1, buf339, 8192, 512, grid=grid(8192), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        buf340 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf339, (8192, 512), (512, 1), 0), reinterpret_tensor(arg225_1, (512, 512), (1, 512), 0), out=buf340)
        del arg225_1
        buf341 = reinterpret_tensor(buf322, (8192, 512), (512, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg227_1, (512, 512), (1, 512), 0), out=buf341)
        del arg227_1
        buf342 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg229_1, (512, 512), (1, 512), 0), out=buf342)
        del arg229_1
        buf343 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf340, arg226_1, buf343, 4194304, grid=grid(4194304), stream=stream0)
        del arg226_1
        buf344 = reinterpret_tensor(buf340, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf341, arg228_1, buf344, 4194304, grid=grid(4194304), stream=stream0)
        del arg228_1
        buf345 = reinterpret_tensor(buf341, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf342, arg230_1, buf345, 4194304, grid=grid(4194304), stream=stream0)
        del arg230_1
        del buf342
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf346 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf343, buf344, buf345, None, False, scale=1.0)
        buf347 = buf346[0]
        del buf346
        buf351 = reinterpret_tensor(buf345, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [attn_output_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf347, buf351, 4194304, grid=grid(4194304), stream=stream0)
        buf352 = reinterpret_tensor(buf347, (8192, 512), (512, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf351, (8192, 512), (512, 1), 0), reinterpret_tensor(arg231_1, (512, 512), (1, 512), 0), out=buf352)
        del arg231_1
        buf356 = reinterpret_tensor(buf351, (64, 128, 512), (65536, 512, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_117, hidden_states_118], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf339, buf352, arg232_1, arg233_1, arg234_1, buf356, 8192, 512, grid=grid(8192), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        buf357 = reinterpret_tensor(buf331, (8192, 2048), (2048, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (8192, 512), (512, 1), 0), reinterpret_tensor(arg235_1, (512, 2048), (1, 512), 0), out=buf357)
        del arg235_1
        buf358 = reinterpret_tensor(buf357, (64, 128, 2048), (262144, 2048, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_119], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf358, arg236_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg236_1
        buf359 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg237_1, (2048, 512), (1, 2048), 0), out=buf359)
        del arg237_1
        buf363 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_123, hidden_states_124], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf356, buf359, arg238_1, arg239_1, arg240_1, buf363, 8192, 512, grid=grid(8192), stream=stream0)
        del arg238_1
        del arg239_1
        del arg240_1
        buf364 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf363, (8192, 512), (512, 1), 0), reinterpret_tensor(arg241_1, (512, 512), (1, 512), 0), out=buf364)
        del arg241_1
        buf365 = reinterpret_tensor(buf356, (8192, 512), (512, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf363, (8192, 512), (512, 1), 0), reinterpret_tensor(arg243_1, (512, 512), (1, 512), 0), out=buf365)
        del arg243_1
        buf366 = reinterpret_tensor(buf344, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf364, arg242_1, buf366, 4194304, grid=grid(4194304), stream=stream0)
        del arg242_1
        buf367 = reinterpret_tensor(buf364, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf365, arg244_1, buf367, 4194304, grid=grid(4194304), stream=stream0)
        del arg244_1
        buf368 = reinterpret_tensor(buf358, (1024, 128, 128), (16384, 128, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf366, (1024, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf367, (1024, 32, 128), (4096, 1, 32), 0), out=buf368)
        buf372 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_43], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf368, buf372, 131072, 128, grid=grid(131072), stream=stream0)
        buf371 = reinterpret_tensor(buf367, (8192, 512), (512, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf363, (8192, 512), (512, 1), 0), reinterpret_tensor(arg245_1, (512, 512), (1, 512), 0), out=buf371)
        del arg245_1
        buf373 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf371, arg246_1, buf373, 4194304, grid=grid(4194304), stream=stream0)
        del arg246_1
        buf374 = reinterpret_tensor(buf371, (1024, 128, 32), (4096, 32, 1), 0); del buf371  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_43, attn_output_80], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf372, reinterpret_tensor(buf373, (1024, 128, 32), (4096, 32, 1), 0), out=buf374)
        buf375 = reinterpret_tensor(buf373, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [attn_output_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf374, buf375, 4194304, grid=grid(4194304), stream=stream0)
        buf376 = reinterpret_tensor(buf374, (8192, 512), (512, 1), 0); del buf374  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf375, (8192, 512), (512, 1), 0), reinterpret_tensor(arg247_1, (512, 512), (1, 512), 0), out=buf376)
        del arg247_1
        buf380 = reinterpret_tensor(buf375, (64, 128, 512), (65536, 512, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_126, hidden_states_127], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf363, buf376, arg248_1, arg249_1, arg250_1, buf380, 8192, 512, grid=grid(8192), stream=stream0)
        del arg248_1
        del arg249_1
        del arg250_1
        buf381 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf380, (8192, 512), (512, 1), 0), reinterpret_tensor(arg251_1, (512, 512), (1, 512), 0), out=buf381)
        del arg251_1
        buf382 = reinterpret_tensor(buf363, (8192, 512), (512, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg253_1, (512, 512), (1, 512), 0), out=buf382)
        del arg253_1
        buf383 = buf365; del buf365  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg255_1, (512, 512), (1, 512), 0), out=buf383)
        del arg255_1
        buf384 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf381, arg252_1, buf384, 4194304, grid=grid(4194304), stream=stream0)
        del arg252_1
        buf385 = reinterpret_tensor(buf381, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf382, arg254_1, buf385, 4194304, grid=grid(4194304), stream=stream0)
        del arg254_1
        buf386 = reinterpret_tensor(buf382, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf383, arg256_1, buf386, 4194304, grid=grid(4194304), stream=stream0)
        del arg256_1
        del buf383
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf387 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf384, buf385, buf386, None, False, scale=1.0)
        buf388 = buf387[0]
        del buf387
        buf392 = reinterpret_tensor(buf386, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [attn_output_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf388, buf392, 4194304, grid=grid(4194304), stream=stream0)
        buf393 = reinterpret_tensor(buf388, (8192, 512), (512, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (8192, 512), (512, 1), 0), reinterpret_tensor(arg257_1, (512, 512), (1, 512), 0), out=buf393)
        del arg257_1
        buf397 = reinterpret_tensor(buf392, (64, 128, 512), (65536, 512, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_129, hidden_states_130], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf380, buf393, arg258_1, arg259_1, arg260_1, buf397, 8192, 512, grid=grid(8192), stream=stream0)
        del arg258_1
        del arg259_1
        del arg260_1
        buf398 = reinterpret_tensor(buf372, (8192, 2048), (2048, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf397, (8192, 512), (512, 1), 0), reinterpret_tensor(arg261_1, (512, 2048), (1, 512), 0), out=buf398)
        del arg261_1
        buf399 = reinterpret_tensor(buf398, (64, 128, 2048), (262144, 2048, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_131], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf399, arg262_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg262_1
        buf400 = buf393; del buf393  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf399, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg263_1, (2048, 512), (1, 2048), 0), out=buf400)
        del arg263_1
        buf404 = buf380; del buf380  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_135, hidden_states_136], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf397, buf400, arg264_1, arg265_1, arg266_1, buf404, 8192, 512, grid=grid(8192), stream=stream0)
        del arg264_1
        del arg265_1
        del arg266_1
        buf405 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (8192, 512), (512, 1), 0), reinterpret_tensor(arg267_1, (512, 512), (1, 512), 0), out=buf405)
        del arg267_1
        buf406 = reinterpret_tensor(buf397, (8192, 512), (512, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (8192, 512), (512, 1), 0), reinterpret_tensor(arg269_1, (512, 512), (1, 512), 0), out=buf406)
        del arg269_1
        buf407 = reinterpret_tensor(buf385, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [contiguous_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf405, arg268_1, buf407, 4194304, grid=grid(4194304), stream=stream0)
        del arg268_1
        buf408 = reinterpret_tensor(buf405, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [key_states_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf406, arg270_1, buf408, 4194304, grid=grid(4194304), stream=stream0)
        del arg270_1
        buf409 = reinterpret_tensor(buf399, (1024, 128, 128), (16384, 128, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf407, (1024, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf408, (1024, 32, 128), (4096, 1, 32), 0), out=buf409)
        buf413 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_49], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf409, buf413, 131072, 128, grid=grid(131072), stream=stream0)
        buf412 = reinterpret_tensor(buf408, (8192, 512), (512, 1), 0); del buf408  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (8192, 512), (512, 1), 0), reinterpret_tensor(arg271_1, (512, 512), (1, 512), 0), out=buf412)
        del arg271_1
        buf414 = buf407; del buf407  # reuse
        # Topologically Sorted Source Nodes: [value_states_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf412, arg272_1, buf414, 4194304, grid=grid(4194304), stream=stream0)
        del arg272_1
        buf415 = reinterpret_tensor(buf412, (1024, 128, 32), (4096, 32, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_49, attn_output_90], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf413, reinterpret_tensor(buf414, (1024, 128, 32), (4096, 32, 1), 0), out=buf415)
        buf416 = reinterpret_tensor(buf414, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [attn_output_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf415, buf416, 4194304, grid=grid(4194304), stream=stream0)
        buf417 = reinterpret_tensor(buf415, (8192, 512), (512, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf416, (8192, 512), (512, 1), 0), reinterpret_tensor(arg273_1, (512, 512), (1, 512), 0), out=buf417)
        del arg273_1
        buf421 = reinterpret_tensor(buf416, (64, 128, 512), (65536, 512, 1), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_138, hidden_states_139], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf404, buf417, arg274_1, arg275_1, arg276_1, buf421, 8192, 512, grid=grid(8192), stream=stream0)
        del arg274_1
        del arg275_1
        del arg276_1
        buf422 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf421, (8192, 512), (512, 1), 0), reinterpret_tensor(arg277_1, (512, 512), (1, 512), 0), out=buf422)
        del arg277_1
        buf423 = reinterpret_tensor(buf404, (8192, 512), (512, 1), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg279_1, (512, 512), (1, 512), 0), out=buf423)
        del arg279_1
        buf424 = buf406; del buf406  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg281_1, (512, 512), (1, 512), 0), out=buf424)
        del arg281_1
        buf425 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf422, arg278_1, buf425, 4194304, grid=grid(4194304), stream=stream0)
        del arg278_1
        buf426 = reinterpret_tensor(buf422, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf422  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf423, arg280_1, buf426, 4194304, grid=grid(4194304), stream=stream0)
        del arg280_1
        buf427 = reinterpret_tensor(buf423, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf424, arg282_1, buf427, 4194304, grid=grid(4194304), stream=stream0)
        del arg282_1
        del buf424
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf428 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf425, buf426, buf427, None, False, scale=1.0)
        buf429 = buf428[0]
        del buf428
        buf433 = reinterpret_tensor(buf427, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [attn_output_98], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf429, buf433, 4194304, grid=grid(4194304), stream=stream0)
        buf434 = reinterpret_tensor(buf429, (8192, 512), (512, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf433, (8192, 512), (512, 1), 0), reinterpret_tensor(arg283_1, (512, 512), (1, 512), 0), out=buf434)
        del arg283_1
        buf438 = reinterpret_tensor(buf433, (64, 128, 512), (65536, 512, 1), 0); del buf433  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_141, hidden_states_142], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf421, buf434, arg284_1, arg285_1, arg286_1, buf438, 8192, 512, grid=grid(8192), stream=stream0)
        del arg284_1
        del arg285_1
        del arg286_1
        buf439 = reinterpret_tensor(buf413, (8192, 2048), (2048, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf438, (8192, 512), (512, 1), 0), reinterpret_tensor(arg287_1, (512, 2048), (1, 512), 0), out=buf439)
        del arg287_1
        buf440 = reinterpret_tensor(buf439, (64, 128, 2048), (262144, 2048, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_143], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf440, arg288_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg288_1
        buf441 = buf434; del buf434  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf440, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg289_1, (2048, 512), (1, 2048), 0), out=buf441)
        del arg289_1
        buf445 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_147, hidden_states_148], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf438, buf441, arg290_1, arg291_1, arg292_1, buf445, 8192, 512, grid=grid(8192), stream=stream0)
        del arg290_1
        del arg291_1
        del arg292_1
        buf446 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf445, (8192, 512), (512, 1), 0), reinterpret_tensor(arg293_1, (512, 512), (1, 512), 0), out=buf446)
        del arg293_1
        buf447 = reinterpret_tensor(buf438, (8192, 512), (512, 1), 0); del buf438  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf445, (8192, 512), (512, 1), 0), reinterpret_tensor(arg295_1, (512, 512), (1, 512), 0), out=buf447)
        del arg295_1
        buf448 = reinterpret_tensor(buf426, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [contiguous_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf446, arg294_1, buf448, 4194304, grid=grid(4194304), stream=stream0)
        del arg294_1
        buf449 = reinterpret_tensor(buf446, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf446  # reuse
        # Topologically Sorted Source Nodes: [key_states_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf447, arg296_1, buf449, 4194304, grid=grid(4194304), stream=stream0)
        del arg296_1
        buf450 = reinterpret_tensor(buf440, (1024, 128, 128), (16384, 128, 1), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_52], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf448, (1024, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf449, (1024, 32, 128), (4096, 1, 32), 0), out=buf450)
        buf454 = buf409; del buf409  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_55], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf450, buf454, 131072, 128, grid=grid(131072), stream=stream0)
        buf453 = reinterpret_tensor(buf449, (8192, 512), (512, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf445, (8192, 512), (512, 1), 0), reinterpret_tensor(arg297_1, (512, 512), (1, 512), 0), out=buf453)
        del arg297_1
        buf455 = buf448; del buf448  # reuse
        # Topologically Sorted Source Nodes: [value_states_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf453, arg298_1, buf455, 4194304, grid=grid(4194304), stream=stream0)
        del arg298_1
        buf456 = reinterpret_tensor(buf453, (1024, 128, 32), (4096, 32, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_55, attn_output_100], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf454, reinterpret_tensor(buf455, (1024, 128, 32), (4096, 32, 1), 0), out=buf456)
        buf457 = reinterpret_tensor(buf455, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf455  # reuse
        # Topologically Sorted Source Nodes: [attn_output_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf456, buf457, 4194304, grid=grid(4194304), stream=stream0)
        buf458 = reinterpret_tensor(buf456, (8192, 512), (512, 1), 0); del buf456  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf457, (8192, 512), (512, 1), 0), reinterpret_tensor(arg299_1, (512, 512), (1, 512), 0), out=buf458)
        del arg299_1
        buf462 = reinterpret_tensor(buf457, (64, 128, 512), (65536, 512, 1), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf445, buf458, arg300_1, arg301_1, arg302_1, buf462, 8192, 512, grid=grid(8192), stream=stream0)
        del arg300_1
        del arg301_1
        del arg302_1
        buf463 = buf458; del buf458  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf462, (8192, 512), (512, 1), 0), reinterpret_tensor(arg303_1, (512, 512), (1, 512), 0), out=buf463)
        del arg303_1
        buf464 = reinterpret_tensor(buf445, (8192, 512), (512, 1), 0); del buf445  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg305_1, (512, 512), (1, 512), 0), out=buf464)
        del arg305_1
        buf465 = buf447; del buf447  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg307_1, (512, 512), (1, 512), 0), out=buf465)
        del arg307_1
        buf466 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf463, arg304_1, buf466, 4194304, grid=grid(4194304), stream=stream0)
        del arg304_1
        buf467 = reinterpret_tensor(buf463, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf463  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf464, arg306_1, buf467, 4194304, grid=grid(4194304), stream=stream0)
        del arg306_1
        buf468 = reinterpret_tensor(buf464, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf464  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf465, arg308_1, buf468, 4194304, grid=grid(4194304), stream=stream0)
        del arg308_1
        del buf465
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf469 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf466, buf467, buf468, None, False, scale=1.0)
        buf470 = buf469[0]
        del buf469
        buf474 = reinterpret_tensor(buf468, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf468  # reuse
        # Topologically Sorted Source Nodes: [attn_output_108], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf470, buf474, 4194304, grid=grid(4194304), stream=stream0)
        buf475 = reinterpret_tensor(buf470, (8192, 512), (512, 1), 0); del buf470  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf474, (8192, 512), (512, 1), 0), reinterpret_tensor(arg309_1, (512, 512), (1, 512), 0), out=buf475)
        del arg309_1
        buf479 = reinterpret_tensor(buf474, (64, 128, 512), (65536, 512, 1), 0); del buf474  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_153, hidden_states_154], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf462, buf475, arg310_1, arg311_1, arg312_1, buf479, 8192, 512, grid=grid(8192), stream=stream0)
        del arg310_1
        del arg311_1
        del arg312_1
        buf480 = reinterpret_tensor(buf454, (8192, 2048), (2048, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf479, (8192, 512), (512, 1), 0), reinterpret_tensor(arg313_1, (512, 2048), (1, 512), 0), out=buf480)
        del arg313_1
        buf481 = reinterpret_tensor(buf480, (64, 128, 2048), (262144, 2048, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_155], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf481, arg314_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg314_1
        buf482 = buf475; del buf475  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf481, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg315_1, (2048, 512), (1, 2048), 0), out=buf482)
        del arg315_1
        buf486 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_159, hidden_states_160], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf479, buf482, arg316_1, arg317_1, arg318_1, buf486, 8192, 512, grid=grid(8192), stream=stream0)
        del arg316_1
        del arg317_1
        del arg318_1
        buf487 = buf482; del buf482  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf486, (8192, 512), (512, 1), 0), reinterpret_tensor(arg319_1, (512, 512), (1, 512), 0), out=buf487)
        del arg319_1
        buf488 = reinterpret_tensor(buf479, (8192, 512), (512, 1), 0); del buf479  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf486, (8192, 512), (512, 1), 0), reinterpret_tensor(arg321_1, (512, 512), (1, 512), 0), out=buf488)
        del arg321_1
        buf489 = reinterpret_tensor(buf467, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [contiguous_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf487, arg320_1, buf489, 4194304, grid=grid(4194304), stream=stream0)
        del arg320_1
        buf490 = reinterpret_tensor(buf487, (64, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf487  # reuse
        # Topologically Sorted Source Nodes: [key_states_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf488, arg322_1, buf490, 4194304, grid=grid(4194304), stream=stream0)
        del arg322_1
        buf491 = reinterpret_tensor(buf481, (1024, 128, 128), (16384, 128, 1), 0); del buf481  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_58], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf489, (1024, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf490, (1024, 32, 128), (4096, 1, 32), 0), out=buf491)
        buf495 = buf450; del buf450  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_61], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf491, buf495, 131072, 128, grid=grid(131072), stream=stream0)
        del buf491
        buf494 = reinterpret_tensor(buf490, (8192, 512), (512, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf486, (8192, 512), (512, 1), 0), reinterpret_tensor(arg323_1, (512, 512), (1, 512), 0), out=buf494)
        del arg323_1
        buf496 = buf489; del buf489  # reuse
        # Topologically Sorted Source Nodes: [value_states_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf494, arg324_1, buf496, 4194304, grid=grid(4194304), stream=stream0)
        del arg324_1
        buf497 = reinterpret_tensor(buf494, (1024, 128, 32), (4096, 32, 1), 0); del buf494  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_61, attn_output_110], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf495, reinterpret_tensor(buf496, (1024, 128, 32), (4096, 32, 1), 0), out=buf497)
        buf498 = reinterpret_tensor(buf496, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf496  # reuse
        # Topologically Sorted Source Nodes: [attn_output_113], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf497, buf498, 4194304, grid=grid(4194304), stream=stream0)
        buf499 = reinterpret_tensor(buf497, (8192, 512), (512, 1), 0); del buf497  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf498, (8192, 512), (512, 1), 0), reinterpret_tensor(arg325_1, (512, 512), (1, 512), 0), out=buf499)
        del arg325_1
        buf503 = reinterpret_tensor(buf498, (64, 128, 512), (65536, 512, 1), 0); del buf498  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_162, hidden_states_163], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf486, buf499, arg326_1, arg327_1, arg328_1, buf503, 8192, 512, grid=grid(8192), stream=stream0)
        del arg326_1
        del arg327_1
        del arg328_1
        buf504 = buf499; del buf499  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf503, (8192, 512), (512, 1), 0), reinterpret_tensor(arg329_1, (512, 512), (1, 512), 0), out=buf504)
        del arg329_1
        buf505 = reinterpret_tensor(buf486, (8192, 512), (512, 1), 0); del buf486  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg331_1, (512, 512), (1, 512), 0), out=buf505)
        del arg331_1
        buf506 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (8192, 512), (512, 1), 0), reinterpret_tensor(arg333_1, (512, 512), (1, 512), 0), out=buf506)
        del arg333_1
        buf507 = buf466; del buf466  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf504, arg330_1, buf507, 4194304, grid=grid(4194304), stream=stream0)
        del arg330_1
        buf508 = reinterpret_tensor(buf504, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf504  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf505, arg332_1, buf508, 4194304, grid=grid(4194304), stream=stream0)
        del arg332_1
        buf509 = reinterpret_tensor(buf505, (1, 1024, 128, 32), (4194304, 4096, 32, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf506, arg334_1, buf509, 4194304, grid=grid(4194304), stream=stream0)
        del arg334_1
        del buf506
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf510 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf507, buf508, buf509, None, False, scale=1.0)
        del buf507
        del buf508
        buf511 = buf510[0]
        del buf510
        buf515 = reinterpret_tensor(buf509, (64, 128, 16, 32), (65536, 512, 32, 1), 0); del buf509  # reuse
        # Topologically Sorted Source Nodes: [attn_output_118], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf511, buf515, 4194304, grid=grid(4194304), stream=stream0)
        buf516 = reinterpret_tensor(buf511, (8192, 512), (512, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf515, (8192, 512), (512, 1), 0), reinterpret_tensor(arg335_1, (512, 512), (1, 512), 0), out=buf516)
        del arg335_1
        buf520 = reinterpret_tensor(buf515, (64, 128, 512), (65536, 512, 1), 0); del buf515  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_165, hidden_states_166], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf503, buf516, arg336_1, arg337_1, arg338_1, buf520, 8192, 512, grid=grid(8192), stream=stream0)
        del arg336_1
        del arg337_1
        del arg338_1
        buf521 = reinterpret_tensor(buf495, (8192, 2048), (2048, 1), 0); del buf495  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf520, (8192, 512), (512, 1), 0), reinterpret_tensor(arg339_1, (512, 2048), (1, 512), 0), out=buf521)
        del arg339_1
        buf522 = reinterpret_tensor(buf521, (64, 128, 2048), (262144, 2048, 1), 0); del buf521  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_167], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_5.run(buf522, arg340_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg340_1
        buf523 = buf516; del buf516  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf522, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg341_1, (2048, 512), (1, 2048), 0), out=buf523)
        del arg341_1
        del buf522
        buf527 = buf503; del buf503  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_171, hidden_states_172], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf520, buf523, arg342_1, arg343_1, arg344_1, buf527, 8192, 512, grid=grid(8192), stream=stream0)
        del arg342_1
        del arg343_1
        del arg344_1
        del buf520
        del buf523
        buf528 = empty_strided_cuda((512, 50268), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_11.run(arg2_1, buf528, 25737216, grid=grid(25737216), stream=stream0)
        del arg2_1
        buf529 = empty_strided_cuda((8192, 50268), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf527, (8192, 512), (512, 1), 0), buf528, out=buf529)
        del buf527
        del buf528
        buf530 = empty_strided_cuda((64, 128, 50265), (6433920, 50265, 1), torch.float32)
        buf531 = empty_strided_cuda((8192, 1), (1, 8192), torch.float32)
        buf532 = empty_strided_cuda((8192, 1), (1, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [lm_logits, masked_lm_loss], Original ATen: [aten.add, aten._log_softmax]
        triton_red_fused__log_softmax_add_12.run(buf529, arg345_1, buf530, buf531, buf532, 8192, 50265, grid=grid(8192), stream=stream0)
        del arg345_1
        del buf529
        buf533 = empty_strided_cuda((), (), torch.float32)
        buf535 = buf533; del buf533  # reuse
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_13.run(buf535, arg0_1, buf530, buf531, buf532, 1, 8192, grid=grid(1), stream=stream0)
        del arg0_1
        del buf531
        del buf532
    return (buf535, buf530, buf217, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((50265, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1, 50265), (50265, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BlenderbotSmallForConditionalGeneration', benchmark_compiled_module)
