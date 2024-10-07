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


# kernel path: /tmp/torchinductor_sahanp/42/c42qj7i2s4b7mosxrmy25gwp3srv3u6mcyl4susj35tnxk4ytvts.py
# Topologically Sorted Source Nodes: [embedding, inputs_embeds, add_1, positions_1, hidden_states, hidden_states_1, hidden_states_3], Original ATen: [aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_1 => add_1
#   embedding => embedding
#   hidden_states => add_2
#   hidden_states_1 => add_3, add_4, mul_1, mul_2, rsqrt, sub, var_mean
#   hidden_states_3 => add_5, add_6, mul_3, mul_4, rsqrt_1, sub_1, var_mean_1
#   inputs_embeds => mul
#   positions_1 => embedding_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view, 1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, 1.0), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_1, 2), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %add_1), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg3_1), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg4_1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_3), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg5_1), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg6_1), kwargs = {})
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 1024
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
        tmp9 = tl.load(in_ptr2 + (2048 + r2 + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp37_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp37_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp37_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp23 = tl.load(in_ptr2 + (2048 + r2 + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
        tmp37_mean_next, tmp37_m2_next, tmp37_weight_next = triton_helpers.welford_reduce(
            tmp36, tmp37_mean, tmp37_m2, tmp37_weight, roffset == 0
        )
        tmp37_mean = tl.where(rmask, tmp37_mean_next, tmp37_mean)
        tmp37_m2 = tl.where(rmask, tmp37_m2_next, tmp37_m2)
        tmp37_weight = tl.where(rmask, tmp37_weight_next, tmp37_weight)
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp35, rmask)
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
        tmp40 = tl.load(out_ptr2 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp48 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp50 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tmp40 - tmp37
        tmp42 = 1024.0
        tmp43 = tmp38 / tmp42
        tmp44 = 1e-05
        tmp45 = tmp43 + tmp44
        tmp46 = libdevice.rsqrt(tmp45)
        tmp47 = tmp41 * tmp46
        tmp49 = tmp47 * tmp48
        tmp51 = tmp49 + tmp50
        tl.store(out_ptr5 + (r2 + (1024*x3)), tmp51, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yp/cypgobbifx5epxaifblmsdwdfsibgzyglhttyaxphslt5472flp7.py
# Topologically Sorted Source Nodes: [key_states], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   key_states => clone_1
# Graph fragment:
#   %clone_1 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536) % 16
    x3 = (xindex // 1048576)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1) + (1048576*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kb/ckbtcnh6kfwpg2dlh2csitsc7q42bxiqu3h6wqiwtonf6ckfmehu.py
# Topologically Sorted Source Nodes: [query_states_1, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_efficient_attention
#   query_states_1 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_3, %clone_1, %clone_2, %expand_4, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 1024
    x3 = xindex
    tmp0 = x0
    tmp1 = 1 + x1
    tmp2 = tmp0 < tmp1
    tmp3 = 0.0
    tmp4 = -3.4028234663852886e+38
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fp/cfpma5unrr4tvybklcuv5ymzw2zo6rg45qyd7cawmouojuf4bwu2.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_5 => add_7
#   hidden_states_6 => add_8, add_9, mul_5, mul_6, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_13), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_7, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_9), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %arg15_1), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %arg16_1), kwargs = {})
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/x2/cx27j7wrjs42e7e6bixsw2ny2fjfzbjspsc7v37ug7a7xd2wiwt7.py
# Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_7 => add_10, erf, mul_7, mul_8, mul_9
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, 0.5), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_8,), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_10), kwargs = {})
triton_poi_fused_gelu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/7q/c7qnj4a7hetrzeoydlfxeyrdz6ibfbeckhfjezu4s7e7dn56plaw.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_11, hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_11 => add_11
#   hidden_states_12 => add_12, add_13, mul_10, mul_11, rsqrt_3, sub_3, var_mean_3
#   hidden_states_5 => add_7
# Graph fragment:
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_13), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_17), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_11), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %arg21_1), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %arg22_1), kwargs = {})
triton_per_fused_add_native_layer_norm_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/qo/cqodmbnzkpgxdm23bwtimxmb2bncn376okepj7qfixwwfe3455s3.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_120, %full_default_4], 1), kwargs = {})
triton_poi_fused_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/2z/c2zdgstuy57xw2lugzkn5rnlti5ccwmgdtiybtkhfcehqvyeirqt.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   loss => amax, exp, sub_26, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_196, [1], True), kwargs = {})
#   %sub_26 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_196, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_26,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_red_fused__log_softmax_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 50265
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
        tmp0 = tl.load(in_ptr0 + (r1 + (50272*x0)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (50272*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ww/cwwe47a4uuuniwokynzlan7cuugpk3f3q5gt2sqvyntnqjdutruf.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type, div, full_default_3, ne_1, ne_2, neg, sum_2, sum_3, where_2
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_197, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_3), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_2,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_197, -100), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type), kwargs = {})
triton_red_fused_nll_loss_forward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
        tmp10 = tl.load(in_ptr1 + (tmp8 + (50272*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 1024), (1024, 1))
    assert_size_stride(arg1_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg2_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg3_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (1024, ), (1, ))
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
    assert_size_stride(arg199_1, (4, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((4, 1024, 1024), (1048576, 1024, 1), torch.float32)
        buf7 = empty_strided_cuda((4, 1024, 1024), (1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, add_1, positions_1, hidden_states, hidden_states_1, hidden_states_3], Original ATen: [aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, buf3, buf7, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg0_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        buf8 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg7_1, (1024, 1024), (1, 1024), 0), out=buf8)
        del arg7_1
        buf9 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg9_1, (1024, 1024), (1, 1024), 0), out=buf9)
        del arg9_1
        buf10 = empty_strided_cuda((4, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf9, arg10_1, buf10, 4194304, grid=grid(4194304), stream=stream0)
        del arg10_1
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg11_1, (1024, 1024), (1, 1024), 0), out=buf11)
        del arg11_1
        buf12 = reinterpret_tensor(buf7, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf11, arg12_1, buf12, 4194304, grid=grid(4194304), stream=stream0)
        del arg12_1
        buf13 = reinterpret_tensor(buf11, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [query_states_1, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused_clone_1.run(buf8, arg8_1, buf13, 4194304, grid=grid(4194304), stream=stream0)
        del arg8_1
        buf14 = empty_strided_cuda((4, 16, 1024, 1024), (16777216, 1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [query_states_1, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf14, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_1, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf13, buf10, buf12, buf14, False)
        buf16 = buf15[0]
        del buf15
        buf20 = reinterpret_tensor(buf13, (4096, 1024), (1024, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf16, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg13_1, (1024, 1024), (1, 1024), 0), out=buf20)
        del arg13_1
        buf24 = reinterpret_tensor(buf16, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf3, buf20, arg14_1, arg15_1, arg16_1, buf24, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg15_1
        del arg16_1
        buf25 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf24, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg17_1, (1024, 4096), (1, 1024), 0), out=buf25)
        del arg17_1
        buf26 = reinterpret_tensor(buf25, (4, 1024, 4096), (4194304, 4096, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf26, arg18_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg18_1
        buf27 = reinterpret_tensor(buf24, (4096, 1024), (1024, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg19_1, (4096, 1024), (1, 4096), 0), out=buf27)
        del arg19_1
        buf28 = reinterpret_tensor(buf27, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf27  # reuse
        buf32 = reinterpret_tensor(buf8, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_11, hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf28, buf3, buf20, arg14_1, arg20_1, arg21_1, arg22_1, buf32, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg14_1
        del arg20_1
        del arg21_1
        del arg22_1
        buf33 = reinterpret_tensor(buf3, (4096, 1024), (1024, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg23_1, (1024, 1024), (1, 1024), 0), out=buf33)
        del arg23_1
        buf34 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg25_1, (1024, 1024), (1, 1024), 0), out=buf34)
        del arg25_1
        buf35 = empty_strided_cuda((4, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf34, arg26_1, buf35, 4194304, grid=grid(4194304), stream=stream0)
        del arg26_1
        buf36 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg27_1, (1024, 1024), (1, 1024), 0), out=buf36)
        del arg27_1
        buf37 = reinterpret_tensor(buf32, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [value_states_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf36, arg28_1, buf37, 4194304, grid=grid(4194304), stream=stream0)
        del arg28_1
        buf38 = reinterpret_tensor(buf36, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [query_states_3, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused_clone_1.run(buf33, arg24_1, buf38, 4194304, grid=grid(4194304), stream=stream0)
        del arg24_1
        buf39 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [query_states_3, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf39, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_3, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf40 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf38, buf35, buf37, buf39, False)
        buf41 = buf40[0]
        del buf40
        buf45 = reinterpret_tensor(buf38, (4096, 1024), (1024, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf41, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg29_1, (1024, 1024), (1, 1024), 0), out=buf45)
        del arg29_1
        buf49 = reinterpret_tensor(buf41, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_14, hidden_states_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf28, buf45, arg30_1, arg31_1, arg32_1, buf49, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg31_1
        del arg32_1
        buf50 = reinterpret_tensor(buf26, (4096, 4096), (4096, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg33_1, (1024, 4096), (1, 1024), 0), out=buf50)
        del arg33_1
        buf51 = reinterpret_tensor(buf50, (4, 1024, 4096), (4194304, 4096, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf51, arg34_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg34_1
        buf52 = reinterpret_tensor(buf49, (4096, 1024), (1024, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg35_1, (4096, 1024), (1, 4096), 0), out=buf52)
        del arg35_1
        buf53 = reinterpret_tensor(buf52, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf52  # reuse
        buf57 = reinterpret_tensor(buf33, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_14, hidden_states_20, hidden_states_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf53, buf28, buf45, arg30_1, arg36_1, arg37_1, arg38_1, buf57, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg30_1
        del arg36_1
        del arg37_1
        del arg38_1
        buf58 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg39_1, (1024, 1024), (1, 1024), 0), out=buf58)
        del arg39_1
        buf59 = reinterpret_tensor(buf28, (4096, 1024), (1024, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg41_1, (1024, 1024), (1, 1024), 0), out=buf59)
        del arg41_1
        buf60 = empty_strided_cuda((4, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf59, arg42_1, buf60, 4194304, grid=grid(4194304), stream=stream0)
        del arg42_1
        buf61 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg43_1, (1024, 1024), (1, 1024), 0), out=buf61)
        del arg43_1
        buf62 = reinterpret_tensor(buf57, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [value_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf61, arg44_1, buf62, 4194304, grid=grid(4194304), stream=stream0)
        del arg44_1
        buf63 = reinterpret_tensor(buf61, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [query_states_5, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused_clone_1.run(buf58, arg40_1, buf63, 4194304, grid=grid(4194304), stream=stream0)
        del arg40_1
        buf64 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [query_states_5, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf64, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_5, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf65 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf63, buf60, buf62, buf64, False)
        buf66 = buf65[0]
        del buf65
        buf70 = reinterpret_tensor(buf63, (4096, 1024), (1024, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf66, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg45_1, (1024, 1024), (1, 1024), 0), out=buf70)
        del arg45_1
        buf74 = reinterpret_tensor(buf66, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23, hidden_states_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf53, buf70, arg46_1, arg47_1, arg48_1, buf74, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg47_1
        del arg48_1
        buf75 = reinterpret_tensor(buf51, (4096, 4096), (4096, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg49_1, (1024, 4096), (1, 1024), 0), out=buf75)
        del arg49_1
        buf76 = reinterpret_tensor(buf75, (4, 1024, 4096), (4194304, 4096, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_25], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf76, arg50_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg50_1
        buf77 = reinterpret_tensor(buf74, (4096, 1024), (1024, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg51_1, (4096, 1024), (1, 4096), 0), out=buf77)
        del arg51_1
        buf78 = reinterpret_tensor(buf77, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf77  # reuse
        buf82 = reinterpret_tensor(buf58, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23, hidden_states_29, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf78, buf53, buf70, arg46_1, arg52_1, arg53_1, arg54_1, buf82, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg46_1
        del arg52_1
        del arg53_1
        del arg54_1
        buf83 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg55_1, (1024, 1024), (1, 1024), 0), out=buf83)
        del arg55_1
        buf84 = reinterpret_tensor(buf53, (4096, 1024), (1024, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 1024), (1, 1024), 0), out=buf84)
        del arg57_1
        buf85 = empty_strided_cuda((4, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf84, arg58_1, buf85, 4194304, grid=grid(4194304), stream=stream0)
        del arg58_1
        buf86 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg59_1, (1024, 1024), (1, 1024), 0), out=buf86)
        del arg59_1
        buf87 = reinterpret_tensor(buf82, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [value_states_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf86, arg60_1, buf87, 4194304, grid=grid(4194304), stream=stream0)
        del arg60_1
        buf88 = reinterpret_tensor(buf86, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [query_states_7, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused_clone_1.run(buf83, arg56_1, buf88, 4194304, grid=grid(4194304), stream=stream0)
        del arg56_1
        buf89 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [query_states_7, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf89, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_7, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf90 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf88, buf85, buf87, buf89, False)
        buf91 = buf90[0]
        del buf90
        buf95 = reinterpret_tensor(buf88, (4096, 1024), (1024, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf91, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg61_1, (1024, 1024), (1, 1024), 0), out=buf95)
        del arg61_1
        buf99 = reinterpret_tensor(buf91, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf78, buf95, arg62_1, arg63_1, arg64_1, buf99, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg63_1
        del arg64_1
        buf100 = reinterpret_tensor(buf76, (4096, 4096), (4096, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg65_1, (1024, 4096), (1, 1024), 0), out=buf100)
        del arg65_1
        buf101 = reinterpret_tensor(buf100, (4, 1024, 4096), (4194304, 4096, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_34], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf101, arg66_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg66_1
        buf102 = reinterpret_tensor(buf99, (4096, 1024), (1024, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg67_1, (4096, 1024), (1, 4096), 0), out=buf102)
        del arg67_1
        buf103 = reinterpret_tensor(buf102, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf102  # reuse
        buf107 = reinterpret_tensor(buf83, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_38, hidden_states_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf103, buf78, buf95, arg62_1, arg68_1, arg69_1, arg70_1, buf107, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg62_1
        del arg68_1
        del arg69_1
        del arg70_1
        buf108 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 1024), (1, 1024), 0), out=buf108)
        del arg71_1
        buf109 = reinterpret_tensor(buf78, (4096, 1024), (1024, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 1024), (1, 1024), 0), out=buf109)
        del arg73_1
        buf110 = empty_strided_cuda((4, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf109, arg74_1, buf110, 4194304, grid=grid(4194304), stream=stream0)
        del arg74_1
        buf111 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg75_1, (1024, 1024), (1, 1024), 0), out=buf111)
        del arg75_1
        buf112 = reinterpret_tensor(buf107, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [value_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf111, arg76_1, buf112, 4194304, grid=grid(4194304), stream=stream0)
        del arg76_1
        buf113 = reinterpret_tensor(buf111, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [query_states_9, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused_clone_1.run(buf108, arg72_1, buf113, 4194304, grid=grid(4194304), stream=stream0)
        del arg72_1
        buf114 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [query_states_9, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf114, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_9, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf115 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf113, buf110, buf112, buf114, False)
        buf116 = buf115[0]
        del buf115
        buf120 = reinterpret_tensor(buf113, (4096, 1024), (1024, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg77_1, (1024, 1024), (1, 1024), 0), out=buf120)
        del arg77_1
        buf124 = reinterpret_tensor(buf116, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_41, hidden_states_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf103, buf120, arg78_1, arg79_1, arg80_1, buf124, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg79_1
        del arg80_1
        buf125 = reinterpret_tensor(buf101, (4096, 4096), (4096, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg81_1, (1024, 4096), (1, 1024), 0), out=buf125)
        del arg81_1
        buf126 = reinterpret_tensor(buf125, (4, 1024, 4096), (4194304, 4096, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_43], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf126, arg82_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg82_1
        buf127 = reinterpret_tensor(buf124, (4096, 1024), (1024, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg83_1, (4096, 1024), (1, 4096), 0), out=buf127)
        del arg83_1
        buf128 = reinterpret_tensor(buf127, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf127  # reuse
        buf132 = reinterpret_tensor(buf108, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_41, hidden_states_47, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf128, buf103, buf120, arg78_1, arg84_1, arg85_1, arg86_1, buf132, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg78_1
        del arg84_1
        del arg85_1
        del arg86_1
        buf133 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg87_1, (1024, 1024), (1, 1024), 0), out=buf133)
        del arg87_1
        buf134 = reinterpret_tensor(buf103, (4096, 1024), (1024, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg89_1, (1024, 1024), (1, 1024), 0), out=buf134)
        del arg89_1
        buf135 = empty_strided_cuda((4, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf134, arg90_1, buf135, 4194304, grid=grid(4194304), stream=stream0)
        del arg90_1
        buf136 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg91_1, (1024, 1024), (1, 1024), 0), out=buf136)
        del arg91_1
        buf137 = reinterpret_tensor(buf132, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [value_states_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf136, arg92_1, buf137, 4194304, grid=grid(4194304), stream=stream0)
        del arg92_1
        buf138 = reinterpret_tensor(buf136, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [query_states_11, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused_clone_1.run(buf133, arg88_1, buf138, 4194304, grid=grid(4194304), stream=stream0)
        del arg88_1
        buf139 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [query_states_11, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf139, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_11, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf140 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf138, buf135, buf137, buf139, False)
        buf141 = buf140[0]
        del buf140
        buf145 = reinterpret_tensor(buf138, (4096, 1024), (1024, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg93_1, (1024, 1024), (1, 1024), 0), out=buf145)
        del arg93_1
        buf149 = reinterpret_tensor(buf141, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_51], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf128, buf145, arg94_1, arg95_1, arg96_1, buf149, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg95_1
        del arg96_1
        buf150 = reinterpret_tensor(buf126, (4096, 4096), (4096, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg97_1, (1024, 4096), (1, 1024), 0), out=buf150)
        del arg97_1
        buf151 = reinterpret_tensor(buf150, (4, 1024, 4096), (4194304, 4096, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_52], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf151, arg98_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg98_1
        buf152 = reinterpret_tensor(buf149, (4096, 1024), (1024, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg99_1, (4096, 1024), (1, 4096), 0), out=buf152)
        del arg99_1
        buf153 = reinterpret_tensor(buf152, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf152  # reuse
        buf157 = reinterpret_tensor(buf133, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_56, hidden_states_57], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf153, buf128, buf145, arg94_1, arg100_1, arg101_1, arg102_1, buf157, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        del arg94_1
        buf158 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg103_1, (1024, 1024), (1, 1024), 0), out=buf158)
        del arg103_1
        buf159 = reinterpret_tensor(buf128, (4096, 1024), (1024, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg105_1, (1024, 1024), (1, 1024), 0), out=buf159)
        del arg105_1
        buf160 = empty_strided_cuda((4, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf159, arg106_1, buf160, 4194304, grid=grid(4194304), stream=stream0)
        del arg106_1
        buf161 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg107_1, (1024, 1024), (1, 1024), 0), out=buf161)
        del arg107_1
        buf162 = reinterpret_tensor(buf157, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [value_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf161, arg108_1, buf162, 4194304, grid=grid(4194304), stream=stream0)
        del arg108_1
        buf163 = reinterpret_tensor(buf161, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [query_states_13, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused_clone_1.run(buf158, arg104_1, buf163, 4194304, grid=grid(4194304), stream=stream0)
        del arg104_1
        buf164 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [query_states_13, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf164, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_13, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf165 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf163, buf160, buf162, buf164, False)
        buf166 = buf165[0]
        del buf165
        buf170 = reinterpret_tensor(buf163, (4096, 1024), (1024, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg109_1, (1024, 1024), (1, 1024), 0), out=buf170)
        del arg109_1
        buf174 = reinterpret_tensor(buf166, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf153, buf170, arg110_1, arg111_1, arg112_1, buf174, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg111_1
        del arg112_1
        buf175 = reinterpret_tensor(buf151, (4096, 4096), (4096, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg113_1, (1024, 4096), (1, 1024), 0), out=buf175)
        del arg113_1
        buf176 = reinterpret_tensor(buf175, (4, 1024, 4096), (4194304, 4096, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf176, arg114_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg114_1
        buf177 = reinterpret_tensor(buf174, (4096, 1024), (1024, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg115_1, (4096, 1024), (1, 4096), 0), out=buf177)
        del arg115_1
        buf178 = reinterpret_tensor(buf177, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf177  # reuse
        buf182 = reinterpret_tensor(buf158, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_65, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf178, buf153, buf170, arg110_1, arg116_1, arg117_1, arg118_1, buf182, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg110_1
        del arg116_1
        del arg117_1
        del arg118_1
        buf183 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg119_1, (1024, 1024), (1, 1024), 0), out=buf183)
        del arg119_1
        buf184 = reinterpret_tensor(buf153, (4096, 1024), (1024, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg121_1, (1024, 1024), (1, 1024), 0), out=buf184)
        del arg121_1
        buf185 = empty_strided_cuda((4, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf184, arg122_1, buf185, 4194304, grid=grid(4194304), stream=stream0)
        del arg122_1
        buf186 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg123_1, (1024, 1024), (1, 1024), 0), out=buf186)
        del arg123_1
        buf187 = reinterpret_tensor(buf182, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [value_states_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf186, arg124_1, buf187, 4194304, grid=grid(4194304), stream=stream0)
        del arg124_1
        buf188 = reinterpret_tensor(buf186, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [query_states_15, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused_clone_1.run(buf183, arg120_1, buf188, 4194304, grid=grid(4194304), stream=stream0)
        del arg120_1
        buf189 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [query_states_15, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf189, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_15, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf190 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf188, buf185, buf187, buf189, False)
        buf191 = buf190[0]
        del buf190
        buf195 = reinterpret_tensor(buf188, (4096, 1024), (1024, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg125_1, (1024, 1024), (1, 1024), 0), out=buf195)
        del arg125_1
        buf199 = reinterpret_tensor(buf191, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_68, hidden_states_69], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf178, buf195, arg126_1, arg127_1, arg128_1, buf199, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg127_1
        del arg128_1
        buf200 = reinterpret_tensor(buf176, (4096, 4096), (4096, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg129_1, (1024, 4096), (1, 1024), 0), out=buf200)
        del arg129_1
        buf201 = reinterpret_tensor(buf200, (4, 1024, 4096), (4194304, 4096, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf201, arg130_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg130_1
        buf202 = reinterpret_tensor(buf199, (4096, 1024), (1024, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg131_1, (4096, 1024), (1, 4096), 0), out=buf202)
        del arg131_1
        buf203 = reinterpret_tensor(buf202, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf202  # reuse
        buf207 = reinterpret_tensor(buf183, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_68, hidden_states_74, hidden_states_75], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf203, buf178, buf195, arg126_1, arg132_1, arg133_1, arg134_1, buf207, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg126_1
        del arg132_1
        del arg133_1
        del arg134_1
        buf208 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg135_1, (1024, 1024), (1, 1024), 0), out=buf208)
        del arg135_1
        buf209 = reinterpret_tensor(buf178, (4096, 1024), (1024, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 1024), (1, 1024), 0), out=buf209)
        del arg137_1
        buf210 = empty_strided_cuda((4, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf209, arg138_1, buf210, 4194304, grid=grid(4194304), stream=stream0)
        del arg138_1
        buf211 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg139_1, (1024, 1024), (1, 1024), 0), out=buf211)
        del arg139_1
        buf212 = reinterpret_tensor(buf207, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [value_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf211, arg140_1, buf212, 4194304, grid=grid(4194304), stream=stream0)
        del arg140_1
        buf213 = reinterpret_tensor(buf211, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [query_states_17, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused_clone_1.run(buf208, arg136_1, buf213, 4194304, grid=grid(4194304), stream=stream0)
        del arg136_1
        buf214 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [query_states_17, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf214, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_17, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf215 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf213, buf210, buf212, buf214, False)
        buf216 = buf215[0]
        del buf215
        buf220 = reinterpret_tensor(buf213, (4096, 1024), (1024, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf216, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg141_1, (1024, 1024), (1, 1024), 0), out=buf220)
        del arg141_1
        buf224 = reinterpret_tensor(buf216, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77, hidden_states_78], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf203, buf220, arg142_1, arg143_1, arg144_1, buf224, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg143_1
        del arg144_1
        buf225 = reinterpret_tensor(buf201, (4096, 4096), (4096, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf224, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg145_1, (1024, 4096), (1, 1024), 0), out=buf225)
        del arg145_1
        buf226 = reinterpret_tensor(buf225, (4, 1024, 4096), (4194304, 4096, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_79], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf226, arg146_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg146_1
        buf227 = reinterpret_tensor(buf224, (4096, 1024), (1024, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg147_1, (4096, 1024), (1, 4096), 0), out=buf227)
        del arg147_1
        buf228 = reinterpret_tensor(buf227, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf227  # reuse
        buf232 = reinterpret_tensor(buf208, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77, hidden_states_83, hidden_states_84], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf228, buf203, buf220, arg142_1, arg148_1, arg149_1, arg150_1, buf232, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg142_1
        del arg148_1
        del arg149_1
        del arg150_1
        buf233 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg151_1, (1024, 1024), (1, 1024), 0), out=buf233)
        del arg151_1
        buf234 = reinterpret_tensor(buf203, (4096, 1024), (1024, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg153_1, (1024, 1024), (1, 1024), 0), out=buf234)
        del arg153_1
        buf235 = empty_strided_cuda((4, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf234, arg154_1, buf235, 4194304, grid=grid(4194304), stream=stream0)
        del arg154_1
        buf236 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg155_1, (1024, 1024), (1, 1024), 0), out=buf236)
        del arg155_1
        buf237 = reinterpret_tensor(buf232, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [value_states_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf236, arg156_1, buf237, 4194304, grid=grid(4194304), stream=stream0)
        del arg156_1
        buf238 = reinterpret_tensor(buf236, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [query_states_19, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused_clone_1.run(buf233, arg152_1, buf238, 4194304, grid=grid(4194304), stream=stream0)
        del arg152_1
        buf239 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [query_states_19, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf239, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_19, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf240 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf238, buf235, buf237, buf239, False)
        buf241 = buf240[0]
        del buf240
        buf245 = reinterpret_tensor(buf238, (4096, 1024), (1024, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf241, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg157_1, (1024, 1024), (1, 1024), 0), out=buf245)
        del arg157_1
        buf249 = reinterpret_tensor(buf241, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_86, hidden_states_87], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf228, buf245, arg158_1, arg159_1, arg160_1, buf249, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg159_1
        del arg160_1
        buf250 = reinterpret_tensor(buf226, (4096, 4096), (4096, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg161_1, (1024, 4096), (1, 1024), 0), out=buf250)
        del arg161_1
        buf251 = reinterpret_tensor(buf250, (4, 1024, 4096), (4194304, 4096, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf251, arg162_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg162_1
        buf252 = reinterpret_tensor(buf249, (4096, 1024), (1024, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf251, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg163_1, (4096, 1024), (1, 4096), 0), out=buf252)
        del arg163_1
        buf253 = reinterpret_tensor(buf252, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf252  # reuse
        buf257 = reinterpret_tensor(buf233, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_86, hidden_states_92, hidden_states_93], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf253, buf228, buf245, arg158_1, arg164_1, arg165_1, arg166_1, buf257, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg158_1
        del arg164_1
        del arg165_1
        del arg166_1
        buf258 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg167_1, (1024, 1024), (1, 1024), 0), out=buf258)
        del arg167_1
        buf259 = reinterpret_tensor(buf228, (4096, 1024), (1024, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg169_1, (1024, 1024), (1, 1024), 0), out=buf259)
        del arg169_1
        buf260 = empty_strided_cuda((4, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf259, arg170_1, buf260, 4194304, grid=grid(4194304), stream=stream0)
        del arg170_1
        buf261 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 1024), (1, 1024), 0), out=buf261)
        del arg171_1
        buf262 = reinterpret_tensor(buf257, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [value_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf261, arg172_1, buf262, 4194304, grid=grid(4194304), stream=stream0)
        del arg172_1
        buf263 = reinterpret_tensor(buf261, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [query_states_21, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused_clone_1.run(buf258, arg168_1, buf263, 4194304, grid=grid(4194304), stream=stream0)
        del arg168_1
        buf264 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [query_states_21, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf264, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_21, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf265 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf263, buf260, buf262, buf264, False)
        buf266 = buf265[0]
        del buf265
        buf270 = reinterpret_tensor(buf263, (4096, 1024), (1024, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf266, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg173_1, (1024, 1024), (1, 1024), 0), out=buf270)
        del arg173_1
        buf274 = reinterpret_tensor(buf266, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_95, hidden_states_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf253, buf270, arg174_1, arg175_1, arg176_1, buf274, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg175_1
        del arg176_1
        buf275 = reinterpret_tensor(buf251, (4096, 4096), (4096, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg177_1, (1024, 4096), (1, 1024), 0), out=buf275)
        del arg177_1
        buf276 = reinterpret_tensor(buf275, (4, 1024, 4096), (4194304, 4096, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_97], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf276, arg178_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg178_1
        buf277 = reinterpret_tensor(buf274, (4096, 1024), (1024, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg179_1, (4096, 1024), (1, 4096), 0), out=buf277)
        del arg179_1
        buf278 = reinterpret_tensor(buf277, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf277  # reuse
        buf282 = reinterpret_tensor(buf258, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_95, hidden_states_101, hidden_states_102], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf278, buf253, buf270, arg174_1, arg180_1, arg181_1, arg182_1, buf282, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg174_1
        del arg180_1
        del arg181_1
        del arg182_1
        buf283 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg183_1, (1024, 1024), (1, 1024), 0), out=buf283)
        del arg183_1
        buf284 = reinterpret_tensor(buf253, (4096, 1024), (1024, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg185_1, (1024, 1024), (1, 1024), 0), out=buf284)
        del arg185_1
        buf285 = empty_strided_cuda((4, 16, 1024, 64), (1048576, 65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf284, arg186_1, buf285, 4194304, grid=grid(4194304), stream=stream0)
        del arg186_1
        buf286 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg187_1, (1024, 1024), (1, 1024), 0), out=buf286)
        del arg187_1
        buf287 = reinterpret_tensor(buf282, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [value_states_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf286, arg188_1, buf287, 4194304, grid=grid(4194304), stream=stream0)
        del arg188_1
        buf288 = reinterpret_tensor(buf286, (4, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [query_states_23, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused_clone_1.run(buf283, arg184_1, buf288, 4194304, grid=grid(4194304), stream=stream0)
        del arg184_1
        buf289 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [query_states_23, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf289, 67108864, grid=grid(67108864), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_23, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf290 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf288, buf285, buf287, buf289, False)
        del buf289
        buf291 = buf290[0]
        del buf290
        buf295 = reinterpret_tensor(buf288, (4096, 1024), (1024, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf291, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg189_1, (1024, 1024), (1, 1024), 0), out=buf295)
        del arg189_1
        buf299 = reinterpret_tensor(buf291, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_104, hidden_states_105], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf278, buf295, arg190_1, arg191_1, arg192_1, buf299, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg191_1
        del arg192_1
        buf300 = reinterpret_tensor(buf276, (4096, 4096), (4096, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf299, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg193_1, (1024, 4096), (1, 1024), 0), out=buf300)
        del arg193_1
        buf301 = reinterpret_tensor(buf300, (4, 1024, 4096), (4194304, 4096, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_106], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf301, arg194_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg194_1
        buf302 = reinterpret_tensor(buf299, (4096, 1024), (1024, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf301, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg195_1, (4096, 1024), (1, 4096), 0), out=buf302)
        del arg195_1
        del buf301
        buf303 = reinterpret_tensor(buf302, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf302  # reuse
        buf307 = reinterpret_tensor(buf283, (4, 1024, 1024), (1048576, 1024, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_104, hidden_states_110, hidden_states_111], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf303, buf278, buf295, arg190_1, arg196_1, arg197_1, arg198_1, buf307, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg190_1
        del arg196_1
        del arg197_1
        del arg198_1
        del buf278
        del buf295
        del buf303
        buf308 = empty_strided_cuda((1024, 50268), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(arg1_1, buf308, 51474432, grid=grid(51474432), stream=stream0)
        del arg1_1
        buf309 = empty_strided_cuda((4096, 50268), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf307, (4096, 1024), (1024, 1), 0), buf308, out=buf309)
        del buf307
        del buf308
        buf310 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        buf311 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_7.run(buf309, buf310, buf311, 4096, 50265, grid=grid(4096), stream=stream0)
        buf312 = empty_strided_cuda((), (), torch.float32)
        buf314 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_8.run(buf314, arg199_1, buf309, buf310, buf311, 1, 4096, grid=grid(1), stream=stream0)
        del arg199_1
        del buf310
        del buf311
    return (buf314, reinterpret_tensor(buf309, (4, 1024, 50265), (51478528, 50272, 1), 0), buf10, buf12, buf35, buf37, buf60, buf62, buf85, buf87, buf110, buf112, buf135, buf137, buf160, buf162, buf185, buf187, buf210, buf212, buf235, buf237, buf260, buf262, buf285, buf287, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg199_1 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MBartForCausalLM', benchmark_compiled_module)
