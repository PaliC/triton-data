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


# kernel path: /tmp/torchinductor_sahanp/74/c74gvnhshtk5no2ktcdmaeuh4o5abwuz4rqaituzxzahqbdtjd2o.py
# Topologically Sorted Source Nodes: [ne, mask, cumsum], Original ATen: [aten.ne, aten._to_copy, aten.cumsum]
# Source node to ATen node mapping:
#   cumsum => cumsum
#   mask => convert_element_type
#   ne => ne
# Graph fragment:
#   %ne : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view, 1), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne, torch.int32), kwargs = {})
#   %cumsum : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%convert_element_type, 1), kwargs = {})
triton_per_fused__to_copy_cumsum_ne_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints=[16, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_cumsum_ne_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int32)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tmp4.to(tl.int64)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp7, = tl.associative_scan((tmp6,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (r1 + (128*x0)), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/i6/ci6wfra4uwjszcfamfjqegxjoprcua4agf2ilfmnssibkmkqviuf.py
# Topologically Sorted Source Nodes: [embedding, inputs_embeds, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   hidden_states => add_2
#   hidden_states_2 => add_3, add_4, mul_2, mul_3, rsqrt, sub, var_mean
#   inputs_embeds => mul
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %view, 1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, 32.0), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %view_2), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg4_1), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg5_1), kwargs = {})
triton_per_fused_add_embedding_mul_native_layer_norm_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mul_native_layer_norm_1', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, xnumel, rnumel):
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
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.full([RBLOCK], 128112, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 128112), "index out of bounds: 0 <= tmp4 < 128112")
    tmp6 = tl.load(in_ptr1 + (r1 + (1024*tmp4)), None)
    tmp7 = 32.0
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.int32)
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full([1], 1, tl.int64)
    tmp14 = tmp0 != tmp13
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tmp12 * tmp15
    tmp17 = tmp16.to(tl.int64)
    tmp18 = tmp17 + tmp13
    tmp19 = tl.full([RBLOCK], 1026, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 1026), "index out of bounds: 0 <= tmp22 < 1026")
    tmp24 = tl.load(in_ptr3 + (r1 + (1024*tmp22)), None)
    tmp25 = tmp8 + tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = tl.full([1], 1024, tl.int32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 / tmp32
    tmp34 = tmp26 - tmp33
    tmp35 = tmp34 * tmp34
    tmp36 = tl.broadcast_to(tmp35, [RBLOCK])
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp39 = tmp25 - tmp33
    tmp40 = 1024.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-05
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(out_ptr0 + (r1 + (1024*x0)), tmp25, None)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp49, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yk/cykua262qzmdt7hw64uxidiheb5zr26bzqbiaebcge4x5oujun4g.py
# Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_efficient_attention
#   key_states => clone_1
#   query_states_1 => clone_3
#   value_states => clone_2
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_3, %clone_1, %clone_2, None, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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


# kernel path: /tmp/torchinductor_sahanp/fj/cfjfyjdfmw7byrqeh3oii7s6p2hkogxsdbbgbc3mwtxrnh6ja22o.py
# Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_5], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_4 => add_5
#   hidden_states_5 => add_6, add_7, mul_4, mul_5, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_14), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_7), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg14_1), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg15_1), kwargs = {})
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6z/c6zisne5mzzexvlwasnoutnazvr2r2nqgunhetx6ghw3uunay4se.py
# Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   hidden_states_6 => relu
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_16,), kwargs = {})
triton_poi_fused_relu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pd/cpdjfgz6ext2oqfrwhkomkaxzwgddf7mj6la3rnlag2tbn2lnkxc.py
# Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_10, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_10 => add_8
#   hidden_states_11 => add_10, add_9, mul_6, mul_7, rsqrt_2, sub_2, var_mean_2
#   hidden_states_4 => add_5
# Graph fragment:
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_14), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_18), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_9), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %arg20_1), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %arg21_1), kwargs = {})
triton_per_fused_add_native_layer_norm_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, None)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4u/c4u2xdvohtw75r7dkalkvszljecvvc7ku2fpdjm5uruct6up4az2.py
# Topologically Sorted Source Nodes: [query_states_25, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output_48 => _scaled_dot_product_efficient_attention_12
#   query_states_25 => clone_76
# Graph fragment:
#   %clone_76 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_125,), kwargs = {memory_format: torch.contiguous_format})
#   %_scaled_dot_product_efficient_attention_12 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_76, %clone_74, %clone_75, %expand_3, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128) % 128
    x3 = xindex
    tmp0 = x0
    tmp1 = 1 + x1
    tmp2 = tmp0 < tmp1
    tmp3 = 0.0
    tmp4 = -3.4028234663852886e+38
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/x6/cx6w32pfcz3u4w6zcxjg6c7yd2yq3nt4w4hbesswhz57ijfzbgwg.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   masked_lm_loss => amax, exp, sub_62, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_537, [1], True), kwargs = {})
#   %sub_62 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_537, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_62,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_red_fused__log_softmax_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 131072],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128112
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
        tmp0 = tl.load(in_ptr0 + (r1 + (128128*x0)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (128128*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/k4/ck4m4kcje4mxetlxnlhldc272sewuzpgn7zcpav7ieevkv2ppf6t.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   masked_lm_loss => convert_element_type_6, div, full_default_3, ne_3, ne_4, neg, sum_2, sum_3, where_2
# Graph fragment:
#   %ne_3 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_538, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %neg, %full_default_3), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_2,), kwargs = {})
#   %ne_4 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_538, -100), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_4,), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type_6), kwargs = {})
triton_red_fused_nll_loss_forward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
        tmp5 = tl.full([XBLOCK, RBLOCK], 128112, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 128112)) | ~(rmask), "index out of bounds: 0 <= tmp8 < 128112")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (128128*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 128), (128, 1))
    assert_size_stride(arg1_1, (16, 128), (128, 1))
    assert_size_stride(arg2_1, (128112, 1024), (1024, 1))
    assert_size_stride(arg3_1, (1026, 1024), (1024, 1))
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
    assert_size_stride(arg198_1, (1026, 1024), (1024, 1))
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
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 128), (128, 1), torch.int64)
        # Topologically Sorted Source Nodes: [ne, mask, cumsum], Original ATen: [aten.ne, aten._to_copy, aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_cumsum_ne_0.run(arg1_1, buf0, 16, 128, grid=grid(16), stream=stream0)
        buf1 = empty_strided_cuda((16, 128, 1024), (131072, 1024, 1), torch.float32)
        buf5 = empty_strided_cuda((16, 128, 1024), (131072, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_embedding_mul_native_layer_norm_1.run(arg1_1, arg2_1, buf0, arg3_1, arg4_1, arg5_1, buf1, buf5, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg3_1
        del arg4_1
        del arg5_1
        buf6 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf5, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg6_1, (1024, 1024), (1, 1024), 0), out=buf6)
        del arg6_1
        buf7 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf5, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg8_1, (1024, 1024), (1, 1024), 0), out=buf7)
        del arg8_1
        buf8 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf5, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg10_1, (1024, 1024), (1, 1024), 0), out=buf8)
        del arg10_1
        buf9 = reinterpret_tensor(buf5, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf6, arg7_1, buf9, 2097152, grid=grid(2097152), stream=stream0)
        del arg7_1
        buf10 = reinterpret_tensor(buf6, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf7, arg9_1, buf10, 2097152, grid=grid(2097152), stream=stream0)
        del arg9_1
        buf11 = reinterpret_tensor(buf7, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf8, arg11_1, buf11, 2097152, grid=grid(2097152), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [query_states_1, key_states, value_states, attn_output], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf9, buf10, buf11, None, False)
        buf13 = buf12[0]
        del buf12
        buf17 = reinterpret_tensor(buf9, (2048, 1024), (1024, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf13, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg12_1, (1024, 1024), (1, 1024), 0), out=buf17)
        del arg12_1
        buf21 = reinterpret_tensor(buf13, (16, 128, 1024), (131072, 1024, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf1, buf17, arg13_1, arg14_1, arg15_1, buf21, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg14_1
        del arg15_1
        buf22 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg16_1, (1024, 4096), (1, 1024), 0), out=buf22)
        del arg16_1
        buf23 = reinterpret_tensor(buf22, (16, 128, 4096), (524288, 4096, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf23, arg17_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg17_1
        buf24 = reinterpret_tensor(buf21, (2048, 1024), (1024, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg18_1, (4096, 1024), (1, 4096), 0), out=buf24)
        del arg18_1
        buf25 = reinterpret_tensor(buf24, (16, 128, 1024), (131072, 1024, 1), 0); del buf24  # reuse
        buf29 = reinterpret_tensor(buf11, (16, 128, 1024), (131072, 1024, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_10, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf25, buf1, buf17, arg13_1, arg19_1, arg20_1, arg21_1, buf29, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg13_1
        del arg19_1
        del arg20_1
        del arg21_1
        buf30 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg22_1, (1024, 1024), (1, 1024), 0), out=buf30)
        del arg22_1
        buf31 = reinterpret_tensor(buf1, (2048, 1024), (1024, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg24_1, (1024, 1024), (1, 1024), 0), out=buf31)
        del arg24_1
        buf32 = reinterpret_tensor(buf10, (2048, 1024), (1024, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg26_1, (1024, 1024), (1, 1024), 0), out=buf32)
        del arg26_1
        buf33 = reinterpret_tensor(buf29, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [query_states_3, key_states_1, value_states_1, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf30, arg23_1, buf33, 2097152, grid=grid(2097152), stream=stream0)
        del arg23_1
        buf34 = reinterpret_tensor(buf30, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [query_states_3, key_states_1, value_states_1, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf31, arg25_1, buf34, 2097152, grid=grid(2097152), stream=stream0)
        del arg25_1
        buf35 = reinterpret_tensor(buf31, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [query_states_3, key_states_1, value_states_1, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf32, arg27_1, buf35, 2097152, grid=grid(2097152), stream=stream0)
        del arg27_1
        # Topologically Sorted Source Nodes: [query_states_3, key_states_1, value_states_1, attn_output_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf36 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf33, buf34, buf35, None, False)
        buf37 = buf36[0]
        del buf36
        buf41 = reinterpret_tensor(buf35, (2048, 1024), (1024, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf37, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg28_1, (1024, 1024), (1, 1024), 0), out=buf41)
        del arg28_1
        buf45 = reinterpret_tensor(buf37, (16, 128, 1024), (131072, 1024, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf25, buf41, arg29_1, arg30_1, arg31_1, buf45, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg30_1
        del arg31_1
        buf46 = reinterpret_tensor(buf23, (2048, 4096), (4096, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf45, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg32_1, (1024, 4096), (1, 1024), 0), out=buf46)
        del arg32_1
        buf47 = reinterpret_tensor(buf46, (16, 128, 4096), (524288, 4096, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf47, arg33_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg33_1
        buf48 = reinterpret_tensor(buf45, (2048, 1024), (1024, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg34_1, (4096, 1024), (1, 4096), 0), out=buf48)
        del arg34_1
        buf49 = reinterpret_tensor(buf48, (16, 128, 1024), (131072, 1024, 1), 0); del buf48  # reuse
        buf53 = reinterpret_tensor(buf34, (16, 128, 1024), (131072, 1024, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_19, hidden_states_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf49, buf25, buf41, arg29_1, arg35_1, arg36_1, arg37_1, buf53, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg29_1
        del arg35_1
        del arg36_1
        del arg37_1
        buf54 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg38_1, (1024, 1024), (1, 1024), 0), out=buf54)
        del arg38_1
        buf55 = reinterpret_tensor(buf25, (2048, 1024), (1024, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg40_1, (1024, 1024), (1, 1024), 0), out=buf55)
        del arg40_1
        buf56 = reinterpret_tensor(buf33, (2048, 1024), (1024, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg42_1, (1024, 1024), (1, 1024), 0), out=buf56)
        del arg42_1
        buf57 = reinterpret_tensor(buf53, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [query_states_5, key_states_2, value_states_2, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf54, arg39_1, buf57, 2097152, grid=grid(2097152), stream=stream0)
        del arg39_1
        buf58 = reinterpret_tensor(buf54, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [query_states_5, key_states_2, value_states_2, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf55, arg41_1, buf58, 2097152, grid=grid(2097152), stream=stream0)
        del arg41_1
        buf59 = reinterpret_tensor(buf55, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [query_states_5, key_states_2, value_states_2, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf56, arg43_1, buf59, 2097152, grid=grid(2097152), stream=stream0)
        del arg43_1
        # Topologically Sorted Source Nodes: [query_states_5, key_states_2, value_states_2, attn_output_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf60 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf57, buf58, buf59, None, False)
        buf61 = buf60[0]
        del buf60
        buf65 = reinterpret_tensor(buf59, (2048, 1024), (1024, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg44_1, (1024, 1024), (1, 1024), 0), out=buf65)
        del arg44_1
        buf69 = reinterpret_tensor(buf61, (16, 128, 1024), (131072, 1024, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_22, hidden_states_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf49, buf65, arg45_1, arg46_1, arg47_1, buf69, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg46_1
        del arg47_1
        buf70 = reinterpret_tensor(buf47, (2048, 4096), (4096, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg48_1, (1024, 4096), (1, 1024), 0), out=buf70)
        del arg48_1
        buf71 = reinterpret_tensor(buf70, (16, 128, 4096), (524288, 4096, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf71, arg49_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg49_1
        buf72 = reinterpret_tensor(buf69, (2048, 1024), (1024, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg50_1, (4096, 1024), (1, 4096), 0), out=buf72)
        del arg50_1
        buf73 = reinterpret_tensor(buf72, (16, 128, 1024), (131072, 1024, 1), 0); del buf72  # reuse
        buf77 = reinterpret_tensor(buf58, (16, 128, 1024), (131072, 1024, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_22, hidden_states_28, hidden_states_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf73, buf49, buf65, arg45_1, arg51_1, arg52_1, arg53_1, buf77, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg45_1
        del arg51_1
        del arg52_1
        del arg53_1
        buf78 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg54_1, (1024, 1024), (1, 1024), 0), out=buf78)
        del arg54_1
        buf79 = reinterpret_tensor(buf49, (2048, 1024), (1024, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg56_1, (1024, 1024), (1, 1024), 0), out=buf79)
        del arg56_1
        buf80 = reinterpret_tensor(buf57, (2048, 1024), (1024, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg58_1, (1024, 1024), (1, 1024), 0), out=buf80)
        del arg58_1
        buf81 = reinterpret_tensor(buf77, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [query_states_7, key_states_3, value_states_3, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf78, arg55_1, buf81, 2097152, grid=grid(2097152), stream=stream0)
        del arg55_1
        buf82 = reinterpret_tensor(buf78, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [query_states_7, key_states_3, value_states_3, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf79, arg57_1, buf82, 2097152, grid=grid(2097152), stream=stream0)
        del arg57_1
        buf83 = reinterpret_tensor(buf79, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [query_states_7, key_states_3, value_states_3, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf80, arg59_1, buf83, 2097152, grid=grid(2097152), stream=stream0)
        del arg59_1
        # Topologically Sorted Source Nodes: [query_states_7, key_states_3, value_states_3, attn_output_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf84 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf81, buf82, buf83, None, False)
        buf85 = buf84[0]
        del buf84
        buf89 = reinterpret_tensor(buf83, (2048, 1024), (1024, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf85, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg60_1, (1024, 1024), (1, 1024), 0), out=buf89)
        del arg60_1
        buf93 = reinterpret_tensor(buf85, (16, 128, 1024), (131072, 1024, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf73, buf89, arg61_1, arg62_1, arg63_1, buf93, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg62_1
        del arg63_1
        buf94 = reinterpret_tensor(buf71, (2048, 4096), (4096, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg64_1, (1024, 4096), (1, 1024), 0), out=buf94)
        del arg64_1
        buf95 = reinterpret_tensor(buf94, (16, 128, 4096), (524288, 4096, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_33], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf95, arg65_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg65_1
        buf96 = reinterpret_tensor(buf93, (2048, 1024), (1024, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg66_1, (4096, 1024), (1, 4096), 0), out=buf96)
        del arg66_1
        buf97 = reinterpret_tensor(buf96, (16, 128, 1024), (131072, 1024, 1), 0); del buf96  # reuse
        buf101 = reinterpret_tensor(buf82, (16, 128, 1024), (131072, 1024, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_37, hidden_states_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf97, buf73, buf89, arg61_1, arg67_1, arg68_1, arg69_1, buf101, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg61_1
        del arg67_1
        del arg68_1
        del arg69_1
        buf102 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg70_1, (1024, 1024), (1, 1024), 0), out=buf102)
        del arg70_1
        buf103 = reinterpret_tensor(buf73, (2048, 1024), (1024, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg72_1, (1024, 1024), (1, 1024), 0), out=buf103)
        del arg72_1
        buf104 = reinterpret_tensor(buf81, (2048, 1024), (1024, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg74_1, (1024, 1024), (1, 1024), 0), out=buf104)
        del arg74_1
        buf105 = reinterpret_tensor(buf101, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [query_states_9, key_states_4, value_states_4, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf102, arg71_1, buf105, 2097152, grid=grid(2097152), stream=stream0)
        del arg71_1
        buf106 = reinterpret_tensor(buf102, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [query_states_9, key_states_4, value_states_4, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf103, arg73_1, buf106, 2097152, grid=grid(2097152), stream=stream0)
        del arg73_1
        buf107 = reinterpret_tensor(buf103, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [query_states_9, key_states_4, value_states_4, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf104, arg75_1, buf107, 2097152, grid=grid(2097152), stream=stream0)
        del arg75_1
        # Topologically Sorted Source Nodes: [query_states_9, key_states_4, value_states_4, attn_output_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf108 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf105, buf106, buf107, None, False)
        buf109 = buf108[0]
        del buf108
        buf113 = reinterpret_tensor(buf107, (2048, 1024), (1024, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf109, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg76_1, (1024, 1024), (1, 1024), 0), out=buf113)
        del arg76_1
        buf117 = reinterpret_tensor(buf109, (16, 128, 1024), (131072, 1024, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf97, buf113, arg77_1, arg78_1, arg79_1, buf117, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg78_1
        del arg79_1
        buf118 = reinterpret_tensor(buf95, (2048, 4096), (4096, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf117, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg80_1, (1024, 4096), (1, 1024), 0), out=buf118)
        del arg80_1
        buf119 = reinterpret_tensor(buf118, (16, 128, 4096), (524288, 4096, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf119, arg81_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg81_1
        buf120 = reinterpret_tensor(buf117, (2048, 1024), (1024, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg82_1, (4096, 1024), (1, 4096), 0), out=buf120)
        del arg82_1
        buf121 = reinterpret_tensor(buf120, (16, 128, 1024), (131072, 1024, 1), 0); del buf120  # reuse
        buf125 = reinterpret_tensor(buf106, (16, 128, 1024), (131072, 1024, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_46, hidden_states_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf121, buf97, buf113, arg77_1, arg83_1, arg84_1, arg85_1, buf125, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg77_1
        del arg83_1
        del arg84_1
        del arg85_1
        buf126 = reinterpret_tensor(buf97, (2048, 1024), (1024, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf125, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg86_1, (1024, 1024), (1, 1024), 0), out=buf126)
        del arg86_1
        buf127 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf125, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg88_1, (1024, 1024), (1, 1024), 0), out=buf127)
        del arg88_1
        buf128 = reinterpret_tensor(buf105, (2048, 1024), (1024, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf125, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg90_1, (1024, 1024), (1, 1024), 0), out=buf128)
        del arg90_1
        buf129 = reinterpret_tensor(buf125, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [query_states_11, key_states_5, value_states_5, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf126, arg87_1, buf129, 2097152, grid=grid(2097152), stream=stream0)
        del arg87_1
        buf130 = reinterpret_tensor(buf126, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [query_states_11, key_states_5, value_states_5, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf127, arg89_1, buf130, 2097152, grid=grid(2097152), stream=stream0)
        del arg89_1
        buf131 = reinterpret_tensor(buf127, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [query_states_11, key_states_5, value_states_5, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf128, arg91_1, buf131, 2097152, grid=grid(2097152), stream=stream0)
        del arg91_1
        # Topologically Sorted Source Nodes: [query_states_11, key_states_5, value_states_5, attn_output_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf132 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf129, buf130, buf131, None, False)
        buf133 = buf132[0]
        del buf132
        buf137 = reinterpret_tensor(buf131, (2048, 1024), (1024, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf133, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg92_1, (1024, 1024), (1, 1024), 0), out=buf137)
        del arg92_1
        buf141 = reinterpret_tensor(buf133, (16, 128, 1024), (131072, 1024, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf121, buf137, arg93_1, arg94_1, arg95_1, buf141, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg94_1
        del arg95_1
        buf142 = reinterpret_tensor(buf119, (2048, 4096), (4096, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg96_1, (1024, 4096), (1, 1024), 0), out=buf142)
        del arg96_1
        buf143 = reinterpret_tensor(buf142, (16, 128, 4096), (524288, 4096, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf143, arg97_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg97_1
        buf144 = reinterpret_tensor(buf141, (2048, 1024), (1024, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg98_1, (4096, 1024), (1, 4096), 0), out=buf144)
        del arg98_1
        buf145 = reinterpret_tensor(buf144, (16, 128, 1024), (131072, 1024, 1), 0); del buf144  # reuse
        buf149 = reinterpret_tensor(buf130, (16, 128, 1024), (131072, 1024, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_55, hidden_states_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf145, buf121, buf137, arg93_1, arg99_1, arg100_1, arg101_1, buf149, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg100_1
        del arg101_1
        del arg93_1
        del arg99_1
        buf150 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg102_1, (1024, 1024), (1, 1024), 0), out=buf150)
        del arg102_1
        buf151 = reinterpret_tensor(buf121, (2048, 1024), (1024, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg104_1, (1024, 1024), (1, 1024), 0), out=buf151)
        del arg104_1
        buf152 = reinterpret_tensor(buf129, (2048, 1024), (1024, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg106_1, (1024, 1024), (1, 1024), 0), out=buf152)
        del arg106_1
        buf153 = reinterpret_tensor(buf149, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [query_states_13, key_states_6, value_states_6, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf150, arg103_1, buf153, 2097152, grid=grid(2097152), stream=stream0)
        del arg103_1
        buf154 = reinterpret_tensor(buf150, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [query_states_13, key_states_6, value_states_6, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf151, arg105_1, buf154, 2097152, grid=grid(2097152), stream=stream0)
        del arg105_1
        buf155 = reinterpret_tensor(buf151, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [query_states_13, key_states_6, value_states_6, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf152, arg107_1, buf155, 2097152, grid=grid(2097152), stream=stream0)
        del arg107_1
        # Topologically Sorted Source Nodes: [query_states_13, key_states_6, value_states_6, attn_output_24], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf156 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf153, buf154, buf155, None, False)
        buf157 = buf156[0]
        del buf156
        buf161 = reinterpret_tensor(buf155, (2048, 1024), (1024, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg108_1, (1024, 1024), (1, 1024), 0), out=buf161)
        del arg108_1
        buf165 = reinterpret_tensor(buf157, (16, 128, 1024), (131072, 1024, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf145, buf161, arg109_1, arg110_1, arg111_1, buf165, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg110_1
        del arg111_1
        buf166 = reinterpret_tensor(buf143, (2048, 4096), (4096, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg112_1, (1024, 4096), (1, 1024), 0), out=buf166)
        del arg112_1
        buf167 = reinterpret_tensor(buf166, (16, 128, 4096), (524288, 4096, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_60], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf167, arg113_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg113_1
        buf168 = reinterpret_tensor(buf165, (2048, 1024), (1024, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg114_1, (4096, 1024), (1, 4096), 0), out=buf168)
        del arg114_1
        buf169 = reinterpret_tensor(buf168, (16, 128, 1024), (131072, 1024, 1), 0); del buf168  # reuse
        buf173 = reinterpret_tensor(buf154, (16, 128, 1024), (131072, 1024, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_64, hidden_states_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf169, buf145, buf161, arg109_1, arg115_1, arg116_1, arg117_1, buf173, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg109_1
        del arg115_1
        del arg116_1
        del arg117_1
        buf174 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg118_1, (1024, 1024), (1, 1024), 0), out=buf174)
        del arg118_1
        buf175 = reinterpret_tensor(buf145, (2048, 1024), (1024, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg120_1, (1024, 1024), (1, 1024), 0), out=buf175)
        del arg120_1
        buf176 = reinterpret_tensor(buf153, (2048, 1024), (1024, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg122_1, (1024, 1024), (1, 1024), 0), out=buf176)
        del arg122_1
        buf177 = reinterpret_tensor(buf173, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [query_states_15, key_states_7, value_states_7, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf174, arg119_1, buf177, 2097152, grid=grid(2097152), stream=stream0)
        del arg119_1
        buf178 = reinterpret_tensor(buf174, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [query_states_15, key_states_7, value_states_7, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf175, arg121_1, buf178, 2097152, grid=grid(2097152), stream=stream0)
        del arg121_1
        buf179 = reinterpret_tensor(buf175, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [query_states_15, key_states_7, value_states_7, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf176, arg123_1, buf179, 2097152, grid=grid(2097152), stream=stream0)
        del arg123_1
        # Topologically Sorted Source Nodes: [query_states_15, key_states_7, value_states_7, attn_output_28], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf180 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf177, buf178, buf179, None, False)
        buf181 = buf180[0]
        del buf180
        buf185 = reinterpret_tensor(buf179, (2048, 1024), (1024, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf181, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg124_1, (1024, 1024), (1, 1024), 0), out=buf185)
        del arg124_1
        buf189 = reinterpret_tensor(buf181, (16, 128, 1024), (131072, 1024, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf169, buf185, arg125_1, arg126_1, arg127_1, buf189, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg126_1
        del arg127_1
        buf190 = reinterpret_tensor(buf167, (2048, 4096), (4096, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg128_1, (1024, 4096), (1, 1024), 0), out=buf190)
        del arg128_1
        buf191 = reinterpret_tensor(buf190, (16, 128, 4096), (524288, 4096, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf191, arg129_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg129_1
        buf192 = reinterpret_tensor(buf189, (2048, 1024), (1024, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg130_1, (4096, 1024), (1, 4096), 0), out=buf192)
        del arg130_1
        buf193 = reinterpret_tensor(buf192, (16, 128, 1024), (131072, 1024, 1), 0); del buf192  # reuse
        buf197 = reinterpret_tensor(buf178, (16, 128, 1024), (131072, 1024, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_73, hidden_states_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf193, buf169, buf185, arg125_1, arg131_1, arg132_1, arg133_1, buf197, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg125_1
        del arg131_1
        del arg132_1
        del arg133_1
        buf198 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg134_1, (1024, 1024), (1, 1024), 0), out=buf198)
        del arg134_1
        buf199 = reinterpret_tensor(buf169, (2048, 1024), (1024, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg136_1, (1024, 1024), (1, 1024), 0), out=buf199)
        del arg136_1
        buf200 = reinterpret_tensor(buf177, (2048, 1024), (1024, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg138_1, (1024, 1024), (1, 1024), 0), out=buf200)
        del arg138_1
        buf201 = reinterpret_tensor(buf197, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [query_states_17, key_states_8, value_states_8, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf198, arg135_1, buf201, 2097152, grid=grid(2097152), stream=stream0)
        del arg135_1
        buf202 = reinterpret_tensor(buf198, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [query_states_17, key_states_8, value_states_8, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf199, arg137_1, buf202, 2097152, grid=grid(2097152), stream=stream0)
        del arg137_1
        buf203 = reinterpret_tensor(buf199, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [query_states_17, key_states_8, value_states_8, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf200, arg139_1, buf203, 2097152, grid=grid(2097152), stream=stream0)
        del arg139_1
        # Topologically Sorted Source Nodes: [query_states_17, key_states_8, value_states_8, attn_output_32], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf204 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf201, buf202, buf203, None, False)
        buf205 = buf204[0]
        del buf204
        buf209 = reinterpret_tensor(buf203, (2048, 1024), (1024, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf205, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg140_1, (1024, 1024), (1, 1024), 0), out=buf209)
        del arg140_1
        buf213 = reinterpret_tensor(buf205, (16, 128, 1024), (131072, 1024, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76, hidden_states_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf193, buf209, arg141_1, arg142_1, arg143_1, buf213, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg142_1
        del arg143_1
        buf214 = reinterpret_tensor(buf191, (2048, 4096), (4096, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf213, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg144_1, (1024, 4096), (1, 1024), 0), out=buf214)
        del arg144_1
        buf215 = reinterpret_tensor(buf214, (16, 128, 4096), (524288, 4096, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf215, arg145_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg145_1
        buf216 = reinterpret_tensor(buf213, (2048, 1024), (1024, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg146_1, (4096, 1024), (1, 4096), 0), out=buf216)
        del arg146_1
        buf217 = reinterpret_tensor(buf216, (16, 128, 1024), (131072, 1024, 1), 0); del buf216  # reuse
        buf221 = reinterpret_tensor(buf202, (16, 128, 1024), (131072, 1024, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76, hidden_states_82, hidden_states_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf217, buf193, buf209, arg141_1, arg147_1, arg148_1, arg149_1, buf221, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg141_1
        del arg147_1
        del arg148_1
        del arg149_1
        buf222 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg150_1, (1024, 1024), (1, 1024), 0), out=buf222)
        del arg150_1
        buf223 = reinterpret_tensor(buf193, (2048, 1024), (1024, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg152_1, (1024, 1024), (1, 1024), 0), out=buf223)
        del arg152_1
        buf224 = reinterpret_tensor(buf201, (2048, 1024), (1024, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg154_1, (1024, 1024), (1, 1024), 0), out=buf224)
        del arg154_1
        buf225 = reinterpret_tensor(buf221, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [query_states_19, key_states_9, value_states_9, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf222, arg151_1, buf225, 2097152, grid=grid(2097152), stream=stream0)
        del arg151_1
        buf226 = reinterpret_tensor(buf222, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [query_states_19, key_states_9, value_states_9, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf223, arg153_1, buf226, 2097152, grid=grid(2097152), stream=stream0)
        del arg153_1
        buf227 = reinterpret_tensor(buf223, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [query_states_19, key_states_9, value_states_9, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf224, arg155_1, buf227, 2097152, grid=grid(2097152), stream=stream0)
        del arg155_1
        # Topologically Sorted Source Nodes: [query_states_19, key_states_9, value_states_9, attn_output_36], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf228 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf225, buf226, buf227, None, False)
        buf229 = buf228[0]
        del buf228
        buf233 = reinterpret_tensor(buf227, (2048, 1024), (1024, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg156_1, (1024, 1024), (1, 1024), 0), out=buf233)
        del arg156_1
        buf237 = reinterpret_tensor(buf229, (16, 128, 1024), (131072, 1024, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf217, buf233, arg157_1, arg158_1, arg159_1, buf237, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg158_1
        del arg159_1
        buf238 = reinterpret_tensor(buf215, (2048, 4096), (4096, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf237, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg160_1, (1024, 4096), (1, 1024), 0), out=buf238)
        del arg160_1
        buf239 = reinterpret_tensor(buf238, (16, 128, 4096), (524288, 4096, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf239, arg161_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg161_1
        buf240 = reinterpret_tensor(buf237, (2048, 1024), (1024, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg162_1, (4096, 1024), (1, 4096), 0), out=buf240)
        del arg162_1
        buf241 = reinterpret_tensor(buf240, (16, 128, 1024), (131072, 1024, 1), 0); del buf240  # reuse
        buf245 = reinterpret_tensor(buf226, (16, 128, 1024), (131072, 1024, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_91, hidden_states_92], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf241, buf217, buf233, arg157_1, arg163_1, arg164_1, arg165_1, buf245, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg157_1
        del arg163_1
        del arg164_1
        del arg165_1
        buf246 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg166_1, (1024, 1024), (1, 1024), 0), out=buf246)
        del arg166_1
        buf247 = reinterpret_tensor(buf217, (2048, 1024), (1024, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg168_1, (1024, 1024), (1, 1024), 0), out=buf247)
        del arg168_1
        buf248 = reinterpret_tensor(buf225, (2048, 1024), (1024, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg170_1, (1024, 1024), (1, 1024), 0), out=buf248)
        del arg170_1
        buf249 = reinterpret_tensor(buf245, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [query_states_21, key_states_10, value_states_10, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf246, arg167_1, buf249, 2097152, grid=grid(2097152), stream=stream0)
        del arg167_1
        buf250 = reinterpret_tensor(buf246, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [query_states_21, key_states_10, value_states_10, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf247, arg169_1, buf250, 2097152, grid=grid(2097152), stream=stream0)
        del arg169_1
        buf251 = reinterpret_tensor(buf247, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [query_states_21, key_states_10, value_states_10, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf248, arg171_1, buf251, 2097152, grid=grid(2097152), stream=stream0)
        del arg171_1
        # Topologically Sorted Source Nodes: [query_states_21, key_states_10, value_states_10, attn_output_40], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf252 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf249, buf250, buf251, None, False)
        buf253 = buf252[0]
        del buf252
        buf257 = reinterpret_tensor(buf251, (2048, 1024), (1024, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg172_1, (1024, 1024), (1, 1024), 0), out=buf257)
        del arg172_1
        buf261 = reinterpret_tensor(buf253, (16, 128, 1024), (131072, 1024, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf241, buf257, arg173_1, arg174_1, arg175_1, buf261, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg174_1
        del arg175_1
        buf262 = reinterpret_tensor(buf239, (2048, 4096), (4096, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf261, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg176_1, (1024, 4096), (1, 1024), 0), out=buf262)
        del arg176_1
        buf263 = reinterpret_tensor(buf262, (16, 128, 4096), (524288, 4096, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf263, arg177_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg177_1
        buf264 = reinterpret_tensor(buf261, (2048, 1024), (1024, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf263, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg178_1, (4096, 1024), (1, 4096), 0), out=buf264)
        del arg178_1
        buf265 = reinterpret_tensor(buf264, (16, 128, 1024), (131072, 1024, 1), 0); del buf264  # reuse
        buf269 = reinterpret_tensor(buf250, (16, 128, 1024), (131072, 1024, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_100, hidden_states_101], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf265, buf241, buf257, arg173_1, arg179_1, arg180_1, arg181_1, buf269, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg173_1
        del arg179_1
        del arg180_1
        del arg181_1
        buf270 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg182_1, (1024, 1024), (1, 1024), 0), out=buf270)
        del arg182_1
        buf271 = reinterpret_tensor(buf241, (2048, 1024), (1024, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg184_1, (1024, 1024), (1, 1024), 0), out=buf271)
        del arg184_1
        buf272 = reinterpret_tensor(buf249, (2048, 1024), (1024, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg186_1, (1024, 1024), (1, 1024), 0), out=buf272)
        del arg186_1
        buf273 = reinterpret_tensor(buf269, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [query_states_23, key_states_11, value_states_11, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf270, arg183_1, buf273, 2097152, grid=grid(2097152), stream=stream0)
        del arg183_1
        buf274 = reinterpret_tensor(buf270, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [query_states_23, key_states_11, value_states_11, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf271, arg185_1, buf274, 2097152, grid=grid(2097152), stream=stream0)
        del arg185_1
        buf275 = reinterpret_tensor(buf271, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [query_states_23, key_states_11, value_states_11, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf272, arg187_1, buf275, 2097152, grid=grid(2097152), stream=stream0)
        del arg187_1
        # Topologically Sorted Source Nodes: [query_states_23, key_states_11, value_states_11, attn_output_44], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf276 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf273, buf274, buf275, None, False)
        buf277 = buf276[0]
        del buf276
        buf281 = reinterpret_tensor(buf275, (2048, 1024), (1024, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf277, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg188_1, (1024, 1024), (1, 1024), 0), out=buf281)
        del arg188_1
        buf285 = reinterpret_tensor(buf277, (16, 128, 1024), (131072, 1024, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_103, hidden_states_104], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf265, buf281, arg189_1, arg190_1, arg191_1, buf285, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg190_1
        del arg191_1
        buf286 = reinterpret_tensor(buf263, (2048, 4096), (4096, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf285, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg192_1, (1024, 4096), (1, 1024), 0), out=buf286)
        del arg192_1
        buf287 = reinterpret_tensor(buf286, (16, 128, 4096), (524288, 4096, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_105], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf287, arg193_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg193_1
        buf288 = reinterpret_tensor(buf285, (2048, 1024), (1024, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf287, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg194_1, (4096, 1024), (1, 4096), 0), out=buf288)
        del arg194_1
        buf289 = reinterpret_tensor(buf288, (16, 128, 1024), (131072, 1024, 1), 0); del buf288  # reuse
        buf317 = reinterpret_tensor(buf274, (16, 128, 1024), (131072, 1024, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_103, hidden_states_109, hidden_states_110], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf289, buf265, buf281, arg189_1, arg195_1, arg196_1, arg197_1, buf317, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg189_1
        del arg195_1
        del arg196_1
        del arg197_1
        buf293 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [ne_1, mask_3, cumsum_1], Original ATen: [aten.ne, aten._to_copy, aten.cumsum]
        triton_per_fused__to_copy_cumsum_ne_0.run(arg1_1, buf293, 16, 128, grid=grid(16), stream=stream0)
        buf294 = buf289; del buf289  # reuse
        buf298 = reinterpret_tensor(buf281, (16, 128, 1024), (131072, 1024, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [embedding_1, inputs_embeds_1, hidden_states_111, hidden_states_113], Original ATen: [aten.embedding, aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_embedding_mul_native_layer_norm_1.run(arg1_1, arg2_1, buf293, arg198_1, arg199_1, arg200_1, buf294, buf298, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg198_1
        del arg199_1
        del arg1_1
        del arg200_1
        del buf293
        buf299 = reinterpret_tensor(buf265, (2048, 1024), (1024, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg201_1, (1024, 1024), (1, 1024), 0), out=buf299)
        del arg201_1
        buf300 = reinterpret_tensor(buf273, (2048, 1024), (1024, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg203_1, (1024, 1024), (1, 1024), 0), out=buf300)
        del arg203_1
        buf301 = reinterpret_tensor(buf272, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [key_states_12], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf300, arg204_1, buf301, 2097152, grid=grid(2097152), stream=stream0)
        del arg204_1
        buf302 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg205_1, (1024, 1024), (1, 1024), 0), out=buf302)
        del arg205_1
        buf303 = reinterpret_tensor(buf298, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf298  # reuse
        # Topologically Sorted Source Nodes: [value_states_12], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf302, arg206_1, buf303, 2097152, grid=grid(2097152), stream=stream0)
        del arg206_1
        buf304 = reinterpret_tensor(buf302, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [query_states_25, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf299, arg202_1, buf304, 2097152, grid=grid(2097152), stream=stream0)
        del arg202_1
        buf305 = empty_strided_cuda((16, 16, 128, 128), (262144, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [query_states_25, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_6.run(buf305, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_25, attn_output_48], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf306 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf304, buf301, buf303, buf305, False)
        buf307 = buf306[0]
        del buf306
        buf311 = reinterpret_tensor(buf304, (2048, 1024), (1024, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf307, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg207_1, (1024, 1024), (1, 1024), 0), out=buf311)
        del arg207_1
        buf315 = reinterpret_tensor(buf307, (16, 128, 1024), (131072, 1024, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_115, hidden_states_116], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf294, buf311, arg208_1, arg209_1, arg210_1, buf315, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg209_1
        del arg210_1
        buf316 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg211_1, (1024, 1024), (1, 1024), 0), out=buf316)
        del arg211_1
        buf318 = reinterpret_tensor(buf315, (2048, 1024), (1024, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg213_1, (1024, 1024), (1, 1024), 0), out=buf318)
        del arg213_1
        buf319 = reinterpret_tensor(buf248, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [key_states_13], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf318, arg214_1, buf319, 2097152, grid=grid(2097152), stream=stream0)
        del arg214_1
        buf320 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg215_1, (1024, 1024), (1, 1024), 0), out=buf320)
        del arg215_1
        buf321 = reinterpret_tensor(buf224, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [value_states_13], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf320, arg216_1, buf321, 2097152, grid=grid(2097152), stream=stream0)
        del arg216_1
        buf322 = reinterpret_tensor(buf320, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [query_states_27, attn_output_52], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf316, arg212_1, buf322, 2097152, grid=grid(2097152), stream=stream0)
        del arg212_1
        # Topologically Sorted Source Nodes: [query_states_27, attn_output_52], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf323 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf322, buf319, buf321, None, False)
        buf324 = buf323[0]
        del buf323
        buf328 = reinterpret_tensor(buf322, (2048, 1024), (1024, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg217_1, (1024, 1024), (1, 1024), 0), out=buf328)
        del arg217_1
        buf329 = reinterpret_tensor(buf328, (16, 128, 1024), (131072, 1024, 1), 0); del buf328  # reuse
        buf333 = reinterpret_tensor(buf324, (16, 128, 1024), (131072, 1024, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_115, hidden_states_118, hidden_states_119], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf329, buf294, buf311, arg208_1, arg218_1, arg219_1, arg220_1, buf333, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg208_1
        del arg218_1
        del arg219_1
        del arg220_1
        buf334 = reinterpret_tensor(buf287, (2048, 4096), (4096, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf333, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg221_1, (1024, 4096), (1, 1024), 0), out=buf334)
        del arg221_1
        buf335 = reinterpret_tensor(buf334, (16, 128, 4096), (524288, 4096, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_120], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf335, arg222_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg222_1
        buf336 = reinterpret_tensor(buf333, (2048, 1024), (1024, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg223_1, (4096, 1024), (1, 4096), 0), out=buf336)
        del arg223_1
        buf340 = reinterpret_tensor(buf311, (16, 128, 1024), (131072, 1024, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_124, hidden_states_125], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf329, buf336, arg224_1, arg225_1, arg226_1, buf340, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg225_1
        del arg226_1
        buf341 = reinterpret_tensor(buf294, (2048, 1024), (1024, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf340, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg227_1, (1024, 1024), (1, 1024), 0), out=buf341)
        del arg227_1
        buf342 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf340, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg229_1, (1024, 1024), (1, 1024), 0), out=buf342)
        del arg229_1
        buf343 = reinterpret_tensor(buf200, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [key_states_14], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf342, arg230_1, buf343, 2097152, grid=grid(2097152), stream=stream0)
        del arg230_1
        buf344 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf340, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg231_1, (1024, 1024), (1, 1024), 0), out=buf344)
        del arg231_1
        buf345 = reinterpret_tensor(buf340, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [value_states_14], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf344, arg232_1, buf345, 2097152, grid=grid(2097152), stream=stream0)
        del arg232_1
        buf346 = reinterpret_tensor(buf344, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [query_states_29, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf341, arg228_1, buf346, 2097152, grid=grid(2097152), stream=stream0)
        del arg228_1
        buf347 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [query_states_29, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_6.run(buf347, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_29, attn_output_56], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf348 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf346, buf343, buf345, buf347, False)
        buf349 = buf348[0]
        del buf348
        buf353 = reinterpret_tensor(buf346, (2048, 1024), (1024, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf349, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg233_1, (1024, 1024), (1, 1024), 0), out=buf353)
        del arg233_1
        buf354 = reinterpret_tensor(buf353, (16, 128, 1024), (131072, 1024, 1), 0); del buf353  # reuse
        buf358 = reinterpret_tensor(buf349, (16, 128, 1024), (131072, 1024, 1), 0); del buf349  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_124, hidden_states_127, hidden_states_128], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf354, buf329, buf336, arg224_1, arg234_1, arg235_1, arg236_1, buf358, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg224_1
        del arg234_1
        del arg235_1
        del arg236_1
        buf359 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg237_1, (1024, 1024), (1, 1024), 0), out=buf359)
        del arg237_1
        buf360 = reinterpret_tensor(buf358, (2048, 1024), (1024, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg239_1, (1024, 1024), (1, 1024), 0), out=buf360)
        del arg239_1
        buf361 = reinterpret_tensor(buf329, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [key_states_15], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf360, arg240_1, buf361, 2097152, grid=grid(2097152), stream=stream0)
        del arg240_1
        buf362 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg241_1, (1024, 1024), (1, 1024), 0), out=buf362)
        del arg241_1
        buf363 = reinterpret_tensor(buf341, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [value_states_15], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf362, arg242_1, buf363, 2097152, grid=grid(2097152), stream=stream0)
        del arg242_1
        buf364 = reinterpret_tensor(buf362, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf362  # reuse
        # Topologically Sorted Source Nodes: [query_states_31, attn_output_60], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf359, arg238_1, buf364, 2097152, grid=grid(2097152), stream=stream0)
        del arg238_1
        # Topologically Sorted Source Nodes: [query_states_31, attn_output_60], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf365 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf364, buf361, buf363, None, False)
        buf366 = buf365[0]
        del buf365
        buf370 = reinterpret_tensor(buf364, (2048, 1024), (1024, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf366, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg243_1, (1024, 1024), (1, 1024), 0), out=buf370)
        del arg243_1
        buf374 = reinterpret_tensor(buf366, (16, 128, 1024), (131072, 1024, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_130, hidden_states_131], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf354, buf370, arg244_1, arg245_1, arg246_1, buf374, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg245_1
        del arg246_1
        buf375 = reinterpret_tensor(buf335, (2048, 4096), (4096, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf374, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg247_1, (1024, 4096), (1, 1024), 0), out=buf375)
        del arg247_1
        buf376 = reinterpret_tensor(buf375, (16, 128, 4096), (524288, 4096, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_132], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf376, arg248_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg248_1
        buf377 = reinterpret_tensor(buf374, (2048, 1024), (1024, 1), 0); del buf374  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf376, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg249_1, (4096, 1024), (1, 4096), 0), out=buf377)
        del arg249_1
        buf378 = reinterpret_tensor(buf377, (16, 128, 1024), (131072, 1024, 1), 0); del buf377  # reuse
        buf382 = reinterpret_tensor(buf359, (16, 128, 1024), (131072, 1024, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_130, hidden_states_136, hidden_states_137], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf378, buf354, buf370, arg244_1, arg250_1, arg251_1, arg252_1, buf382, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg244_1
        del arg250_1
        del arg251_1
        del arg252_1
        buf383 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf382, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg253_1, (1024, 1024), (1, 1024), 0), out=buf383)
        del arg253_1
        buf384 = reinterpret_tensor(buf354, (2048, 1024), (1024, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf382, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg255_1, (1024, 1024), (1, 1024), 0), out=buf384)
        del arg255_1
        buf385 = reinterpret_tensor(buf176, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf384, arg256_1, buf385, 2097152, grid=grid(2097152), stream=stream0)
        del arg256_1
        buf386 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf382, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg257_1, (1024, 1024), (1, 1024), 0), out=buf386)
        del arg257_1
        buf387 = reinterpret_tensor(buf382, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf386, arg258_1, buf387, 2097152, grid=grid(2097152), stream=stream0)
        del arg258_1
        buf388 = reinterpret_tensor(buf386, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [query_states_33, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf383, arg254_1, buf388, 2097152, grid=grid(2097152), stream=stream0)
        del arg254_1
        buf389 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [query_states_33, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_6.run(buf389, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_33, attn_output_64], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf390 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf388, buf385, buf387, buf389, False)
        buf391 = buf390[0]
        del buf390
        buf395 = reinterpret_tensor(buf388, (2048, 1024), (1024, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf391, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg259_1, (1024, 1024), (1, 1024), 0), out=buf395)
        del arg259_1
        buf399 = reinterpret_tensor(buf391, (16, 128, 1024), (131072, 1024, 1), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_139, hidden_states_140], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf378, buf395, arg260_1, arg261_1, arg262_1, buf399, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg261_1
        del arg262_1
        buf400 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf399, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg263_1, (1024, 1024), (1, 1024), 0), out=buf400)
        del arg263_1
        buf401 = reinterpret_tensor(buf399, (2048, 1024), (1024, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg265_1, (1024, 1024), (1, 1024), 0), out=buf401)
        del arg265_1
        buf402 = reinterpret_tensor(buf152, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [key_states_17], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf401, arg266_1, buf402, 2097152, grid=grid(2097152), stream=stream0)
        del arg266_1
        buf403 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg267_1, (1024, 1024), (1, 1024), 0), out=buf403)
        del arg267_1
        buf404 = reinterpret_tensor(buf128, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [value_states_17], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf403, arg268_1, buf404, 2097152, grid=grid(2097152), stream=stream0)
        del arg268_1
        buf405 = reinterpret_tensor(buf403, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [query_states_35, attn_output_68], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf400, arg264_1, buf405, 2097152, grid=grid(2097152), stream=stream0)
        del arg264_1
        # Topologically Sorted Source Nodes: [query_states_35, attn_output_68], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf406 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf405, buf402, buf404, None, False)
        buf407 = buf406[0]
        del buf406
        buf411 = reinterpret_tensor(buf405, (2048, 1024), (1024, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf407, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg269_1, (1024, 1024), (1, 1024), 0), out=buf411)
        del arg269_1
        buf412 = reinterpret_tensor(buf411, (16, 128, 1024), (131072, 1024, 1), 0); del buf411  # reuse
        buf416 = reinterpret_tensor(buf407, (16, 128, 1024), (131072, 1024, 1), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_139, hidden_states_142, hidden_states_143], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf412, buf378, buf395, arg260_1, arg270_1, arg271_1, arg272_1, buf416, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg260_1
        del arg270_1
        del arg271_1
        del arg272_1
        buf417 = reinterpret_tensor(buf376, (2048, 4096), (4096, 1), 0); del buf376  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf416, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg273_1, (1024, 4096), (1, 1024), 0), out=buf417)
        del arg273_1
        buf418 = reinterpret_tensor(buf417, (16, 128, 4096), (524288, 4096, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_144], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf418, arg274_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg274_1
        buf419 = reinterpret_tensor(buf416, (2048, 1024), (1024, 1), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf418, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg275_1, (4096, 1024), (1, 4096), 0), out=buf419)
        del arg275_1
        buf423 = reinterpret_tensor(buf395, (16, 128, 1024), (131072, 1024, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_148, hidden_states_149], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf412, buf419, arg276_1, arg277_1, arg278_1, buf423, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg277_1
        del arg278_1
        buf424 = reinterpret_tensor(buf378, (2048, 1024), (1024, 1), 0); del buf378  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg279_1, (1024, 1024), (1, 1024), 0), out=buf424)
        del arg279_1
        buf425 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg281_1, (1024, 1024), (1, 1024), 0), out=buf425)
        del arg281_1
        buf426 = reinterpret_tensor(buf104, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [key_states_18], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf425, arg282_1, buf426, 2097152, grid=grid(2097152), stream=stream0)
        del arg282_1
        buf427 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg283_1, (1024, 1024), (1, 1024), 0), out=buf427)
        del arg283_1
        buf428 = reinterpret_tensor(buf423, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [value_states_18], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf427, arg284_1, buf428, 2097152, grid=grid(2097152), stream=stream0)
        del arg284_1
        buf429 = reinterpret_tensor(buf427, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [query_states_37, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf424, arg280_1, buf429, 2097152, grid=grid(2097152), stream=stream0)
        del arg280_1
        buf430 = buf389; del buf389  # reuse
        # Topologically Sorted Source Nodes: [query_states_37, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_6.run(buf430, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_37, attn_output_72], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf431 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf429, buf426, buf428, buf430, False)
        buf432 = buf431[0]
        del buf431
        buf436 = reinterpret_tensor(buf429, (2048, 1024), (1024, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf432, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg285_1, (1024, 1024), (1, 1024), 0), out=buf436)
        del arg285_1
        buf437 = reinterpret_tensor(buf436, (16, 128, 1024), (131072, 1024, 1), 0); del buf436  # reuse
        buf441 = reinterpret_tensor(buf432, (16, 128, 1024), (131072, 1024, 1), 0); del buf432  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_148, hidden_states_151, hidden_states_152], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf437, buf412, buf419, arg276_1, arg286_1, arg287_1, arg288_1, buf441, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg276_1
        del arg286_1
        del arg287_1
        del arg288_1
        buf442 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf441, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg289_1, (1024, 1024), (1, 1024), 0), out=buf442)
        del arg289_1
        buf443 = reinterpret_tensor(buf441, (2048, 1024), (1024, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg291_1, (1024, 1024), (1, 1024), 0), out=buf443)
        del arg291_1
        buf444 = reinterpret_tensor(buf412, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [key_states_19], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf443, arg292_1, buf444, 2097152, grid=grid(2097152), stream=stream0)
        del arg292_1
        buf445 = buf443; del buf443  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg293_1, (1024, 1024), (1, 1024), 0), out=buf445)
        del arg293_1
        buf446 = reinterpret_tensor(buf424, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [value_states_19], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf445, arg294_1, buf446, 2097152, grid=grid(2097152), stream=stream0)
        del arg294_1
        buf447 = reinterpret_tensor(buf445, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf445  # reuse
        # Topologically Sorted Source Nodes: [query_states_39, attn_output_76], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf442, arg290_1, buf447, 2097152, grid=grid(2097152), stream=stream0)
        del arg290_1
        # Topologically Sorted Source Nodes: [query_states_39, attn_output_76], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf448 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf447, buf444, buf446, None, False)
        buf449 = buf448[0]
        del buf448
        buf453 = reinterpret_tensor(buf447, (2048, 1024), (1024, 1), 0); del buf447  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf449, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg295_1, (1024, 1024), (1, 1024), 0), out=buf453)
        del arg295_1
        buf457 = reinterpret_tensor(buf449, (16, 128, 1024), (131072, 1024, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_154, hidden_states_155], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf437, buf453, arg296_1, arg297_1, arg298_1, buf457, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg297_1
        del arg298_1
        buf458 = reinterpret_tensor(buf418, (2048, 4096), (4096, 1), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf457, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg299_1, (1024, 4096), (1, 1024), 0), out=buf458)
        del arg299_1
        buf459 = reinterpret_tensor(buf458, (16, 128, 4096), (524288, 4096, 1), 0); del buf458  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_156], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf459, arg300_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg300_1
        buf460 = reinterpret_tensor(buf457, (2048, 1024), (1024, 1), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf459, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg301_1, (4096, 1024), (1, 4096), 0), out=buf460)
        del arg301_1
        buf461 = reinterpret_tensor(buf460, (16, 128, 1024), (131072, 1024, 1), 0); del buf460  # reuse
        buf465 = reinterpret_tensor(buf442, (16, 128, 1024), (131072, 1024, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_154, hidden_states_160, hidden_states_161], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf461, buf437, buf453, arg296_1, arg302_1, arg303_1, arg304_1, buf465, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg296_1
        del arg302_1
        del arg303_1
        del arg304_1
        buf466 = buf453; del buf453  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf465, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg305_1, (1024, 1024), (1, 1024), 0), out=buf466)
        del arg305_1
        buf467 = reinterpret_tensor(buf437, (2048, 1024), (1024, 1), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf465, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg307_1, (1024, 1024), (1, 1024), 0), out=buf467)
        del arg307_1
        buf468 = reinterpret_tensor(buf80, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf467, arg308_1, buf468, 2097152, grid=grid(2097152), stream=stream0)
        del arg308_1
        buf469 = buf467; del buf467  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf465, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg309_1, (1024, 1024), (1, 1024), 0), out=buf469)
        del arg309_1
        buf470 = reinterpret_tensor(buf465, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf465  # reuse
        # Topologically Sorted Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf469, arg310_1, buf470, 2097152, grid=grid(2097152), stream=stream0)
        del arg310_1
        buf471 = reinterpret_tensor(buf469, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [query_states_41, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf466, arg306_1, buf471, 2097152, grid=grid(2097152), stream=stream0)
        del arg306_1
        buf472 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [query_states_41, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_6.run(buf472, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_41, attn_output_80], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf473 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf471, buf468, buf470, buf472, False)
        buf474 = buf473[0]
        del buf473
        buf478 = reinterpret_tensor(buf471, (2048, 1024), (1024, 1), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf474, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg311_1, (1024, 1024), (1, 1024), 0), out=buf478)
        del arg311_1
        buf482 = reinterpret_tensor(buf474, (16, 128, 1024), (131072, 1024, 1), 0); del buf474  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_163, hidden_states_164], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf461, buf478, arg312_1, arg313_1, arg314_1, buf482, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg313_1
        del arg314_1
        buf483 = buf466; del buf466  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf482, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg315_1, (1024, 1024), (1, 1024), 0), out=buf483)
        del arg315_1
        buf484 = reinterpret_tensor(buf482, (2048, 1024), (1024, 1), 0); del buf482  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg317_1, (1024, 1024), (1, 1024), 0), out=buf484)
        del arg317_1
        buf485 = reinterpret_tensor(buf56, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [key_states_21], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf484, arg318_1, buf485, 2097152, grid=grid(2097152), stream=stream0)
        del arg318_1
        buf486 = buf484; del buf484  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg319_1, (1024, 1024), (1, 1024), 0), out=buf486)
        del arg319_1
        buf487 = reinterpret_tensor(buf32, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [value_states_21], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf486, arg320_1, buf487, 2097152, grid=grid(2097152), stream=stream0)
        del arg320_1
        buf488 = reinterpret_tensor(buf486, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf486  # reuse
        # Topologically Sorted Source Nodes: [query_states_43, attn_output_84], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf483, arg316_1, buf488, 2097152, grid=grid(2097152), stream=stream0)
        del arg316_1
        # Topologically Sorted Source Nodes: [query_states_43, attn_output_84], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf489 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf488, buf485, buf487, None, False)
        buf490 = buf489[0]
        del buf489
        buf494 = reinterpret_tensor(buf488, (2048, 1024), (1024, 1), 0); del buf488  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf490, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg321_1, (1024, 1024), (1, 1024), 0), out=buf494)
        del arg321_1
        buf495 = reinterpret_tensor(buf494, (16, 128, 1024), (131072, 1024, 1), 0); del buf494  # reuse
        buf499 = reinterpret_tensor(buf490, (16, 128, 1024), (131072, 1024, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_163, hidden_states_166, hidden_states_167], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf495, buf461, buf478, arg312_1, arg322_1, arg323_1, arg324_1, buf499, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg312_1
        del arg322_1
        del arg323_1
        del arg324_1
        buf500 = reinterpret_tensor(buf459, (2048, 4096), (4096, 1), 0); del buf459  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf499, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg325_1, (1024, 4096), (1, 1024), 0), out=buf500)
        del arg325_1
        buf501 = reinterpret_tensor(buf500, (16, 128, 4096), (524288, 4096, 1), 0); del buf500  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_168], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf501, arg326_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg326_1
        buf502 = reinterpret_tensor(buf499, (2048, 1024), (1024, 1), 0); del buf499  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf501, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg327_1, (4096, 1024), (1, 4096), 0), out=buf502)
        del arg327_1
        buf506 = reinterpret_tensor(buf478, (16, 128, 1024), (131072, 1024, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_172, hidden_states_173], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf495, buf502, arg328_1, arg329_1, arg330_1, buf506, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg329_1
        del arg330_1
        buf507 = reinterpret_tensor(buf461, (2048, 1024), (1024, 1), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf506, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg331_1, (1024, 1024), (1, 1024), 0), out=buf507)
        del arg331_1
        buf508 = buf483; del buf483  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf506, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg333_1, (1024, 1024), (1, 1024), 0), out=buf508)
        del arg333_1
        buf509 = reinterpret_tensor(buf8, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [key_states_22], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf508, arg334_1, buf509, 2097152, grid=grid(2097152), stream=stream0)
        del arg334_1
        buf510 = buf508; del buf508  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf506, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg335_1, (1024, 1024), (1, 1024), 0), out=buf510)
        del arg335_1
        buf511 = reinterpret_tensor(buf506, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf506  # reuse
        # Topologically Sorted Source Nodes: [value_states_22], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf510, arg336_1, buf511, 2097152, grid=grid(2097152), stream=stream0)
        del arg336_1
        buf512 = reinterpret_tensor(buf510, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf510  # reuse
        # Topologically Sorted Source Nodes: [query_states_45, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf507, arg332_1, buf512, 2097152, grid=grid(2097152), stream=stream0)
        del arg332_1
        buf513 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [query_states_45, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_6.run(buf513, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_45, attn_output_88], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf514 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf512, buf509, buf511, buf513, False)
        buf515 = buf514[0]
        del buf514
        buf519 = reinterpret_tensor(buf512, (2048, 1024), (1024, 1), 0); del buf512  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf515, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg337_1, (1024, 1024), (1, 1024), 0), out=buf519)
        del arg337_1
        buf520 = reinterpret_tensor(buf519, (16, 128, 1024), (131072, 1024, 1), 0); del buf519  # reuse
        buf524 = reinterpret_tensor(buf515, (16, 128, 1024), (131072, 1024, 1), 0); del buf515  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_172, hidden_states_175, hidden_states_176], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf520, buf495, buf502, arg328_1, arg338_1, arg339_1, arg340_1, buf524, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg328_1
        del arg338_1
        del arg339_1
        del arg340_1
        buf525 = buf502; del buf502  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf524, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg341_1, (1024, 1024), (1, 1024), 0), out=buf525)
        del arg341_1
        buf526 = reinterpret_tensor(buf524, (2048, 1024), (1024, 1), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg343_1, (1024, 1024), (1, 1024), 0), out=buf526)
        del arg343_1
        buf527 = reinterpret_tensor(buf495, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf495  # reuse
        # Topologically Sorted Source Nodes: [key_states_23], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf526, arg344_1, buf527, 2097152, grid=grid(2097152), stream=stream0)
        del arg344_1
        buf528 = buf526; del buf526  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg345_1, (1024, 1024), (1, 1024), 0), out=buf528)
        del arg345_1
        buf529 = reinterpret_tensor(buf507, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf507  # reuse
        # Topologically Sorted Source Nodes: [value_states_23], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf528, arg346_1, buf529, 2097152, grid=grid(2097152), stream=stream0)
        del arg346_1
        buf530 = reinterpret_tensor(buf528, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf528  # reuse
        # Topologically Sorted Source Nodes: [query_states_47, attn_output_92], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf525, arg342_1, buf530, 2097152, grid=grid(2097152), stream=stream0)
        del arg342_1
        # Topologically Sorted Source Nodes: [query_states_47, attn_output_92], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf531 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf530, buf527, buf529, None, False)
        buf532 = buf531[0]
        del buf531
        buf536 = reinterpret_tensor(buf530, (2048, 1024), (1024, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf532, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg347_1, (1024, 1024), (1, 1024), 0), out=buf536)
        del arg347_1
        buf540 = reinterpret_tensor(buf532, (16, 128, 1024), (131072, 1024, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_178, hidden_states_179], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf520, buf536, arg348_1, arg349_1, arg350_1, buf540, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg349_1
        del arg350_1
        buf541 = reinterpret_tensor(buf501, (2048, 4096), (4096, 1), 0); del buf501  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf540, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg351_1, (1024, 4096), (1, 1024), 0), out=buf541)
        del arg351_1
        buf542 = reinterpret_tensor(buf541, (16, 128, 4096), (524288, 4096, 1), 0); del buf541  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_180], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf542, arg352_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg352_1
        buf543 = reinterpret_tensor(buf540, (2048, 1024), (1024, 1), 0); del buf540  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf542, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg353_1, (4096, 1024), (1, 4096), 0), out=buf543)
        del arg353_1
        buf544 = reinterpret_tensor(buf543, (16, 128, 1024), (131072, 1024, 1), 0); del buf543  # reuse
        buf548 = reinterpret_tensor(buf525, (16, 128, 1024), (131072, 1024, 1), 0); del buf525  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_178, hidden_states_184, hidden_states_185], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf544, buf520, buf536, arg348_1, arg354_1, arg355_1, arg356_1, buf548, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg348_1
        del arg354_1
        del arg355_1
        del arg356_1
        buf549 = buf536; del buf536  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf548, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg357_1, (1024, 1024), (1, 1024), 0), out=buf549)
        del arg357_1
        buf550 = reinterpret_tensor(buf520, (2048, 1024), (1024, 1), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf548, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg359_1, (1024, 1024), (1, 1024), 0), out=buf550)
        del arg359_1
        buf551 = empty_strided_cuda((16, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf550, arg360_1, buf551, 2097152, grid=grid(2097152), stream=stream0)
        del arg360_1
        buf552 = buf550; del buf550  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf548, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg361_1, (1024, 1024), (1, 1024), 0), out=buf552)
        del arg361_1
        buf553 = reinterpret_tensor(buf548, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf548  # reuse
        # Topologically Sorted Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf552, arg362_1, buf553, 2097152, grid=grid(2097152), stream=stream0)
        del arg362_1
        buf554 = reinterpret_tensor(buf552, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf552  # reuse
        # Topologically Sorted Source Nodes: [query_states_49, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf549, arg358_1, buf554, 2097152, grid=grid(2097152), stream=stream0)
        del arg358_1
        buf555 = buf513; del buf513  # reuse
        # Topologically Sorted Source Nodes: [query_states_49, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_6.run(buf555, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_49, attn_output_96], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf556 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf554, buf551, buf553, buf555, False)
        buf557 = buf556[0]
        del buf556
        buf561 = reinterpret_tensor(buf554, (2048, 1024), (1024, 1), 0); del buf554  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf557, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg363_1, (1024, 1024), (1, 1024), 0), out=buf561)
        del arg363_1
        buf565 = reinterpret_tensor(buf557, (16, 128, 1024), (131072, 1024, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_187, hidden_states_188], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf544, buf561, arg364_1, arg365_1, arg366_1, buf565, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg365_1
        del arg366_1
        buf566 = buf549; del buf549  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf565, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg367_1, (1024, 1024), (1, 1024), 0), out=buf566)
        del arg367_1
        buf567 = reinterpret_tensor(buf565, (2048, 1024), (1024, 1), 0); del buf565  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg369_1, (1024, 1024), (1, 1024), 0), out=buf567)
        del arg369_1
        buf568 = empty_strided_cuda((16, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_25], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf567, arg370_1, buf568, 2097152, grid=grid(2097152), stream=stream0)
        del arg370_1
        buf569 = buf567; del buf567  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg371_1, (1024, 1024), (1, 1024), 0), out=buf569)
        del arg371_1
        buf570 = empty_strided_cuda((16, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [value_states_25], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf569, arg372_1, buf570, 2097152, grid=grid(2097152), stream=stream0)
        del arg372_1
        buf571 = reinterpret_tensor(buf569, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf569  # reuse
        # Topologically Sorted Source Nodes: [query_states_51, attn_output_100], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf566, arg368_1, buf571, 2097152, grid=grid(2097152), stream=stream0)
        del arg368_1
        # Topologically Sorted Source Nodes: [query_states_51, attn_output_100], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf572 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf571, buf568, buf570, None, False)
        buf573 = buf572[0]
        del buf572
        buf577 = reinterpret_tensor(buf571, (2048, 1024), (1024, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf573, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg373_1, (1024, 1024), (1, 1024), 0), out=buf577)
        del arg373_1
        buf578 = reinterpret_tensor(buf577, (16, 128, 1024), (131072, 1024, 1), 0); del buf577  # reuse
        buf582 = reinterpret_tensor(buf573, (16, 128, 1024), (131072, 1024, 1), 0); del buf573  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_187, hidden_states_190, hidden_states_191], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf578, buf544, buf561, arg364_1, arg374_1, arg375_1, arg376_1, buf582, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg364_1
        del arg374_1
        del arg375_1
        del arg376_1
        buf583 = reinterpret_tensor(buf542, (2048, 4096), (4096, 1), 0); del buf542  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf582, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg377_1, (1024, 4096), (1, 1024), 0), out=buf583)
        del arg377_1
        buf584 = reinterpret_tensor(buf583, (16, 128, 4096), (524288, 4096, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_192], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf584, arg378_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg378_1
        buf585 = reinterpret_tensor(buf582, (2048, 1024), (1024, 1), 0); del buf582  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf584, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg379_1, (4096, 1024), (1, 4096), 0), out=buf585)
        del arg379_1
        buf589 = reinterpret_tensor(buf561, (16, 128, 1024), (131072, 1024, 1), 0); del buf561  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_196, hidden_states_197], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf578, buf585, arg380_1, arg381_1, arg382_1, buf589, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg381_1
        del arg382_1
        buf590 = reinterpret_tensor(buf544, (2048, 1024), (1024, 1), 0); del buf544  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf589, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg383_1, (1024, 1024), (1, 1024), 0), out=buf590)
        del arg383_1
        buf591 = buf566; del buf566  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf589, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg385_1, (1024, 1024), (1, 1024), 0), out=buf591)
        del arg385_1
        buf592 = empty_strided_cuda((16, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_26], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf591, arg386_1, buf592, 2097152, grid=grid(2097152), stream=stream0)
        del arg386_1
        buf593 = buf591; del buf591  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf589, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg387_1, (1024, 1024), (1, 1024), 0), out=buf593)
        del arg387_1
        buf594 = reinterpret_tensor(buf589, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf589  # reuse
        # Topologically Sorted Source Nodes: [value_states_26], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf593, arg388_1, buf594, 2097152, grid=grid(2097152), stream=stream0)
        del arg388_1
        buf595 = reinterpret_tensor(buf593, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf593  # reuse
        # Topologically Sorted Source Nodes: [query_states_53, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf590, arg384_1, buf595, 2097152, grid=grid(2097152), stream=stream0)
        del arg384_1
        buf596 = buf555; del buf555  # reuse
        # Topologically Sorted Source Nodes: [query_states_53, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_6.run(buf596, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_53, attn_output_104], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf597 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf595, buf592, buf594, buf596, False)
        buf598 = buf597[0]
        del buf597
        buf602 = reinterpret_tensor(buf595, (2048, 1024), (1024, 1), 0); del buf595  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf598, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg389_1, (1024, 1024), (1, 1024), 0), out=buf602)
        del arg389_1
        buf603 = reinterpret_tensor(buf602, (16, 128, 1024), (131072, 1024, 1), 0); del buf602  # reuse
        buf607 = reinterpret_tensor(buf598, (16, 128, 1024), (131072, 1024, 1), 0); del buf598  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_196, hidden_states_199, hidden_states_200], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf603, buf578, buf585, arg380_1, arg390_1, arg391_1, arg392_1, buf607, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg380_1
        del arg390_1
        del arg391_1
        del arg392_1
        buf608 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf607, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg393_1, (1024, 1024), (1, 1024), 0), out=buf608)
        del arg393_1
        buf609 = reinterpret_tensor(buf607, (2048, 1024), (1024, 1), 0); del buf607  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg395_1, (1024, 1024), (1, 1024), 0), out=buf609)
        del arg395_1
        buf610 = reinterpret_tensor(buf578, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf578  # reuse
        # Topologically Sorted Source Nodes: [key_states_27], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf609, arg396_1, buf610, 2097152, grid=grid(2097152), stream=stream0)
        del arg396_1
        buf611 = buf609; del buf609  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg397_1, (1024, 1024), (1, 1024), 0), out=buf611)
        del arg397_1
        buf612 = reinterpret_tensor(buf590, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf590  # reuse
        # Topologically Sorted Source Nodes: [value_states_27], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf611, arg398_1, buf612, 2097152, grid=grid(2097152), stream=stream0)
        del arg398_1
        buf613 = reinterpret_tensor(buf611, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf611  # reuse
        # Topologically Sorted Source Nodes: [query_states_55, attn_output_108], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf608, arg394_1, buf613, 2097152, grid=grid(2097152), stream=stream0)
        del arg394_1
        # Topologically Sorted Source Nodes: [query_states_55, attn_output_108], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf614 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf613, buf610, buf612, None, False)
        buf615 = buf614[0]
        del buf614
        buf619 = reinterpret_tensor(buf613, (2048, 1024), (1024, 1), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf615, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg399_1, (1024, 1024), (1, 1024), 0), out=buf619)
        del arg399_1
        buf623 = reinterpret_tensor(buf615, (16, 128, 1024), (131072, 1024, 1), 0); del buf615  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_202, hidden_states_203], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf603, buf619, arg400_1, arg401_1, arg402_1, buf623, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg401_1
        del arg402_1
        buf624 = reinterpret_tensor(buf584, (2048, 4096), (4096, 1), 0); del buf584  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf623, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg403_1, (1024, 4096), (1, 1024), 0), out=buf624)
        del arg403_1
        buf625 = reinterpret_tensor(buf624, (16, 128, 4096), (524288, 4096, 1), 0); del buf624  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_204], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf625, arg404_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg404_1
        buf626 = reinterpret_tensor(buf623, (2048, 1024), (1024, 1), 0); del buf623  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf625, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg405_1, (4096, 1024), (1, 4096), 0), out=buf626)
        del arg405_1
        buf627 = reinterpret_tensor(buf626, (16, 128, 1024), (131072, 1024, 1), 0); del buf626  # reuse
        buf631 = reinterpret_tensor(buf608, (16, 128, 1024), (131072, 1024, 1), 0); del buf608  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_202, hidden_states_208, hidden_states_209], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf627, buf603, buf619, arg400_1, arg406_1, arg407_1, arg408_1, buf631, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg400_1
        del arg406_1
        del arg407_1
        del arg408_1
        buf632 = buf619; del buf619  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf631, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg409_1, (1024, 1024), (1, 1024), 0), out=buf632)
        del arg409_1
        buf633 = reinterpret_tensor(buf603, (2048, 1024), (1024, 1), 0); del buf603  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf631, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg411_1, (1024, 1024), (1, 1024), 0), out=buf633)
        del arg411_1
        buf634 = empty_strided_cuda((16, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf633, arg412_1, buf634, 2097152, grid=grid(2097152), stream=stream0)
        del arg412_1
        buf635 = buf633; del buf633  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf631, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg413_1, (1024, 1024), (1, 1024), 0), out=buf635)
        del arg413_1
        buf636 = reinterpret_tensor(buf631, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf631  # reuse
        # Topologically Sorted Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf635, arg414_1, buf636, 2097152, grid=grid(2097152), stream=stream0)
        del arg414_1
        buf637 = reinterpret_tensor(buf635, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf635  # reuse
        # Topologically Sorted Source Nodes: [query_states_57, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf632, arg410_1, buf637, 2097152, grid=grid(2097152), stream=stream0)
        del arg410_1
        buf638 = buf596; del buf596  # reuse
        # Topologically Sorted Source Nodes: [query_states_57, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_6.run(buf638, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_57, attn_output_112], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf639 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf637, buf634, buf636, buf638, False)
        buf640 = buf639[0]
        del buf639
        buf644 = reinterpret_tensor(buf637, (2048, 1024), (1024, 1), 0); del buf637  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf640, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg415_1, (1024, 1024), (1, 1024), 0), out=buf644)
        del arg415_1
        buf648 = reinterpret_tensor(buf640, (16, 128, 1024), (131072, 1024, 1), 0); del buf640  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_211, hidden_states_212], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf627, buf644, arg416_1, arg417_1, arg418_1, buf648, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg417_1
        del arg418_1
        buf649 = buf632; del buf632  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf648, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg419_1, (1024, 1024), (1, 1024), 0), out=buf649)
        del arg419_1
        buf650 = reinterpret_tensor(buf648, (2048, 1024), (1024, 1), 0); del buf648  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg421_1, (1024, 1024), (1, 1024), 0), out=buf650)
        del arg421_1
        buf651 = empty_strided_cuda((16, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_29], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf650, arg422_1, buf651, 2097152, grid=grid(2097152), stream=stream0)
        del arg422_1
        buf652 = buf650; del buf650  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg423_1, (1024, 1024), (1, 1024), 0), out=buf652)
        del arg423_1
        buf653 = empty_strided_cuda((16, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [value_states_29], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf652, arg424_1, buf653, 2097152, grid=grid(2097152), stream=stream0)
        del arg424_1
        buf654 = reinterpret_tensor(buf652, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf652  # reuse
        # Topologically Sorted Source Nodes: [query_states_59, attn_output_116], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf649, arg420_1, buf654, 2097152, grid=grid(2097152), stream=stream0)
        del arg420_1
        # Topologically Sorted Source Nodes: [query_states_59, attn_output_116], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf655 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf654, buf651, buf653, None, False)
        buf656 = buf655[0]
        del buf655
        buf660 = reinterpret_tensor(buf654, (2048, 1024), (1024, 1), 0); del buf654  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf656, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg425_1, (1024, 1024), (1, 1024), 0), out=buf660)
        del arg425_1
        buf661 = reinterpret_tensor(buf660, (16, 128, 1024), (131072, 1024, 1), 0); del buf660  # reuse
        buf665 = reinterpret_tensor(buf656, (16, 128, 1024), (131072, 1024, 1), 0); del buf656  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_211, hidden_states_214, hidden_states_215], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf661, buf627, buf644, arg416_1, arg426_1, arg427_1, arg428_1, buf665, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg416_1
        del arg426_1
        del arg427_1
        del arg428_1
        buf666 = reinterpret_tensor(buf625, (2048, 4096), (4096, 1), 0); del buf625  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf665, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg429_1, (1024, 4096), (1, 1024), 0), out=buf666)
        del arg429_1
        buf667 = reinterpret_tensor(buf666, (16, 128, 4096), (524288, 4096, 1), 0); del buf666  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_216], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf667, arg430_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg430_1
        buf668 = reinterpret_tensor(buf665, (2048, 1024), (1024, 1), 0); del buf665  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf667, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg431_1, (4096, 1024), (1, 4096), 0), out=buf668)
        del arg431_1
        buf672 = reinterpret_tensor(buf644, (16, 128, 1024), (131072, 1024, 1), 0); del buf644  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_220, hidden_states_221], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf661, buf668, arg432_1, arg433_1, arg434_1, buf672, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg433_1
        del arg434_1
        buf673 = reinterpret_tensor(buf627, (2048, 1024), (1024, 1), 0); del buf627  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf672, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg435_1, (1024, 1024), (1, 1024), 0), out=buf673)
        del arg435_1
        buf674 = buf649; del buf649  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf672, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg437_1, (1024, 1024), (1, 1024), 0), out=buf674)
        del arg437_1
        buf675 = empty_strided_cuda((16, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_30], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf674, arg438_1, buf675, 2097152, grid=grid(2097152), stream=stream0)
        del arg438_1
        buf676 = buf674; del buf674  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf672, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg439_1, (1024, 1024), (1, 1024), 0), out=buf676)
        del arg439_1
        buf677 = reinterpret_tensor(buf672, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf672  # reuse
        # Topologically Sorted Source Nodes: [value_states_30], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf676, arg440_1, buf677, 2097152, grid=grid(2097152), stream=stream0)
        del arg440_1
        buf678 = reinterpret_tensor(buf676, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf676  # reuse
        # Topologically Sorted Source Nodes: [query_states_61, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf673, arg436_1, buf678, 2097152, grid=grid(2097152), stream=stream0)
        del arg436_1
        buf679 = buf638; del buf638  # reuse
        # Topologically Sorted Source Nodes: [query_states_61, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_6.run(buf679, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_61, attn_output_120], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf680 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf678, buf675, buf677, buf679, False)
        buf681 = buf680[0]
        del buf680
        buf685 = reinterpret_tensor(buf678, (2048, 1024), (1024, 1), 0); del buf678  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf681, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg441_1, (1024, 1024), (1, 1024), 0), out=buf685)
        del arg441_1
        buf686 = reinterpret_tensor(buf685, (16, 128, 1024), (131072, 1024, 1), 0); del buf685  # reuse
        buf690 = reinterpret_tensor(buf681, (16, 128, 1024), (131072, 1024, 1), 0); del buf681  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_220, hidden_states_223, hidden_states_224], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf686, buf661, buf668, arg432_1, arg442_1, arg443_1, arg444_1, buf690, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg432_1
        del arg442_1
        del arg443_1
        del arg444_1
        buf691 = buf668; del buf668  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf690, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg445_1, (1024, 1024), (1, 1024), 0), out=buf691)
        del arg445_1
        buf692 = reinterpret_tensor(buf690, (2048, 1024), (1024, 1), 0); del buf690  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg447_1, (1024, 1024), (1, 1024), 0), out=buf692)
        del arg447_1
        buf693 = reinterpret_tensor(buf661, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf661  # reuse
        # Topologically Sorted Source Nodes: [key_states_31], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf692, arg448_1, buf693, 2097152, grid=grid(2097152), stream=stream0)
        del arg448_1
        buf694 = buf692; del buf692  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg449_1, (1024, 1024), (1, 1024), 0), out=buf694)
        del arg449_1
        buf695 = reinterpret_tensor(buf673, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf673  # reuse
        # Topologically Sorted Source Nodes: [value_states_31], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf694, arg450_1, buf695, 2097152, grid=grid(2097152), stream=stream0)
        del arg450_1
        buf696 = reinterpret_tensor(buf694, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf694  # reuse
        # Topologically Sorted Source Nodes: [query_states_63, attn_output_124], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf691, arg446_1, buf696, 2097152, grid=grid(2097152), stream=stream0)
        del arg446_1
        # Topologically Sorted Source Nodes: [query_states_63, attn_output_124], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf697 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf696, buf693, buf695, None, False)
        buf698 = buf697[0]
        del buf697
        buf702 = reinterpret_tensor(buf696, (2048, 1024), (1024, 1), 0); del buf696  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf698, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg451_1, (1024, 1024), (1, 1024), 0), out=buf702)
        del arg451_1
        buf706 = reinterpret_tensor(buf698, (16, 128, 1024), (131072, 1024, 1), 0); del buf698  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_226, hidden_states_227], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf686, buf702, arg452_1, arg453_1, arg454_1, buf706, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg453_1
        del arg454_1
        buf707 = reinterpret_tensor(buf667, (2048, 4096), (4096, 1), 0); del buf667  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf706, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg455_1, (1024, 4096), (1, 1024), 0), out=buf707)
        del arg455_1
        buf708 = reinterpret_tensor(buf707, (16, 128, 4096), (524288, 4096, 1), 0); del buf707  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_228], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf708, arg456_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg456_1
        buf709 = reinterpret_tensor(buf706, (2048, 1024), (1024, 1), 0); del buf706  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf708, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg457_1, (4096, 1024), (1, 4096), 0), out=buf709)
        del arg457_1
        buf710 = reinterpret_tensor(buf709, (16, 128, 1024), (131072, 1024, 1), 0); del buf709  # reuse
        buf714 = reinterpret_tensor(buf691, (16, 128, 1024), (131072, 1024, 1), 0); del buf691  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_226, hidden_states_232, hidden_states_233], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf710, buf686, buf702, arg452_1, arg458_1, arg459_1, arg460_1, buf714, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg452_1
        del arg458_1
        del arg459_1
        del arg460_1
        buf715 = buf702; del buf702  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf714, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg461_1, (1024, 1024), (1, 1024), 0), out=buf715)
        del arg461_1
        buf716 = reinterpret_tensor(buf686, (2048, 1024), (1024, 1), 0); del buf686  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf714, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg463_1, (1024, 1024), (1, 1024), 0), out=buf716)
        del arg463_1
        buf717 = empty_strided_cuda((16, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf716, arg464_1, buf717, 2097152, grid=grid(2097152), stream=stream0)
        del arg464_1
        buf718 = buf716; del buf716  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf714, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg465_1, (1024, 1024), (1, 1024), 0), out=buf718)
        del arg465_1
        buf719 = reinterpret_tensor(buf714, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf714  # reuse
        # Topologically Sorted Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf718, arg466_1, buf719, 2097152, grid=grid(2097152), stream=stream0)
        del arg466_1
        buf720 = reinterpret_tensor(buf718, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf718  # reuse
        # Topologically Sorted Source Nodes: [query_states_65, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf715, arg462_1, buf720, 2097152, grid=grid(2097152), stream=stream0)
        del arg462_1
        buf721 = buf679; del buf679  # reuse
        # Topologically Sorted Source Nodes: [query_states_65, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_6.run(buf721, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_65, attn_output_128], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf722 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf720, buf717, buf719, buf721, False)
        buf723 = buf722[0]
        del buf722
        buf727 = reinterpret_tensor(buf720, (2048, 1024), (1024, 1), 0); del buf720  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf723, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg467_1, (1024, 1024), (1, 1024), 0), out=buf727)
        del arg467_1
        buf731 = reinterpret_tensor(buf723, (16, 128, 1024), (131072, 1024, 1), 0); del buf723  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_235, hidden_states_236], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf710, buf727, arg468_1, arg469_1, arg470_1, buf731, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg469_1
        del arg470_1
        buf732 = buf715; del buf715  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf731, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg471_1, (1024, 1024), (1, 1024), 0), out=buf732)
        del arg471_1
        buf733 = reinterpret_tensor(buf731, (2048, 1024), (1024, 1), 0); del buf731  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg473_1, (1024, 1024), (1, 1024), 0), out=buf733)
        del arg473_1
        buf734 = empty_strided_cuda((16, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_33], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf733, arg474_1, buf734, 2097152, grid=grid(2097152), stream=stream0)
        del arg474_1
        buf735 = buf733; del buf733  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg475_1, (1024, 1024), (1, 1024), 0), out=buf735)
        del arg475_1
        buf736 = empty_strided_cuda((16, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [value_states_33], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf735, arg476_1, buf736, 2097152, grid=grid(2097152), stream=stream0)
        del arg476_1
        buf737 = reinterpret_tensor(buf735, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf735  # reuse
        # Topologically Sorted Source Nodes: [query_states_67, attn_output_132], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf732, arg472_1, buf737, 2097152, grid=grid(2097152), stream=stream0)
        del arg472_1
        # Topologically Sorted Source Nodes: [query_states_67, attn_output_132], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf738 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf737, buf734, buf736, None, False)
        buf739 = buf738[0]
        del buf738
        buf743 = reinterpret_tensor(buf737, (2048, 1024), (1024, 1), 0); del buf737  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf739, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg477_1, (1024, 1024), (1, 1024), 0), out=buf743)
        del arg477_1
        buf744 = reinterpret_tensor(buf743, (16, 128, 1024), (131072, 1024, 1), 0); del buf743  # reuse
        buf748 = reinterpret_tensor(buf739, (16, 128, 1024), (131072, 1024, 1), 0); del buf739  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_235, hidden_states_238, hidden_states_239], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf744, buf710, buf727, arg468_1, arg478_1, arg479_1, arg480_1, buf748, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg468_1
        del arg478_1
        del arg479_1
        del arg480_1
        buf749 = reinterpret_tensor(buf708, (2048, 4096), (4096, 1), 0); del buf708  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf748, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg481_1, (1024, 4096), (1, 1024), 0), out=buf749)
        del arg481_1
        buf750 = reinterpret_tensor(buf749, (16, 128, 4096), (524288, 4096, 1), 0); del buf749  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_240], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf750, arg482_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg482_1
        buf751 = reinterpret_tensor(buf748, (2048, 1024), (1024, 1), 0); del buf748  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf750, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg483_1, (4096, 1024), (1, 4096), 0), out=buf751)
        del arg483_1
        buf755 = reinterpret_tensor(buf727, (16, 128, 1024), (131072, 1024, 1), 0); del buf727  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_244, hidden_states_245], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf744, buf751, arg484_1, arg485_1, arg486_1, buf755, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg485_1
        del arg486_1
        buf756 = reinterpret_tensor(buf710, (2048, 1024), (1024, 1), 0); del buf710  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf755, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg487_1, (1024, 1024), (1, 1024), 0), out=buf756)
        del arg487_1
        buf757 = buf732; del buf732  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf755, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg489_1, (1024, 1024), (1, 1024), 0), out=buf757)
        del arg489_1
        buf758 = empty_strided_cuda((16, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_34], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf757, arg490_1, buf758, 2097152, grid=grid(2097152), stream=stream0)
        del arg490_1
        buf759 = buf757; del buf757  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf755, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg491_1, (1024, 1024), (1, 1024), 0), out=buf759)
        del arg491_1
        buf760 = reinterpret_tensor(buf755, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf755  # reuse
        # Topologically Sorted Source Nodes: [value_states_34], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf759, arg492_1, buf760, 2097152, grid=grid(2097152), stream=stream0)
        del arg492_1
        buf761 = reinterpret_tensor(buf759, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf759  # reuse
        # Topologically Sorted Source Nodes: [query_states_69, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf756, arg488_1, buf761, 2097152, grid=grid(2097152), stream=stream0)
        del arg488_1
        buf762 = buf721; del buf721  # reuse
        # Topologically Sorted Source Nodes: [query_states_69, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_6.run(buf762, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [query_states_69, attn_output_136], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf763 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf761, buf758, buf760, buf762, False)
        del buf762
        buf764 = buf763[0]
        del buf763
        buf768 = reinterpret_tensor(buf761, (2048, 1024), (1024, 1), 0); del buf761  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf764, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg493_1, (1024, 1024), (1, 1024), 0), out=buf768)
        del arg493_1
        buf769 = reinterpret_tensor(buf768, (16, 128, 1024), (131072, 1024, 1), 0); del buf768  # reuse
        buf773 = reinterpret_tensor(buf764, (16, 128, 1024), (131072, 1024, 1), 0); del buf764  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_244, hidden_states_247, hidden_states_248], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf769, buf744, buf751, arg484_1, arg494_1, arg495_1, arg496_1, buf773, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg484_1
        del arg494_1
        del arg495_1
        del arg496_1
        buf774 = buf751; del buf751  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf773, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg497_1, (1024, 1024), (1, 1024), 0), out=buf774)
        del arg497_1
        buf775 = reinterpret_tensor(buf773, (2048, 1024), (1024, 1), 0); del buf773  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg499_1, (1024, 1024), (1, 1024), 0), out=buf775)
        del arg499_1
        buf776 = reinterpret_tensor(buf744, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf744  # reuse
        # Topologically Sorted Source Nodes: [key_states_35], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf775, arg500_1, buf776, 2097152, grid=grid(2097152), stream=stream0)
        del arg500_1
        buf777 = buf775; del buf775  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg501_1, (1024, 1024), (1, 1024), 0), out=buf777)
        del arg501_1
        buf778 = reinterpret_tensor(buf756, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf756  # reuse
        # Topologically Sorted Source Nodes: [value_states_35], Original ATen: [aten.clone]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf777, arg502_1, buf778, 2097152, grid=grid(2097152), stream=stream0)
        del arg502_1
        buf779 = reinterpret_tensor(buf777, (16, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf777  # reuse
        # Topologically Sorted Source Nodes: [query_states_71, attn_output_140], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_2.run(buf774, arg498_1, buf779, 2097152, grid=grid(2097152), stream=stream0)
        del arg498_1
        # Topologically Sorted Source Nodes: [query_states_71, attn_output_140], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf780 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf779, buf776, buf778, None, False)
        buf781 = buf780[0]
        del buf780
        buf785 = reinterpret_tensor(buf779, (2048, 1024), (1024, 1), 0); del buf779  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf781, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg503_1, (1024, 1024), (1, 1024), 0), out=buf785)
        del arg503_1
        buf789 = reinterpret_tensor(buf781, (16, 128, 1024), (131072, 1024, 1), 0); del buf781  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_250, hidden_states_251], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf769, buf785, arg504_1, arg505_1, arg506_1, buf789, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg505_1
        del arg506_1
        buf790 = reinterpret_tensor(buf750, (2048, 4096), (4096, 1), 0); del buf750  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf789, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg507_1, (1024, 4096), (1, 1024), 0), out=buf790)
        del arg507_1
        buf791 = reinterpret_tensor(buf790, (16, 128, 4096), (524288, 4096, 1), 0); del buf790  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_252], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf791, arg508_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg508_1
        buf792 = reinterpret_tensor(buf789, (2048, 1024), (1024, 1), 0); del buf789  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf791, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg509_1, (4096, 1024), (1, 4096), 0), out=buf792)
        del arg509_1
        del buf791
        buf793 = reinterpret_tensor(buf792, (16, 128, 1024), (131072, 1024, 1), 0); del buf792  # reuse
        buf797 = reinterpret_tensor(buf774, (16, 128, 1024), (131072, 1024, 1), 0); del buf774  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_250, hidden_states_256, hidden_states_257], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf793, buf769, buf785, arg504_1, arg510_1, arg511_1, arg512_1, buf797, 2048, 1024, grid=grid(2048), stream=stream0)
        del arg504_1
        del arg510_1
        del arg511_1
        del arg512_1
        del buf769
        del buf785
        del buf793
        buf798 = empty_strided_cuda((2048, 128112), (128128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf797, (2048, 1024), (1024, 1), 0), reinterpret_tensor(arg2_1, (1024, 128112), (1, 1024), 0), out=buf798)
        del arg2_1
        del buf797
        buf799 = empty_strided_cuda((2048, 1), (1, 2048), torch.float32)
        buf800 = empty_strided_cuda((2048, 1), (1, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_7.run(buf798, buf799, buf800, 2048, 128112, grid=grid(2048), stream=stream0)
        buf801 = empty_strided_cuda((), (), torch.float32)
        buf803 = buf801; del buf801  # reuse
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_8.run(buf803, arg0_1, buf798, buf799, buf800, 1, 2048, grid=grid(1), stream=stream0)
        del arg0_1
        del buf799
        del buf800
    return (buf803, reinterpret_tensor(buf798, (16, 128, 128112), (16400384, 128128, 1), 0), buf301, buf303, buf319, buf321, buf343, buf345, buf361, buf363, buf385, buf387, buf402, buf404, buf426, buf428, buf444, buf446, buf468, buf470, buf485, buf487, buf509, buf511, buf527, buf529, buf551, buf553, buf568, buf570, buf592, buf594, buf610, buf612, buf634, buf636, buf651, buf653, buf675, buf677, buf693, buf695, buf717, buf719, buf734, buf736, buf758, buf760, buf776, buf778, buf317, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((16, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((128112, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
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
    arg198_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
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
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('M2M100ForConditionalGeneration', benchmark_compiled_module)
