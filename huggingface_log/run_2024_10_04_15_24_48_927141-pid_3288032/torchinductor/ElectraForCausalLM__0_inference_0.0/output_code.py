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


# kernel path: /tmp/torchinductor_sahanp/6s/c6stotksvdp3wiu4zsinmbb5q24f7mrgfijhupssr5g3vg76b6f5.py
# Topologically Sorted Source Nodes: [inputs_embeds, token_type_embeddings, embeddings, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embeddings => add
#   embeddings_1 => add_1
#   embeddings_2 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
#   inputs_embeds => embedding
#   position_embeddings => embedding_2
#   token_type_embeddings => embedding_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg1_1, 0), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %expand), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %arg202_1), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg6_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg7_1), kwargs = {})
triton_per_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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


# kernel path: /tmp/torchinductor_sahanp/7t/c7tfgzdzudcsngjmng6vjnprpbf24ls6yv7vylbaorirck6okt56.py
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
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg18_1), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg19_1), kwargs = {})
triton_per_fused_add_native_layer_norm_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_1', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 16384
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


# kernel path: /tmp/torchinductor_sahanp/6n/c6nxmougehasxgu6ppr4xxbczklpowyl7wlzoju5amytqz4ocfea.py
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
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
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


# kernel path: /tmp/torchinductor_sahanp/fg/cfg3f3mgeuh2chhs5gvs565qxr2imozpnin7iukp3wbbljsllwv6.py
# Topologically Sorted Source Nodes: [hidden_states_98, hidden_states_99], Original ATen: [aten.gelu, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_98 => add_100, erf_12, mul_87, mul_88, mul_89
#   hidden_states_99 => add_101, add_102, mul_90, mul_91, rsqrt_25, sub_38, var_mean_25
# Graph fragment:
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_267, 0.5), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_267, 0.7071067811865476), kwargs = {})
#   %erf_12 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_88,), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_12, 1), kwargs = {})
#   %mul_89 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, %add_100), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_89, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_89, %getitem_51), kwargs = {})
#   %add_101 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-12), kwargs = {})
#   %rsqrt_25 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_101,), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %rsqrt_25), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %arg205_1), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %arg206_1), kwargs = {})
triton_per_fused_gelu_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp11 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tmp10 - tmp18
    tmp25 = 128.0
    tmp26 = tmp23 / tmp25
    tmp27 = 1e-12
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp34, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/s7/cs7upn7l7u4uyozs36eakkrj3tytgepgj5wlojunmvrhaqksuy6q.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_134, %full_default_3], 1), kwargs = {})
triton_poi_fused_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3907072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 30524
    x1 = (xindex // 30524)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 30522, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + (128*x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 30524, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x0 + (30528*x1)), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/l3/cl3tkayvp5rmwm43nu6kemecl2lulddw3fqnszizjelg22yxvj7u.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg207_1, %full_default_4],), kwargs = {})
triton_poi_fused_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30524
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 30522, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 30524, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ja/cjatxsfgxjkf4qthy44usaudjnb6v7nys46fem3cfgd5bj7tbx44.py
# Topologically Sorted Source Nodes: [lm_loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   lm_loss => amax_12, exp_12, sub_39, sum_13
# Graph fragment:
#   %amax_12 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_270, [1], True), kwargs = {})
#   %sub_39 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_270, %amax_12), kwargs = {})
#   %exp_12 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_39,), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_12, [1], True), kwargs = {})
triton_red_fused__log_softmax_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[16384, 32768],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16352
    rnumel = 30522
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (30528*(x0 % 511)) + (15630336*(x0 // 511))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (30528*(x0 % 511)) + (15630336*(x0 // 511))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5u/c5uqfunegmusd43tmqs2jebcnlguzbzizxqcknhjka4kcww3yxhh.py
# Topologically Sorted Source Nodes: [lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   lm_loss => full_default_2, ne_1, ne_2, neg, sum_14, sum_15, where_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_271, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_2), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_271, -100), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
triton_red_fused_nll_loss_forward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 8176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (1 + (512*(r1 // 511)) + (8192*x0) + (r1 % 511)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r1 + (8176*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr3 + (r1 + (8176*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = tl.full([XBLOCK, RBLOCK], 30522, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 30522)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp8 < 30522")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (30528*(r1 % 511)) + (15630336*(r1 // 511)) + (250085376*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 - tmp11
        tmp14 = tl_math.log(tmp13)
        tmp15 = tmp12 - tmp14
        tmp16 = -tmp15
        tmp17 = 0.0
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp22 = tmp2.to(tl.int64)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp20, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ye/cyemp4dcdlsmlfvav7uw4h667fxdqsq257yjblsrpzb7qtl5rzgt.py
# Topologically Sorted Source Nodes: [lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   lm_loss => convert_element_type, div_24, full_default_2, ne_1, ne_2, neg, sum_14, sum_15, where_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_271, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_2), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_271, -100), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_14, torch.float32), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_15, %convert_element_type), kwargs = {})
triton_per_fused_nll_loss_forward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {3: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp4 = tl.load(in_ptr1 + (r0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp3 / tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp9, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 512), (512, 1))
    assert_size_stride(arg1_1, (32, 512), (512, 1))
    assert_size_stride(arg2_1, (1, 512), (512, 1))
    assert_size_stride(arg3_1, (30522, 128), (128, 1))
    assert_size_stride(arg4_1, (512, 128), (128, 1))
    assert_size_stride(arg5_1, (2, 128), (128, 1))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (256, 128), (128, 1))
    assert_size_stride(arg9_1, (256, ), (1, ))
    assert_size_stride(arg10_1, (256, 256), (256, 1))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, 256), (256, 1))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (256, 256), (256, 1))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, 256), (256, 1))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (1024, 256), (256, 1))
    assert_size_stride(arg21_1, (1024, ), (1, ))
    assert_size_stride(arg22_1, (256, 1024), (1024, 1))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, 256), (256, 1))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (256, 256), (256, 1))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (256, 256), (256, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, 256), (256, 1))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (1024, 256), (256, 1))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (256, 1024), (1024, 1))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (256, 256), (256, 1))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, 256), (256, 1))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (256, 256), (256, 1))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (256, 256), (256, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (1024, 256), (256, 1))
    assert_size_stride(arg53_1, (1024, ), (1, ))
    assert_size_stride(arg54_1, (256, 1024), (1024, 1))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (256, 256), (256, 1))
    assert_size_stride(arg59_1, (256, ), (1, ))
    assert_size_stride(arg60_1, (256, 256), (256, 1))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (256, 256), (256, 1))
    assert_size_stride(arg63_1, (256, ), (1, ))
    assert_size_stride(arg64_1, (256, 256), (256, 1))
    assert_size_stride(arg65_1, (256, ), (1, ))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg67_1, (256, ), (1, ))
    assert_size_stride(arg68_1, (1024, 256), (256, 1))
    assert_size_stride(arg69_1, (1024, ), (1, ))
    assert_size_stride(arg70_1, (256, 1024), (1024, 1))
    assert_size_stride(arg71_1, (256, ), (1, ))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, 256), (256, 1))
    assert_size_stride(arg75_1, (256, ), (1, ))
    assert_size_stride(arg76_1, (256, 256), (256, 1))
    assert_size_stride(arg77_1, (256, ), (1, ))
    assert_size_stride(arg78_1, (256, 256), (256, 1))
    assert_size_stride(arg79_1, (256, ), (1, ))
    assert_size_stride(arg80_1, (256, 256), (256, 1))
    assert_size_stride(arg81_1, (256, ), (1, ))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (256, ), (1, ))
    assert_size_stride(arg84_1, (1024, 256), (256, 1))
    assert_size_stride(arg85_1, (1024, ), (1, ))
    assert_size_stride(arg86_1, (256, 1024), (1024, 1))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (256, 256), (256, 1))
    assert_size_stride(arg91_1, (256, ), (1, ))
    assert_size_stride(arg92_1, (256, 256), (256, 1))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (256, 256), (256, 1))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (256, 256), (256, 1))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (256, ), (1, ))
    assert_size_stride(arg100_1, (1024, 256), (256, 1))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (256, 1024), (1024, 1))
    assert_size_stride(arg103_1, (256, ), (1, ))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, ), (1, ))
    assert_size_stride(arg106_1, (256, 256), (256, 1))
    assert_size_stride(arg107_1, (256, ), (1, ))
    assert_size_stride(arg108_1, (256, 256), (256, 1))
    assert_size_stride(arg109_1, (256, ), (1, ))
    assert_size_stride(arg110_1, (256, 256), (256, 1))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (256, 256), (256, 1))
    assert_size_stride(arg113_1, (256, ), (1, ))
    assert_size_stride(arg114_1, (256, ), (1, ))
    assert_size_stride(arg115_1, (256, ), (1, ))
    assert_size_stride(arg116_1, (1024, 256), (256, 1))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (256, 1024), (1024, 1))
    assert_size_stride(arg119_1, (256, ), (1, ))
    assert_size_stride(arg120_1, (256, ), (1, ))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, 256), (256, 1))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (256, 256), (256, 1))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (256, 256), (256, 1))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (256, 256), (256, 1))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (256, ), (1, ))
    assert_size_stride(arg131_1, (256, ), (1, ))
    assert_size_stride(arg132_1, (1024, 256), (256, 1))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (256, 1024), (1024, 1))
    assert_size_stride(arg135_1, (256, ), (1, ))
    assert_size_stride(arg136_1, (256, ), (1, ))
    assert_size_stride(arg137_1, (256, ), (1, ))
    assert_size_stride(arg138_1, (256, 256), (256, 1))
    assert_size_stride(arg139_1, (256, ), (1, ))
    assert_size_stride(arg140_1, (256, 256), (256, 1))
    assert_size_stride(arg141_1, (256, ), (1, ))
    assert_size_stride(arg142_1, (256, 256), (256, 1))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, 256), (256, 1))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, ), (1, ))
    assert_size_stride(arg147_1, (256, ), (1, ))
    assert_size_stride(arg148_1, (1024, 256), (256, 1))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (256, 1024), (1024, 1))
    assert_size_stride(arg151_1, (256, ), (1, ))
    assert_size_stride(arg152_1, (256, ), (1, ))
    assert_size_stride(arg153_1, (256, ), (1, ))
    assert_size_stride(arg154_1, (256, 256), (256, 1))
    assert_size_stride(arg155_1, (256, ), (1, ))
    assert_size_stride(arg156_1, (256, 256), (256, 1))
    assert_size_stride(arg157_1, (256, ), (1, ))
    assert_size_stride(arg158_1, (256, 256), (256, 1))
    assert_size_stride(arg159_1, (256, ), (1, ))
    assert_size_stride(arg160_1, (256, 256), (256, 1))
    assert_size_stride(arg161_1, (256, ), (1, ))
    assert_size_stride(arg162_1, (256, ), (1, ))
    assert_size_stride(arg163_1, (256, ), (1, ))
    assert_size_stride(arg164_1, (1024, 256), (256, 1))
    assert_size_stride(arg165_1, (1024, ), (1, ))
    assert_size_stride(arg166_1, (256, 1024), (1024, 1))
    assert_size_stride(arg167_1, (256, ), (1, ))
    assert_size_stride(arg168_1, (256, ), (1, ))
    assert_size_stride(arg169_1, (256, ), (1, ))
    assert_size_stride(arg170_1, (256, 256), (256, 1))
    assert_size_stride(arg171_1, (256, ), (1, ))
    assert_size_stride(arg172_1, (256, 256), (256, 1))
    assert_size_stride(arg173_1, (256, ), (1, ))
    assert_size_stride(arg174_1, (256, 256), (256, 1))
    assert_size_stride(arg175_1, (256, ), (1, ))
    assert_size_stride(arg176_1, (256, 256), (256, 1))
    assert_size_stride(arg177_1, (256, ), (1, ))
    assert_size_stride(arg178_1, (256, ), (1, ))
    assert_size_stride(arg179_1, (256, ), (1, ))
    assert_size_stride(arg180_1, (1024, 256), (256, 1))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (256, 1024), (1024, 1))
    assert_size_stride(arg183_1, (256, ), (1, ))
    assert_size_stride(arg184_1, (256, ), (1, ))
    assert_size_stride(arg185_1, (256, ), (1, ))
    assert_size_stride(arg186_1, (256, 256), (256, 1))
    assert_size_stride(arg187_1, (256, ), (1, ))
    assert_size_stride(arg188_1, (256, 256), (256, 1))
    assert_size_stride(arg189_1, (256, ), (1, ))
    assert_size_stride(arg190_1, (256, 256), (256, 1))
    assert_size_stride(arg191_1, (256, ), (1, ))
    assert_size_stride(arg192_1, (256, 256), (256, 1))
    assert_size_stride(arg193_1, (256, ), (1, ))
    assert_size_stride(arg194_1, (256, ), (1, ))
    assert_size_stride(arg195_1, (256, ), (1, ))
    assert_size_stride(arg196_1, (1024, 256), (256, 1))
    assert_size_stride(arg197_1, (1024, ), (1, ))
    assert_size_stride(arg198_1, (256, 1024), (1024, 1))
    assert_size_stride(arg199_1, (256, ), (1, ))
    assert_size_stride(arg200_1, (256, ), (1, ))
    assert_size_stride(arg201_1, (256, ), (1, ))
    assert_size_stride(arg202_1, (1, 512), (512, 1))
    assert_size_stride(arg203_1, (128, 256), (256, 1))
    assert_size_stride(arg204_1, (128, ), (1, ))
    assert_size_stride(arg205_1, (128, ), (1, ))
    assert_size_stride(arg206_1, (128, ), (1, ))
    assert_size_stride(arg207_1, (30522, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 512, 128), (65536, 128, 1), torch.float32)
        buf4 = empty_strided_cuda((32, 512, 128), (65536, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, token_type_embeddings, embeddings, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg1_1, arg3_1, arg2_1, arg5_1, arg202_1, arg4_1, arg6_1, arg7_1, buf0, buf4, 16384, 128, grid=grid(16384), stream=stream0)
        del arg1_1
        del arg202_1
        del arg2_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        buf5 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg9_1, reinterpret_tensor(buf4, (16384, 128), (128, 1), 0), reinterpret_tensor(arg8_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del arg8_1
        del arg9_1
        buf6 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, buf5, reinterpret_tensor(arg10_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf6)
        del arg10_1
        del arg11_1
        buf7 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, buf5, reinterpret_tensor(arg12_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf7)
        del arg12_1
        del arg13_1
        buf8 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, buf5, reinterpret_tensor(arg14_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf8)
        del arg14_1
        del arg15_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf6, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf7, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf8, (32, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf6
        buf10 = buf9[0]
        del buf9
        buf14 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf10, (16384, 256), (256, 1), 0), reinterpret_tensor(arg16_1, (256, 256), (1, 256), 0), out=buf14)
        del arg16_1
        buf18 = reinterpret_tensor(buf10, (32, 512, 256), (131072, 256, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [add_2, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf14, arg17_1, buf5, arg18_1, arg19_1, buf18, 16384, 256, grid=grid(16384), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        buf19 = empty_strided_cuda((16384, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (16384, 256), (256, 1), 0), reinterpret_tensor(arg20_1, (256, 1024), (1, 256), 0), out=buf19)
        del arg20_1
        buf20 = reinterpret_tensor(buf19, (32, 512, 1024), (524288, 1024, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf20, arg21_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg21_1
        buf21 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg22_1, (1024, 256), (1, 1024), 0), out=buf21)
        del arg22_1
        buf25 = reinterpret_tensor(buf14, (32, 512, 256), (131072, 256, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [add_3, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf21, arg23_1, buf18, arg24_1, arg25_1, buf25, 16384, 256, grid=grid(16384), stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        buf26 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf25, (16384, 256), (256, 1), 0), reinterpret_tensor(arg26_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf26)
        del arg26_1
        del arg27_1
        buf27 = reinterpret_tensor(buf18, (16384, 256), (256, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg29_1, reinterpret_tensor(buf25, (16384, 256), (256, 1), 0), reinterpret_tensor(arg28_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf27)
        del arg28_1
        del arg29_1
        buf28 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg31_1, reinterpret_tensor(buf25, (16384, 256), (256, 1), 0), reinterpret_tensor(arg30_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf28)
        del arg30_1
        del arg31_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf29 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf26, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf27, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf28, (32, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf26
        buf30 = buf29[0]
        del buf29
        buf34 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (16384, 256), (256, 1), 0), reinterpret_tensor(arg32_1, (256, 256), (1, 256), 0), out=buf34)
        del arg32_1
        buf38 = reinterpret_tensor(buf30, (32, 512, 256), (131072, 256, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [add_5, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf34, arg33_1, buf25, arg34_1, arg35_1, buf38, 16384, 256, grid=grid(16384), stream=stream0)
        del arg33_1
        del arg34_1
        del arg35_1
        buf39 = reinterpret_tensor(buf20, (16384, 1024), (1024, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (16384, 256), (256, 1), 0), reinterpret_tensor(arg36_1, (256, 1024), (1, 256), 0), out=buf39)
        del arg36_1
        buf40 = reinterpret_tensor(buf39, (32, 512, 1024), (524288, 1024, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_13], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf40, arg37_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg37_1
        buf41 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg38_1, (1024, 256), (1, 1024), 0), out=buf41)
        del arg38_1
        buf45 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [add_6, hidden_states_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf41, arg39_1, buf38, arg40_1, arg41_1, buf45, 16384, 256, grid=grid(16384), stream=stream0)
        del arg39_1
        del arg40_1
        del arg41_1
        buf46 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg43_1, reinterpret_tensor(buf45, (16384, 256), (256, 1), 0), reinterpret_tensor(arg42_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf46)
        del arg42_1
        del arg43_1
        buf47 = reinterpret_tensor(buf38, (16384, 256), (256, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg45_1, reinterpret_tensor(buf45, (16384, 256), (256, 1), 0), reinterpret_tensor(arg44_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf47)
        del arg44_1
        del arg45_1
        buf48 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg47_1, reinterpret_tensor(buf45, (16384, 256), (256, 1), 0), reinterpret_tensor(arg46_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf48)
        del arg46_1
        del arg47_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf49 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf46, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf47, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf48, (32, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf46
        buf50 = buf49[0]
        del buf49
        buf54 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf50, (16384, 256), (256, 1), 0), reinterpret_tensor(arg48_1, (256, 256), (1, 256), 0), out=buf54)
        del arg48_1
        buf58 = reinterpret_tensor(buf50, (32, 512, 256), (131072, 256, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [add_8, hidden_states_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf54, arg49_1, buf45, arg50_1, arg51_1, buf58, 16384, 256, grid=grid(16384), stream=stream0)
        del arg49_1
        del arg50_1
        del arg51_1
        buf59 = reinterpret_tensor(buf40, (16384, 1024), (1024, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (16384, 256), (256, 1), 0), reinterpret_tensor(arg52_1, (256, 1024), (1, 256), 0), out=buf59)
        del arg52_1
        buf60 = reinterpret_tensor(buf59, (32, 512, 1024), (524288, 1024, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf60, arg53_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg53_1
        buf61 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg54_1, (1024, 256), (1, 1024), 0), out=buf61)
        del arg54_1
        buf65 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [add_9, hidden_states_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf61, arg55_1, buf58, arg56_1, arg57_1, buf65, 16384, 256, grid=grid(16384), stream=stream0)
        del arg55_1
        del arg56_1
        del arg57_1
        buf66 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg59_1, reinterpret_tensor(buf65, (16384, 256), (256, 1), 0), reinterpret_tensor(arg58_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf66)
        del arg58_1
        del arg59_1
        buf67 = reinterpret_tensor(buf58, (16384, 256), (256, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg61_1, reinterpret_tensor(buf65, (16384, 256), (256, 1), 0), reinterpret_tensor(arg60_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf67)
        del arg60_1
        del arg61_1
        buf68 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg63_1, reinterpret_tensor(buf65, (16384, 256), (256, 1), 0), reinterpret_tensor(arg62_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf68)
        del arg62_1
        del arg63_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf69 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf66, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf67, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf68, (32, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf66
        buf70 = buf69[0]
        del buf69
        buf74 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (16384, 256), (256, 1), 0), reinterpret_tensor(arg64_1, (256, 256), (1, 256), 0), out=buf74)
        del arg64_1
        buf78 = reinterpret_tensor(buf70, (32, 512, 256), (131072, 256, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [add_11, hidden_states_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf74, arg65_1, buf65, arg66_1, arg67_1, buf78, 16384, 256, grid=grid(16384), stream=stream0)
        del arg65_1
        del arg66_1
        del arg67_1
        buf79 = reinterpret_tensor(buf60, (16384, 1024), (1024, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (16384, 256), (256, 1), 0), reinterpret_tensor(arg68_1, (256, 1024), (1, 256), 0), out=buf79)
        del arg68_1
        buf80 = reinterpret_tensor(buf79, (32, 512, 1024), (524288, 1024, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf80, arg69_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg69_1
        buf81 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg70_1, (1024, 256), (1, 1024), 0), out=buf81)
        del arg70_1
        buf85 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [add_12, hidden_states_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf81, arg71_1, buf78, arg72_1, arg73_1, buf85, 16384, 256, grid=grid(16384), stream=stream0)
        del arg71_1
        del arg72_1
        del arg73_1
        buf86 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg75_1, reinterpret_tensor(buf85, (16384, 256), (256, 1), 0), reinterpret_tensor(arg74_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf86)
        del arg74_1
        del arg75_1
        buf87 = reinterpret_tensor(buf78, (16384, 256), (256, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg77_1, reinterpret_tensor(buf85, (16384, 256), (256, 1), 0), reinterpret_tensor(arg76_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf87)
        del arg76_1
        del arg77_1
        buf88 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg79_1, reinterpret_tensor(buf85, (16384, 256), (256, 1), 0), reinterpret_tensor(arg78_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf88)
        del arg78_1
        del arg79_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf89 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf86, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf87, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf88, (32, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf86
        buf90 = buf89[0]
        del buf89
        buf94 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (16384, 256), (256, 1), 0), reinterpret_tensor(arg80_1, (256, 256), (1, 256), 0), out=buf94)
        del arg80_1
        buf98 = reinterpret_tensor(buf90, (32, 512, 256), (131072, 256, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [add_14, hidden_states_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf94, arg81_1, buf85, arg82_1, arg83_1, buf98, 16384, 256, grid=grid(16384), stream=stream0)
        del arg81_1
        del arg82_1
        del arg83_1
        buf99 = reinterpret_tensor(buf80, (16384, 1024), (1024, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (16384, 256), (256, 1), 0), reinterpret_tensor(arg84_1, (256, 1024), (1, 256), 0), out=buf99)
        del arg84_1
        buf100 = reinterpret_tensor(buf99, (32, 512, 1024), (524288, 1024, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf100, arg85_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg85_1
        buf101 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg86_1, (1024, 256), (1, 1024), 0), out=buf101)
        del arg86_1
        buf105 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [add_15, hidden_states_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf101, arg87_1, buf98, arg88_1, arg89_1, buf105, 16384, 256, grid=grid(16384), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        buf106 = reinterpret_tensor(buf98, (16384, 256), (256, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg91_1, reinterpret_tensor(buf105, (16384, 256), (256, 1), 0), reinterpret_tensor(arg90_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf106)
        del arg90_1
        del arg91_1
        buf107 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg93_1, reinterpret_tensor(buf105, (16384, 256), (256, 1), 0), reinterpret_tensor(arg92_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf107)
        del arg92_1
        del arg93_1
        buf108 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg95_1, reinterpret_tensor(buf105, (16384, 256), (256, 1), 0), reinterpret_tensor(arg94_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf108)
        del arg94_1
        del arg95_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf109 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf106, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf107, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf108, (32, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf106
        buf110 = buf109[0]
        del buf109
        buf114 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (16384, 256), (256, 1), 0), reinterpret_tensor(arg96_1, (256, 256), (1, 256), 0), out=buf114)
        del arg96_1
        buf118 = reinterpret_tensor(buf110, (32, 512, 256), (131072, 256, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [add_17, hidden_states_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf114, arg97_1, buf105, arg98_1, arg99_1, buf118, 16384, 256, grid=grid(16384), stream=stream0)
        del arg97_1
        del arg98_1
        del arg99_1
        buf119 = reinterpret_tensor(buf100, (16384, 1024), (1024, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (16384, 256), (256, 1), 0), reinterpret_tensor(arg100_1, (256, 1024), (1, 256), 0), out=buf119)
        del arg100_1
        buf120 = reinterpret_tensor(buf119, (32, 512, 1024), (524288, 1024, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf120, arg101_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg101_1
        buf121 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg102_1, (1024, 256), (1, 1024), 0), out=buf121)
        del arg102_1
        buf125 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [add_18, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf121, arg103_1, buf118, arg104_1, arg105_1, buf125, 16384, 256, grid=grid(16384), stream=stream0)
        del arg103_1
        del arg104_1
        del arg105_1
        buf126 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg107_1, reinterpret_tensor(buf125, (16384, 256), (256, 1), 0), reinterpret_tensor(arg106_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf126)
        del arg106_1
        del arg107_1
        buf127 = reinterpret_tensor(buf118, (16384, 256), (256, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg109_1, reinterpret_tensor(buf125, (16384, 256), (256, 1), 0), reinterpret_tensor(arg108_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf127)
        del arg108_1
        del arg109_1
        buf128 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg111_1, reinterpret_tensor(buf125, (16384, 256), (256, 1), 0), reinterpret_tensor(arg110_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf128)
        del arg110_1
        del arg111_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf129 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf126, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf127, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf128, (32, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf126
        buf130 = buf129[0]
        del buf129
        buf134 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (16384, 256), (256, 1), 0), reinterpret_tensor(arg112_1, (256, 256), (1, 256), 0), out=buf134)
        del arg112_1
        buf138 = reinterpret_tensor(buf130, (32, 512, 256), (131072, 256, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [add_20, hidden_states_51], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf134, arg113_1, buf125, arg114_1, arg115_1, buf138, 16384, 256, grid=grid(16384), stream=stream0)
        del arg113_1
        del arg114_1
        del arg115_1
        buf139 = reinterpret_tensor(buf120, (16384, 1024), (1024, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (16384, 256), (256, 1), 0), reinterpret_tensor(arg116_1, (256, 1024), (1, 256), 0), out=buf139)
        del arg116_1
        buf140 = reinterpret_tensor(buf139, (32, 512, 1024), (524288, 1024, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_53], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf140, arg117_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg117_1
        buf141 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg118_1, (1024, 256), (1, 1024), 0), out=buf141)
        del arg118_1
        buf145 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [add_21, hidden_states_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf141, arg119_1, buf138, arg120_1, arg121_1, buf145, 16384, 256, grid=grid(16384), stream=stream0)
        del arg119_1
        del arg120_1
        del arg121_1
        buf146 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg123_1, reinterpret_tensor(buf145, (16384, 256), (256, 1), 0), reinterpret_tensor(arg122_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf146)
        del arg122_1
        del arg123_1
        buf147 = reinterpret_tensor(buf138, (16384, 256), (256, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg125_1, reinterpret_tensor(buf145, (16384, 256), (256, 1), 0), reinterpret_tensor(arg124_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf147)
        del arg124_1
        del arg125_1
        buf148 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg127_1, reinterpret_tensor(buf145, (16384, 256), (256, 1), 0), reinterpret_tensor(arg126_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf148)
        del arg126_1
        del arg127_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf149 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf146, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf147, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf148, (32, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf146
        buf150 = buf149[0]
        del buf149
        buf154 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf150, (16384, 256), (256, 1), 0), reinterpret_tensor(arg128_1, (256, 256), (1, 256), 0), out=buf154)
        del arg128_1
        buf158 = reinterpret_tensor(buf150, (32, 512, 256), (131072, 256, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [add_23, hidden_states_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf154, arg129_1, buf145, arg130_1, arg131_1, buf158, 16384, 256, grid=grid(16384), stream=stream0)
        del arg129_1
        del arg130_1
        del arg131_1
        buf159 = reinterpret_tensor(buf140, (16384, 1024), (1024, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (16384, 256), (256, 1), 0), reinterpret_tensor(arg132_1, (256, 1024), (1, 256), 0), out=buf159)
        del arg132_1
        buf160 = reinterpret_tensor(buf159, (32, 512, 1024), (524288, 1024, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf160, arg133_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg133_1
        buf161 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg134_1, (1024, 256), (1, 1024), 0), out=buf161)
        del arg134_1
        buf165 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [add_24, hidden_states_64], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf161, arg135_1, buf158, arg136_1, arg137_1, buf165, 16384, 256, grid=grid(16384), stream=stream0)
        del arg135_1
        del arg136_1
        del arg137_1
        buf166 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg139_1, reinterpret_tensor(buf165, (16384, 256), (256, 1), 0), reinterpret_tensor(arg138_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf166)
        del arg138_1
        del arg139_1
        buf167 = reinterpret_tensor(buf158, (16384, 256), (256, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg141_1, reinterpret_tensor(buf165, (16384, 256), (256, 1), 0), reinterpret_tensor(arg140_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf167)
        del arg140_1
        del arg141_1
        buf168 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg143_1, reinterpret_tensor(buf165, (16384, 256), (256, 1), 0), reinterpret_tensor(arg142_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf168)
        del arg142_1
        del arg143_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf169 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf166, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf167, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf168, (32, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf166
        buf170 = buf169[0]
        del buf169
        buf174 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (16384, 256), (256, 1), 0), reinterpret_tensor(arg144_1, (256, 256), (1, 256), 0), out=buf174)
        del arg144_1
        buf178 = reinterpret_tensor(buf170, (32, 512, 256), (131072, 256, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [add_26, hidden_states_67], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf174, arg145_1, buf165, arg146_1, arg147_1, buf178, 16384, 256, grid=grid(16384), stream=stream0)
        del arg145_1
        del arg146_1
        del arg147_1
        buf179 = reinterpret_tensor(buf160, (16384, 1024), (1024, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (16384, 256), (256, 1), 0), reinterpret_tensor(arg148_1, (256, 1024), (1, 256), 0), out=buf179)
        del arg148_1
        buf180 = reinterpret_tensor(buf179, (32, 512, 1024), (524288, 1024, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf180, arg149_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg149_1
        buf181 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg150_1, (1024, 256), (1, 1024), 0), out=buf181)
        del arg150_1
        buf185 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [add_27, hidden_states_72], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf181, arg151_1, buf178, arg152_1, arg153_1, buf185, 16384, 256, grid=grid(16384), stream=stream0)
        del arg151_1
        del arg152_1
        del arg153_1
        buf186 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg155_1, reinterpret_tensor(buf185, (16384, 256), (256, 1), 0), reinterpret_tensor(arg154_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf186)
        del arg154_1
        del arg155_1
        buf187 = reinterpret_tensor(buf178, (16384, 256), (256, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg157_1, reinterpret_tensor(buf185, (16384, 256), (256, 1), 0), reinterpret_tensor(arg156_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf187)
        del arg156_1
        del arg157_1
        buf188 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg159_1, reinterpret_tensor(buf185, (16384, 256), (256, 1), 0), reinterpret_tensor(arg158_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf188)
        del arg158_1
        del arg159_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf189 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf186, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf187, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf188, (32, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf186
        buf190 = buf189[0]
        del buf189
        buf194 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (16384, 256), (256, 1), 0), reinterpret_tensor(arg160_1, (256, 256), (1, 256), 0), out=buf194)
        del arg160_1
        buf198 = reinterpret_tensor(buf190, (32, 512, 256), (131072, 256, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [add_29, hidden_states_75], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf194, arg161_1, buf185, arg162_1, arg163_1, buf198, 16384, 256, grid=grid(16384), stream=stream0)
        del arg161_1
        del arg162_1
        del arg163_1
        buf199 = reinterpret_tensor(buf180, (16384, 1024), (1024, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (16384, 256), (256, 1), 0), reinterpret_tensor(arg164_1, (256, 1024), (1, 256), 0), out=buf199)
        del arg164_1
        buf200 = reinterpret_tensor(buf199, (32, 512, 1024), (524288, 1024, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf200, arg165_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg165_1
        buf201 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg166_1, (1024, 256), (1, 1024), 0), out=buf201)
        del arg166_1
        buf205 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [add_30, hidden_states_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf201, arg167_1, buf198, arg168_1, arg169_1, buf205, 16384, 256, grid=grid(16384), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        buf206 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg171_1, reinterpret_tensor(buf205, (16384, 256), (256, 1), 0), reinterpret_tensor(arg170_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf206)
        del arg170_1
        del arg171_1
        buf207 = reinterpret_tensor(buf198, (16384, 256), (256, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg173_1, reinterpret_tensor(buf205, (16384, 256), (256, 1), 0), reinterpret_tensor(arg172_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf207)
        del arg172_1
        del arg173_1
        buf208 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg175_1, reinterpret_tensor(buf205, (16384, 256), (256, 1), 0), reinterpret_tensor(arg174_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf208)
        del arg174_1
        del arg175_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf209 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf206, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf207, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf208, (32, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf206
        buf210 = buf209[0]
        del buf209
        buf214 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (16384, 256), (256, 1), 0), reinterpret_tensor(arg176_1, (256, 256), (1, 256), 0), out=buf214)
        del arg176_1
        buf218 = reinterpret_tensor(buf210, (32, 512, 256), (131072, 256, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [add_32, hidden_states_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf214, arg177_1, buf205, arg178_1, arg179_1, buf218, 16384, 256, grid=grid(16384), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        buf219 = reinterpret_tensor(buf200, (16384, 1024), (1024, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (16384, 256), (256, 1), 0), reinterpret_tensor(arg180_1, (256, 1024), (1, 256), 0), out=buf219)
        del arg180_1
        buf220 = reinterpret_tensor(buf219, (32, 512, 1024), (524288, 1024, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf220, arg181_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg181_1
        buf221 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg182_1, (1024, 256), (1, 1024), 0), out=buf221)
        del arg182_1
        buf225 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [add_33, hidden_states_88], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf221, arg183_1, buf218, arg184_1, arg185_1, buf225, 16384, 256, grid=grid(16384), stream=stream0)
        del arg183_1
        del arg184_1
        del arg185_1
        buf226 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg187_1, reinterpret_tensor(buf225, (16384, 256), (256, 1), 0), reinterpret_tensor(arg186_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf226)
        del arg186_1
        del arg187_1
        buf227 = reinterpret_tensor(buf218, (16384, 256), (256, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg189_1, reinterpret_tensor(buf225, (16384, 256), (256, 1), 0), reinterpret_tensor(arg188_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf227)
        del arg188_1
        del arg189_1
        buf228 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [linear_69], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg191_1, reinterpret_tensor(buf225, (16384, 256), (256, 1), 0), reinterpret_tensor(arg190_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf228)
        del arg190_1
        del arg191_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf229 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf226, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf227, (32, 4, 512, 64), (131072, 64, 256, 1), 0), reinterpret_tensor(buf228, (32, 4, 512, 64), (131072, 64, 256, 1), 0), None, False, scale=0.125)
        del buf226
        del buf227
        buf230 = buf229[0]
        del buf229
        buf234 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (16384, 256), (256, 1), 0), reinterpret_tensor(arg192_1, (256, 256), (1, 256), 0), out=buf234)
        del arg192_1
        buf238 = reinterpret_tensor(buf230, (32, 512, 256), (131072, 256, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [add_35, hidden_states_91], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf234, arg193_1, buf225, arg194_1, arg195_1, buf238, 16384, 256, grid=grid(16384), stream=stream0)
        del arg193_1
        del arg194_1
        del arg195_1
        buf239 = reinterpret_tensor(buf220, (16384, 1024), (1024, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (16384, 256), (256, 1), 0), reinterpret_tensor(arg196_1, (256, 1024), (1, 256), 0), out=buf239)
        del arg196_1
        buf240 = reinterpret_tensor(buf239, (32, 512, 1024), (524288, 1024, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf240, arg197_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg197_1
        buf241 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg198_1, (1024, 256), (1, 1024), 0), out=buf241)
        del arg198_1
        del buf240
        buf245 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [add_36, hidden_states_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf241, arg199_1, buf238, arg200_1, arg201_1, buf245, 16384, 256, grid=grid(16384), stream=stream0)
        del arg199_1
        del arg200_1
        del arg201_1
        del buf238
        del buf241
        buf246 = reinterpret_tensor(buf4, (16384, 128), (128, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (16384, 256), (256, 1), 0), reinterpret_tensor(arg203_1, (256, 128), (1, 256), 0), out=buf246)
        del arg203_1
        del buf245
        buf250 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_98, hidden_states_99], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_3.run(buf246, arg204_1, arg205_1, arg206_1, buf250, 16384, 128, grid=grid(16384), stream=stream0)
        del arg204_1
        del arg205_1
        del arg206_1
        del buf246
        buf251 = empty_strided_cuda((128, 30524), (30528, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(arg3_1, buf251, 3907072, grid=grid(3907072), stream=stream0)
        del arg3_1
        buf252 = empty_strided_cuda((30524, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(arg207_1, buf252, 30524, grid=grid(30524), stream=stream0)
        del arg207_1
        buf253 = empty_strided_cuda((16384, 30524), (30528, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(buf252, reinterpret_tensor(buf250, (16384, 128), (128, 1), 0), buf251, alpha=1, beta=1, out=buf253)
        del buf250
        del buf251
        del buf252
        buf254 = empty_strided_cuda((16352, 1), (1, 16352), torch.float32)
        buf255 = empty_strided_cuda((16352, 1), (1, 16352), torch.float32)
        # Topologically Sorted Source Nodes: [lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_6.run(buf253, buf254, buf255, 16352, 30522, grid=grid(16352), stream=stream0)
        buf256 = empty_strided_cuda((2, ), (1, ), torch.float32)
        buf258 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_7.run(arg0_1, buf253, buf254, buf255, buf256, buf258, 2, 8176, grid=grid(2), stream=stream0)
        del arg0_1
        del buf254
        del buf255
        buf257 = empty_strided_cuda((), (), torch.float32)
        buf260 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_8.run(buf260, buf256, buf258, 1, 2, grid=grid(1), stream=stream0)
        del buf256
        del buf258
    return (buf260, reinterpret_tensor(buf253, (32, 512, 30522), (15630336, 30528, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((30522, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg203_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ElectraForCausalLM', benchmark_compiled_module)
