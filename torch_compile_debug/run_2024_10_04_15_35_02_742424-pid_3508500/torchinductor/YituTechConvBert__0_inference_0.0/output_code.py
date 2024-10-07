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


# kernel path: /tmp/torchinductor_sahanp/m3/cm3xacpfuixw33c65sjmynt2eekphlqcit2gw65szh6wkkr46g6n.py
# Topologically Sorted Source Nodes: [inputs_embeds, position_embeddings, add, token_type_embeddings, embeddings, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add
#   embeddings => add_1
#   embeddings_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
#   inputs_embeds => embedding
#   position_embeddings => embedding_1
#   token_type_embeddings => embedding_2
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %arg0_1, 0), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg283_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %expand), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   %add_3 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg6_1), kwargs = {})
triton_per_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x3 = xindex
    r2 = rindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([RBLOCK], 30522, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 30522), "index out of bounds: 0 <= tmp4 < 30522")
    tmp6 = tl.load(in_ptr1 + (r2 + (768*tmp4)), rmask, other=0.0)
    tmp8 = tl.full([RBLOCK], 512, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 512), "index out of bounds: 0 <= tmp11 < 512")
    tmp13 = tl.load(in_ptr3 + (r2 + (768*tmp11)), rmask, other=0.0)
    tmp14 = tmp6 + tmp13
    tmp16 = tl.full([RBLOCK], 2, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert((0 <= tmp19) & (tmp19 < 2), "index out of bounds: 0 <= tmp19 < 2")
    tmp21 = tl.load(in_ptr5 + (r2 + (768*tmp19)), rmask, other=0.0)
    tmp22 = tmp14 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tl.full([1], 768, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = tmp22 - tmp32
    tmp40 = 768.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-12
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp22, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp49, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yx/cyx5igvl7zbiughawezyv65rqpqdqdrk4fyvi7o5gxlglmkzea7c.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_3, %arg14_1, None, [1], [4], [1], False, [0], 768), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (393216*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/df/cdflt5z6uqbwbkcy4fmv3jian2haata5m2p4syx5oqamchcabr4y.py
# Topologically Sorted Source Nodes: [conv_attn_layer, conv_kernel_layer], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   conv_attn_layer => mul_3
#   conv_kernel_layer => clone_1
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_8, %view_1), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (196608*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (384*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (384*y3)), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7z/c7z44nvgkgqu3gkuys42lkw3vtzz2rzsrzcbsrrmkmxe5kpzyuin.py
# Topologically Sorted Source Nodes: [conv_kernel_layer_2], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   conv_kernel_layer_2 => amax, div, exp, sub_2, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_11, [1], True), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_11, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_per_fused__softmax_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + ((r1 + (9*x0)) % 54), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (9*x0)), tmp13, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eb/cebbbrt3i76jmdng55dpzorplbwobnt3rmibkvm2q3vf4rbks3vc.py
# Topologically Sorted Source Nodes: [conv_out_layer_5], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   conv_out_layer_5 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_16,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6291456
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y1 = (yindex // 384) % 512
    y4 = yindex
    y0 = yindex % 384
    tmp0 = (-4) + x3 + y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-1536) + y4 + (384*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x3 + (9*y4)), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gt/cgtoa3k4223jcmzvol5w2jilwh55eouptckbruysuv6432tv35r3.py
# Topologically Sorted Source Nodes: [context_layer_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   context_layer_2 => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_15, %view_28], 2), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 64) % 12
    x0 = xindex % 64
    x2 = (xindex // 768)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (64*x1) + (384*x2)), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 12, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + (64*((-6) + x1)) + (384*x2)), tmp6, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gk/cgk24vs64dzxho3gikt6h3ord3id6qgn2wfnjryj57hrqemfrfcj.py
# Topologically Sorted Source Nodes: [add_3, hidden_states_2], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_3 => add_9
#   hidden_states_2 => add_10, add_11, mul_4, mul_5, rsqrt_1, sub_4, var_mean_1
# Graph fragment:
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_31, %add_3), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_3), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-12), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg22_1), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg23_1), kwargs = {})
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 16384
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
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dy/cdyrvd3fxdatwrvoc7v3qe7bixblkgtqqr5dgfrh3cq7tpiq3gzv.py
# Topologically Sorted Source Nodes: [hidden_states_4], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_4 => add_12, erf, mul_6, mul_7, mul_8
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_33, 0.5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_33, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_7,), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %add_12), kwargs = {})
triton_poi_fused_gelu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
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


# kernel path: /tmp/torchinductor_sahanp/dn/cdnklqwwxxq3c3uuypujqkyu5v72ln7kz354klmglfvdnkpw7csz.py
# Topologically Sorted Source Nodes: [hidden_states_97, hidden_states_98], Original ATen: [aten.gelu, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_97 => add_148, erf_12, mul_100, mul_101, mul_99
#   hidden_states_98 => add_149, add_150, mul_102, mul_103, rsqrt_25, sub_50, var_mean_25
# Graph fragment:
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_433, 0.5), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_433, 0.7071067811865476), kwargs = {})
#   %erf_12 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_100,), kwargs = {})
#   %add_148 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_12, 1), kwargs = {})
#   %mul_101 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_99, %add_148), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_101, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_101, %getitem_51), kwargs = {})
#   %add_149 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-12), kwargs = {})
#   %rsqrt_25 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_149,), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %rsqrt_25), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, %arg286_1), kwargs = {})
#   %add_150 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_103, %arg287_1), kwargs = {})
triton_per_fused_gelu_native_layer_norm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_8', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 16384
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
    tmp34 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 768.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-12
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp37, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/u4/cu4c32hpyo6h7fjdamb36u2rvcjphwdl2ejxjjrbv5mktamc6a6h.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_229, %full_default_3], 1), kwargs = {})
triton_poi_fused_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23442432
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
    tmp5 = tl.load(in_ptr0 + (x1 + (768*x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 30524, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x0 + (30528*x1)), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wt/cwtxphmfktpdamus4oucnndk2ji3ajntc3tgsf3ket5n3dmziqf2.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg288_1, %full_default_4],), kwargs = {})
triton_poi_fused_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/ck/cck4j76ouzmoujcgzukyzzie2ojnql4tm33jjpqgp4cyxbbmpgcc.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   loss => amax_24, exp_24, sub_51, sum_25
# Graph fragment:
#   %amax_24 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_436, [1], True), kwargs = {})
#   %sub_51 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_436, %amax_24), kwargs = {})
#   %exp_24 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_51,), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [1], True), kwargs = {})
triton_red_fused__log_softmax_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 30522
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
        tmp0 = tl.load(in_ptr0 + (r1 + (30528*x0)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (30528*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4i/c4i33eju23tzes3mooczwfhrzj2jzv6o37dd6da3lhyu2ysqvqod.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => full_default_2, ne_1, ne_2, neg, sum_26, sum_27, where_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_437, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_2), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_437, -100), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
triton_red_fused_nll_loss_forward_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr3 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = tl.full([XBLOCK, RBLOCK], 30522, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 30522)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp8 < 30522")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (30528*r1) + (250085376*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/cn/ccn44rbeeiu5ci3t4dniuwzsxka4zzijgwvoj23t6dilsd62mkiq.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type, div_36, full_default_2, ne_1, ne_2, neg, sum_26, sum_27, where_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_437, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_2), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_437, -100), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_26, torch.float32), kwargs = {})
#   %div_36 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_27, %convert_element_type), kwargs = {})
triton_per_fused_nll_loss_forward_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_13', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 512), (512, 1))
    assert_size_stride(arg1_1, (1, 512), (512, 1))
    assert_size_stride(arg2_1, (30522, 768), (768, 1))
    assert_size_stride(arg3_1, (512, 768), (768, 1))
    assert_size_stride(arg4_1, (2, 768), (768, 1))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (384, 768), (768, 1))
    assert_size_stride(arg8_1, (384, ), (1, ))
    assert_size_stride(arg9_1, (384, 768), (768, 1))
    assert_size_stride(arg10_1, (384, ), (1, ))
    assert_size_stride(arg11_1, (384, 768), (768, 1))
    assert_size_stride(arg12_1, (384, ), (1, ))
    assert_size_stride(arg13_1, (384, 1), (1, 1))
    assert_size_stride(arg14_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg15_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg16_1, (54, 384), (384, 1))
    assert_size_stride(arg17_1, (54, ), (1, ))
    assert_size_stride(arg18_1, (384, 768), (768, 1))
    assert_size_stride(arg19_1, (384, ), (1, ))
    assert_size_stride(arg20_1, (768, 768), (768, 1))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (3072, 768), (768, 1))
    assert_size_stride(arg25_1, (3072, ), (1, ))
    assert_size_stride(arg26_1, (768, 3072), (3072, 1))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (384, 768), (768, 1))
    assert_size_stride(arg31_1, (384, ), (1, ))
    assert_size_stride(arg32_1, (384, 768), (768, 1))
    assert_size_stride(arg33_1, (384, ), (1, ))
    assert_size_stride(arg34_1, (384, 768), (768, 1))
    assert_size_stride(arg35_1, (384, ), (1, ))
    assert_size_stride(arg36_1, (384, 1), (1, 1))
    assert_size_stride(arg37_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg38_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg39_1, (54, 384), (384, 1))
    assert_size_stride(arg40_1, (54, ), (1, ))
    assert_size_stride(arg41_1, (384, 768), (768, 1))
    assert_size_stride(arg42_1, (384, ), (1, ))
    assert_size_stride(arg43_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (3072, 768), (768, 1))
    assert_size_stride(arg48_1, (3072, ), (1, ))
    assert_size_stride(arg49_1, (768, 3072), (3072, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (384, 768), (768, 1))
    assert_size_stride(arg54_1, (384, ), (1, ))
    assert_size_stride(arg55_1, (384, 768), (768, 1))
    assert_size_stride(arg56_1, (384, ), (1, ))
    assert_size_stride(arg57_1, (384, 768), (768, 1))
    assert_size_stride(arg58_1, (384, ), (1, ))
    assert_size_stride(arg59_1, (384, 1), (1, 1))
    assert_size_stride(arg60_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg61_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg62_1, (54, 384), (384, 1))
    assert_size_stride(arg63_1, (54, ), (1, ))
    assert_size_stride(arg64_1, (384, 768), (768, 1))
    assert_size_stride(arg65_1, (384, ), (1, ))
    assert_size_stride(arg66_1, (768, 768), (768, 1))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (3072, 768), (768, 1))
    assert_size_stride(arg71_1, (3072, ), (1, ))
    assert_size_stride(arg72_1, (768, 3072), (3072, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (384, 768), (768, 1))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (384, 768), (768, 1))
    assert_size_stride(arg79_1, (384, ), (1, ))
    assert_size_stride(arg80_1, (384, 768), (768, 1))
    assert_size_stride(arg81_1, (384, ), (1, ))
    assert_size_stride(arg82_1, (384, 1), (1, 1))
    assert_size_stride(arg83_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg84_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg85_1, (54, 384), (384, 1))
    assert_size_stride(arg86_1, (54, ), (1, ))
    assert_size_stride(arg87_1, (384, 768), (768, 1))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (768, 768), (768, 1))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (3072, 768), (768, 1))
    assert_size_stride(arg94_1, (3072, ), (1, ))
    assert_size_stride(arg95_1, (768, 3072), (3072, 1))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (384, 768), (768, 1))
    assert_size_stride(arg100_1, (384, ), (1, ))
    assert_size_stride(arg101_1, (384, 768), (768, 1))
    assert_size_stride(arg102_1, (384, ), (1, ))
    assert_size_stride(arg103_1, (384, 768), (768, 1))
    assert_size_stride(arg104_1, (384, ), (1, ))
    assert_size_stride(arg105_1, (384, 1), (1, 1))
    assert_size_stride(arg106_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg107_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg108_1, (54, 384), (384, 1))
    assert_size_stride(arg109_1, (54, ), (1, ))
    assert_size_stride(arg110_1, (384, 768), (768, 1))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (768, 768), (768, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (3072, 768), (768, 1))
    assert_size_stride(arg117_1, (3072, ), (1, ))
    assert_size_stride(arg118_1, (768, 3072), (3072, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (384, 768), (768, 1))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (384, 768), (768, 1))
    assert_size_stride(arg125_1, (384, ), (1, ))
    assert_size_stride(arg126_1, (384, 768), (768, 1))
    assert_size_stride(arg127_1, (384, ), (1, ))
    assert_size_stride(arg128_1, (384, 1), (1, 1))
    assert_size_stride(arg129_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg130_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg131_1, (54, 384), (384, 1))
    assert_size_stride(arg132_1, (54, ), (1, ))
    assert_size_stride(arg133_1, (384, 768), (768, 1))
    assert_size_stride(arg134_1, (384, ), (1, ))
    assert_size_stride(arg135_1, (768, 768), (768, 1))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (3072, 768), (768, 1))
    assert_size_stride(arg140_1, (3072, ), (1, ))
    assert_size_stride(arg141_1, (768, 3072), (3072, 1))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (384, 768), (768, 1))
    assert_size_stride(arg146_1, (384, ), (1, ))
    assert_size_stride(arg147_1, (384, 768), (768, 1))
    assert_size_stride(arg148_1, (384, ), (1, ))
    assert_size_stride(arg149_1, (384, 768), (768, 1))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (384, 1), (1, 1))
    assert_size_stride(arg152_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg153_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg154_1, (54, 384), (384, 1))
    assert_size_stride(arg155_1, (54, ), (1, ))
    assert_size_stride(arg156_1, (384, 768), (768, 1))
    assert_size_stride(arg157_1, (384, ), (1, ))
    assert_size_stride(arg158_1, (768, 768), (768, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (3072, 768), (768, 1))
    assert_size_stride(arg163_1, (3072, ), (1, ))
    assert_size_stride(arg164_1, (768, 3072), (3072, 1))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (384, 768), (768, 1))
    assert_size_stride(arg169_1, (384, ), (1, ))
    assert_size_stride(arg170_1, (384, 768), (768, 1))
    assert_size_stride(arg171_1, (384, ), (1, ))
    assert_size_stride(arg172_1, (384, 768), (768, 1))
    assert_size_stride(arg173_1, (384, ), (1, ))
    assert_size_stride(arg174_1, (384, 1), (1, 1))
    assert_size_stride(arg175_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg176_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg177_1, (54, 384), (384, 1))
    assert_size_stride(arg178_1, (54, ), (1, ))
    assert_size_stride(arg179_1, (384, 768), (768, 1))
    assert_size_stride(arg180_1, (384, ), (1, ))
    assert_size_stride(arg181_1, (768, 768), (768, 1))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (3072, 768), (768, 1))
    assert_size_stride(arg186_1, (3072, ), (1, ))
    assert_size_stride(arg187_1, (768, 3072), (3072, 1))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (384, 768), (768, 1))
    assert_size_stride(arg192_1, (384, ), (1, ))
    assert_size_stride(arg193_1, (384, 768), (768, 1))
    assert_size_stride(arg194_1, (384, ), (1, ))
    assert_size_stride(arg195_1, (384, 768), (768, 1))
    assert_size_stride(arg196_1, (384, ), (1, ))
    assert_size_stride(arg197_1, (384, 1), (1, 1))
    assert_size_stride(arg198_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg199_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg200_1, (54, 384), (384, 1))
    assert_size_stride(arg201_1, (54, ), (1, ))
    assert_size_stride(arg202_1, (384, 768), (768, 1))
    assert_size_stride(arg203_1, (384, ), (1, ))
    assert_size_stride(arg204_1, (768, 768), (768, 1))
    assert_size_stride(arg205_1, (768, ), (1, ))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (3072, 768), (768, 1))
    assert_size_stride(arg209_1, (3072, ), (1, ))
    assert_size_stride(arg210_1, (768, 3072), (3072, 1))
    assert_size_stride(arg211_1, (768, ), (1, ))
    assert_size_stride(arg212_1, (768, ), (1, ))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (384, 768), (768, 1))
    assert_size_stride(arg215_1, (384, ), (1, ))
    assert_size_stride(arg216_1, (384, 768), (768, 1))
    assert_size_stride(arg217_1, (384, ), (1, ))
    assert_size_stride(arg218_1, (384, 768), (768, 1))
    assert_size_stride(arg219_1, (384, ), (1, ))
    assert_size_stride(arg220_1, (384, 1), (1, 1))
    assert_size_stride(arg221_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg222_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg223_1, (54, 384), (384, 1))
    assert_size_stride(arg224_1, (54, ), (1, ))
    assert_size_stride(arg225_1, (384, 768), (768, 1))
    assert_size_stride(arg226_1, (384, ), (1, ))
    assert_size_stride(arg227_1, (768, 768), (768, 1))
    assert_size_stride(arg228_1, (768, ), (1, ))
    assert_size_stride(arg229_1, (768, ), (1, ))
    assert_size_stride(arg230_1, (768, ), (1, ))
    assert_size_stride(arg231_1, (3072, 768), (768, 1))
    assert_size_stride(arg232_1, (3072, ), (1, ))
    assert_size_stride(arg233_1, (768, 3072), (3072, 1))
    assert_size_stride(arg234_1, (768, ), (1, ))
    assert_size_stride(arg235_1, (768, ), (1, ))
    assert_size_stride(arg236_1, (768, ), (1, ))
    assert_size_stride(arg237_1, (384, 768), (768, 1))
    assert_size_stride(arg238_1, (384, ), (1, ))
    assert_size_stride(arg239_1, (384, 768), (768, 1))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (384, 768), (768, 1))
    assert_size_stride(arg242_1, (384, ), (1, ))
    assert_size_stride(arg243_1, (384, 1), (1, 1))
    assert_size_stride(arg244_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg245_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg246_1, (54, 384), (384, 1))
    assert_size_stride(arg247_1, (54, ), (1, ))
    assert_size_stride(arg248_1, (384, 768), (768, 1))
    assert_size_stride(arg249_1, (384, ), (1, ))
    assert_size_stride(arg250_1, (768, 768), (768, 1))
    assert_size_stride(arg251_1, (768, ), (1, ))
    assert_size_stride(arg252_1, (768, ), (1, ))
    assert_size_stride(arg253_1, (768, ), (1, ))
    assert_size_stride(arg254_1, (3072, 768), (768, 1))
    assert_size_stride(arg255_1, (3072, ), (1, ))
    assert_size_stride(arg256_1, (768, 3072), (3072, 1))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (768, ), (1, ))
    assert_size_stride(arg259_1, (768, ), (1, ))
    assert_size_stride(arg260_1, (384, 768), (768, 1))
    assert_size_stride(arg261_1, (384, ), (1, ))
    assert_size_stride(arg262_1, (384, 768), (768, 1))
    assert_size_stride(arg263_1, (384, ), (1, ))
    assert_size_stride(arg264_1, (384, 768), (768, 1))
    assert_size_stride(arg265_1, (384, ), (1, ))
    assert_size_stride(arg266_1, (384, 1), (1, 1))
    assert_size_stride(arg267_1, (768, 1, 9), (9, 9, 1))
    assert_size_stride(arg268_1, (384, 768, 1), (768, 1, 1))
    assert_size_stride(arg269_1, (54, 384), (384, 1))
    assert_size_stride(arg270_1, (54, ), (1, ))
    assert_size_stride(arg271_1, (384, 768), (768, 1))
    assert_size_stride(arg272_1, (384, ), (1, ))
    assert_size_stride(arg273_1, (768, 768), (768, 1))
    assert_size_stride(arg274_1, (768, ), (1, ))
    assert_size_stride(arg275_1, (768, ), (1, ))
    assert_size_stride(arg276_1, (768, ), (1, ))
    assert_size_stride(arg277_1, (3072, 768), (768, 1))
    assert_size_stride(arg278_1, (3072, ), (1, ))
    assert_size_stride(arg279_1, (768, 3072), (3072, 1))
    assert_size_stride(arg280_1, (768, ), (1, ))
    assert_size_stride(arg281_1, (768, ), (1, ))
    assert_size_stride(arg282_1, (768, ), (1, ))
    assert_size_stride(arg283_1, (1, 512), (512, 1))
    assert_size_stride(arg284_1, (768, 768), (768, 1))
    assert_size_stride(arg285_1, (768, ), (1, ))
    assert_size_stride(arg286_1, (768, ), (1, ))
    assert_size_stride(arg287_1, (768, ), (1, ))
    assert_size_stride(arg288_1, (30522, ), (1, ))
    assert_size_stride(arg289_1, (32, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 512, 768), (393216, 768, 1), torch.float32)
        buf4 = empty_strided_cuda((32, 512, 768), (393216, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, position_embeddings, add, token_type_embeddings, embeddings, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg0_1, arg2_1, arg283_1, arg3_1, arg1_1, arg4_1, arg5_1, arg6_1, buf0, buf4, 16384, 768, grid=grid(16384), stream=stream0)
        del arg0_1
        del arg1_1
        del arg283_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        buf5 = empty_strided_cuda((16384, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf4, (16384, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf5)
        del arg7_1
        del arg8_1
        buf6 = empty_strided_cuda((16384, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mixed_key_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf4, (16384, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf6)
        del arg10_1
        del arg9_1
        buf7 = empty_strided_cuda((16384, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mixed_value_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf4, (16384, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf7)
        del arg11_1
        del arg12_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf5, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf6, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf7, (32, 6, 512, 64), (196608, 64, 384, 1), 0), None, False, scale=0.125)
        del buf6
        buf9 = buf8[0]
        del buf8
        buf13 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (16384, 768), (768, 1), 0), reinterpret_tensor(arg18_1, (768, 384), (1, 768), 0), out=buf13)
        del arg18_1
        buf14 = reinterpret_tensor(buf0, (32, 768, 512), (393216, 512, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf4, buf14, 24576, 512, grid=grid(24576, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg14_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf15, (32, 768, 512), (393216, 512, 1))
        del arg14_1
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg15_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf16, (32, 384, 512), (196608, 512, 1))
        del arg15_1
        buf17 = reinterpret_tensor(buf5, (32, 512, 384), (196608, 384, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [conv_attn_layer, conv_kernel_layer], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf17, buf16, arg13_1, 16384, 384, grid=grid(16384, 384), stream=stream0)
        del arg13_1
        del buf16
        buf18 = empty_strided_cuda((16384, 54), (54, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv_kernel_layer], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (16384, 384), (384, 1), 0), reinterpret_tensor(arg16_1, (384, 54), (1, 384), 0), out=buf18)
        del arg16_1
        buf22 = empty_strided_cuda((98304, 9, 1), (9, 1, 884736), torch.float32)
        # Topologically Sorted Source Nodes: [conv_kernel_layer_2], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf18, arg17_1, buf22, 98304, 9, grid=grid(98304), stream=stream0)
        del arg17_1
        buf21 = empty_strided_cuda((32, 512, 384, 9), (1769472, 3456, 9, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv_out_layer_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf13, arg19_1, buf21, 6291456, 9, grid=grid(6291456, 9), stream=stream0)
        del arg19_1
        buf23 = reinterpret_tensor(buf13, (98304, 64, 1), (64, 1, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_2, conv_out_layer_6], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (98304, 64, 9), (576, 9, 1), 0), buf22, out=buf23)
        buf24 = reinterpret_tensor(buf15, (32, 512, 12, 64), (393216, 768, 64, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [context_layer_2], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf9, buf23, buf24, 12582912, grid=grid(12582912), stream=stream0)
        buf25 = reinterpret_tensor(buf14, (16384, 768), (768, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf24, (16384, 768), (768, 1), 0), reinterpret_tensor(arg20_1, (768, 768), (1, 768), 0), out=buf25)
        del arg20_1
        buf29 = reinterpret_tensor(buf24, (32, 512, 768), (393216, 768, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [add_3, hidden_states_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf25, arg21_1, buf4, arg22_1, arg23_1, buf29, 16384, 768, grid=grid(16384), stream=stream0)
        del arg21_1
        del arg22_1
        del arg23_1
        buf30 = empty_strided_cuda((16384, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (16384, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 3072), (1, 768), 0), out=buf30)
        del arg24_1
        buf31 = reinterpret_tensor(buf30, (32, 512, 3072), (1572864, 3072, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf31, arg25_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg25_1
        buf32 = reinterpret_tensor(buf4, (16384, 768), (768, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg26_1, (3072, 768), (1, 3072), 0), out=buf32)
        del arg26_1
        buf36 = reinterpret_tensor(buf25, (32, 512, 768), (393216, 768, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [add_4, hidden_states_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf32, arg27_1, buf29, arg28_1, arg29_1, buf36, 16384, 768, grid=grid(16384), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del buf29
        buf37 = reinterpret_tensor(buf9, (16384, 384), (384, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg31_1, reinterpret_tensor(buf36, (16384, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf37)
        del arg30_1
        del arg31_1
        buf38 = reinterpret_tensor(buf23, (16384, 384), (384, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg33_1, reinterpret_tensor(buf36, (16384, 768), (768, 1), 0), reinterpret_tensor(arg32_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf38)
        del arg32_1
        del arg33_1
        buf39 = reinterpret_tensor(buf17, (16384, 384), (384, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg35_1, reinterpret_tensor(buf36, (16384, 768), (768, 1), 0), reinterpret_tensor(arg34_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf39)
        del arg34_1
        del arg35_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf40 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf37, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf38, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf39, (32, 6, 512, 64), (196608, 64, 384, 1), 0), None, False, scale=0.125)
        del buf38
        buf41 = buf40[0]
        del buf40
        buf45 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf36, (16384, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 384), (1, 768), 0), out=buf45)
        del arg41_1
        buf46 = reinterpret_tensor(buf32, (32, 768, 512), (393216, 512, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf36, buf46, 24576, 512, grid=grid(24576, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg37_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf47, (32, 768, 512), (393216, 512, 1))
        del arg37_1
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg38_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf48, (32, 384, 512), (196608, 512, 1))
        del arg38_1
        buf49 = reinterpret_tensor(buf37, (32, 512, 384), (196608, 384, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [conv_attn_layer_1, conv_kernel_layer_3], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf49, buf48, arg36_1, 16384, 384, grid=grid(16384, 384), stream=stream0)
        del arg36_1
        del buf48
        buf50 = reinterpret_tensor(buf22, (16384, 54), (54, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (16384, 384), (384, 1), 0), reinterpret_tensor(arg39_1, (384, 54), (1, 384), 0), out=buf50)
        del arg39_1
        buf54 = reinterpret_tensor(buf18, (98304, 9, 1), (9, 1, 884736), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_5], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf50, arg40_1, buf54, 98304, 9, grid=grid(98304), stream=stream0)
        del arg40_1
        buf53 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [conv_out_layer_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf45, arg42_1, buf53, 6291456, 9, grid=grid(6291456, 9), stream=stream0)
        del arg42_1
        buf55 = reinterpret_tensor(buf45, (98304, 64, 1), (64, 1, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_5, conv_out_layer_14], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (98304, 64, 9), (576, 9, 1), 0), buf54, out=buf55)
        buf56 = reinterpret_tensor(buf47, (32, 512, 12, 64), (393216, 768, 64, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [context_layer_6], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf41, buf55, buf56, 12582912, grid=grid(12582912), stream=stream0)
        buf57 = reinterpret_tensor(buf46, (16384, 768), (768, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (16384, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), out=buf57)
        del arg43_1
        buf61 = reinterpret_tensor(buf56, (32, 512, 768), (393216, 768, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [add_6, hidden_states_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf57, arg44_1, buf36, arg45_1, arg46_1, buf61, 16384, 768, grid=grid(16384), stream=stream0)
        del arg44_1
        del arg45_1
        del arg46_1
        buf62 = reinterpret_tensor(buf31, (16384, 3072), (3072, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (16384, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 3072), (1, 768), 0), out=buf62)
        del arg47_1
        buf63 = reinterpret_tensor(buf62, (32, 512, 3072), (1572864, 3072, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_12], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf63, arg48_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg48_1
        buf64 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg49_1, (3072, 768), (1, 3072), 0), out=buf64)
        del arg49_1
        buf68 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [add_7, hidden_states_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf64, arg50_1, buf61, arg51_1, arg52_1, buf68, 16384, 768, grid=grid(16384), stream=stream0)
        del arg50_1
        del arg51_1
        del arg52_1
        del buf61
        buf69 = reinterpret_tensor(buf55, (16384, 384), (384, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg54_1, reinterpret_tensor(buf68, (16384, 768), (768, 1), 0), reinterpret_tensor(arg53_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf69)
        del arg53_1
        del arg54_1
        buf70 = reinterpret_tensor(buf41, (16384, 384), (384, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg56_1, reinterpret_tensor(buf68, (16384, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf70)
        del arg55_1
        del arg56_1
        buf71 = reinterpret_tensor(buf49, (16384, 384), (384, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg58_1, reinterpret_tensor(buf68, (16384, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf71)
        del arg57_1
        del arg58_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf72 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf69, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf70, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf71, (32, 6, 512, 64), (196608, 64, 384, 1), 0), None, False, scale=0.125)
        del buf70
        buf73 = buf72[0]
        del buf72
        buf77 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (16384, 768), (768, 1), 0), reinterpret_tensor(arg64_1, (768, 384), (1, 768), 0), out=buf77)
        del arg64_1
        buf78 = reinterpret_tensor(buf64, (32, 768, 512), (393216, 512, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf68, buf78, 24576, 512, grid=grid(24576, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, arg60_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf79, (32, 768, 512), (393216, 512, 1))
        del arg60_1
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg61_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf80, (32, 384, 512), (196608, 512, 1))
        del arg61_1
        buf81 = reinterpret_tensor(buf69, (32, 512, 384), (196608, 384, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [conv_attn_layer_2, conv_kernel_layer_6], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf81, buf80, arg59_1, 16384, 384, grid=grid(16384, 384), stream=stream0)
        del arg59_1
        del buf80
        buf82 = reinterpret_tensor(buf54, (16384, 54), (54, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (16384, 384), (384, 1), 0), reinterpret_tensor(arg62_1, (384, 54), (1, 384), 0), out=buf82)
        del arg62_1
        buf86 = reinterpret_tensor(buf50, (98304, 9, 1), (9, 1, 884736), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_8], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf82, arg63_1, buf86, 98304, 9, grid=grid(98304), stream=stream0)
        del arg63_1
        buf85 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [conv_out_layer_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf77, arg65_1, buf85, 6291456, 9, grid=grid(6291456, 9), stream=stream0)
        del arg65_1
        buf87 = reinterpret_tensor(buf77, (98304, 64, 1), (64, 1, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_8, conv_out_layer_22], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf85, (98304, 64, 9), (576, 9, 1), 0), buf86, out=buf87)
        buf88 = reinterpret_tensor(buf79, (32, 512, 12, 64), (393216, 768, 64, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [context_layer_10], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf73, buf87, buf88, 12582912, grid=grid(12582912), stream=stream0)
        buf89 = reinterpret_tensor(buf78, (16384, 768), (768, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf88, (16384, 768), (768, 1), 0), reinterpret_tensor(arg66_1, (768, 768), (1, 768), 0), out=buf89)
        del arg66_1
        buf93 = reinterpret_tensor(buf88, (32, 512, 768), (393216, 768, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [add_9, hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf89, arg67_1, buf68, arg68_1, arg69_1, buf93, 16384, 768, grid=grid(16384), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        buf94 = reinterpret_tensor(buf63, (16384, 3072), (3072, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (16384, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 3072), (1, 768), 0), out=buf94)
        del arg70_1
        buf95 = reinterpret_tensor(buf94, (32, 512, 3072), (1572864, 3072, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf95, arg71_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg71_1
        buf96 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg72_1, (3072, 768), (1, 3072), 0), out=buf96)
        del arg72_1
        buf100 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [add_10, hidden_states_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf96, arg73_1, buf93, arg74_1, arg75_1, buf100, 16384, 768, grid=grid(16384), stream=stream0)
        del arg73_1
        del arg74_1
        del arg75_1
        del buf93
        buf101 = reinterpret_tensor(buf87, (16384, 384), (384, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg77_1, reinterpret_tensor(buf100, (16384, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf101)
        del arg76_1
        del arg77_1
        buf102 = reinterpret_tensor(buf73, (16384, 384), (384, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg79_1, reinterpret_tensor(buf100, (16384, 768), (768, 1), 0), reinterpret_tensor(arg78_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf102)
        del arg78_1
        del arg79_1
        buf103 = reinterpret_tensor(buf81, (16384, 384), (384, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg81_1, reinterpret_tensor(buf100, (16384, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf103)
        del arg80_1
        del arg81_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf104 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf101, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf102, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf103, (32, 6, 512, 64), (196608, 64, 384, 1), 0), None, False, scale=0.125)
        del buf102
        buf105 = buf104[0]
        del buf104
        buf109 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (16384, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 384), (1, 768), 0), out=buf109)
        del arg87_1
        buf110 = reinterpret_tensor(buf96, (32, 768, 512), (393216, 512, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf100, buf110, 24576, 512, grid=grid(24576, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, arg83_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf111, (32, 768, 512), (393216, 512, 1))
        del arg83_1
        # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg84_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf112, (32, 384, 512), (196608, 512, 1))
        del arg84_1
        buf113 = reinterpret_tensor(buf101, (32, 512, 384), (196608, 384, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [conv_attn_layer_3, conv_kernel_layer_9], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf113, buf112, arg82_1, 16384, 384, grid=grid(16384, 384), stream=stream0)
        del arg82_1
        del buf112
        buf114 = reinterpret_tensor(buf86, (16384, 54), (54, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (16384, 384), (384, 1), 0), reinterpret_tensor(arg85_1, (384, 54), (1, 384), 0), out=buf114)
        del arg85_1
        buf118 = reinterpret_tensor(buf82, (98304, 9, 1), (9, 1, 884736), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf114, arg86_1, buf118, 98304, 9, grid=grid(98304), stream=stream0)
        del arg86_1
        buf117 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [conv_out_layer_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf109, arg88_1, buf117, 6291456, 9, grid=grid(6291456, 9), stream=stream0)
        del arg88_1
        buf119 = reinterpret_tensor(buf109, (98304, 64, 1), (64, 1, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_11, conv_out_layer_30], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf117, (98304, 64, 9), (576, 9, 1), 0), buf118, out=buf119)
        buf120 = reinterpret_tensor(buf111, (32, 512, 12, 64), (393216, 768, 64, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [context_layer_14], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf105, buf119, buf120, 12582912, grid=grid(12582912), stream=stream0)
        buf121 = reinterpret_tensor(buf110, (16384, 768), (768, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (16384, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), out=buf121)
        del arg89_1
        buf125 = reinterpret_tensor(buf120, (32, 512, 768), (393216, 768, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [add_12, hidden_states_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf121, arg90_1, buf100, arg91_1, arg92_1, buf125, 16384, 768, grid=grid(16384), stream=stream0)
        del arg90_1
        del arg91_1
        del arg92_1
        buf126 = reinterpret_tensor(buf95, (16384, 3072), (3072, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf125, (16384, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 3072), (1, 768), 0), out=buf126)
        del arg93_1
        buf127 = reinterpret_tensor(buf126, (32, 512, 3072), (1572864, 3072, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf127, arg94_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg94_1
        buf128 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg95_1, (3072, 768), (1, 3072), 0), out=buf128)
        del arg95_1
        buf132 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [add_13, hidden_states_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf128, arg96_1, buf125, arg97_1, arg98_1, buf132, 16384, 768, grid=grid(16384), stream=stream0)
        del arg96_1
        del arg97_1
        del arg98_1
        del buf125
        buf133 = reinterpret_tensor(buf119, (16384, 384), (384, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg100_1, reinterpret_tensor(buf132, (16384, 768), (768, 1), 0), reinterpret_tensor(arg99_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf133)
        del arg100_1
        del arg99_1
        buf134 = reinterpret_tensor(buf105, (16384, 384), (384, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg102_1, reinterpret_tensor(buf132, (16384, 768), (768, 1), 0), reinterpret_tensor(arg101_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf134)
        del arg101_1
        del arg102_1
        buf135 = reinterpret_tensor(buf113, (16384, 384), (384, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg104_1, reinterpret_tensor(buf132, (16384, 768), (768, 1), 0), reinterpret_tensor(arg103_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf135)
        del arg103_1
        del arg104_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf136 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf133, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf134, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf135, (32, 6, 512, 64), (196608, 64, 384, 1), 0), None, False, scale=0.125)
        del buf134
        buf137 = buf136[0]
        del buf136
        buf141 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (16384, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 384), (1, 768), 0), out=buf141)
        del arg110_1
        buf142 = reinterpret_tensor(buf128, (32, 768, 512), (393216, 512, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf132, buf142, 24576, 512, grid=grid(24576, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, arg106_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf143, (32, 768, 512), (393216, 512, 1))
        del arg106_1
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, arg107_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf144, (32, 384, 512), (196608, 512, 1))
        del arg107_1
        buf145 = reinterpret_tensor(buf133, (32, 512, 384), (196608, 384, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [conv_attn_layer_4, conv_kernel_layer_12], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf145, buf144, arg105_1, 16384, 384, grid=grid(16384, 384), stream=stream0)
        del arg105_1
        del buf144
        buf146 = reinterpret_tensor(buf118, (16384, 54), (54, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (16384, 384), (384, 1), 0), reinterpret_tensor(arg108_1, (384, 54), (1, 384), 0), out=buf146)
        del arg108_1
        buf150 = reinterpret_tensor(buf114, (98304, 9, 1), (9, 1, 884736), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_14], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf146, arg109_1, buf150, 98304, 9, grid=grid(98304), stream=stream0)
        del arg109_1
        buf149 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [conv_out_layer_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf141, arg111_1, buf149, 6291456, 9, grid=grid(6291456, 9), stream=stream0)
        del arg111_1
        buf151 = reinterpret_tensor(buf141, (98304, 64, 1), (64, 1, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_14, conv_out_layer_38], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf149, (98304, 64, 9), (576, 9, 1), 0), buf150, out=buf151)
        buf152 = reinterpret_tensor(buf143, (32, 512, 12, 64), (393216, 768, 64, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [context_layer_18], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf137, buf151, buf152, 12582912, grid=grid(12582912), stream=stream0)
        buf153 = reinterpret_tensor(buf142, (16384, 768), (768, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf152, (16384, 768), (768, 1), 0), reinterpret_tensor(arg112_1, (768, 768), (1, 768), 0), out=buf153)
        del arg112_1
        buf157 = reinterpret_tensor(buf152, (32, 512, 768), (393216, 768, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [add_15, hidden_states_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf153, arg113_1, buf132, arg114_1, arg115_1, buf157, 16384, 768, grid=grid(16384), stream=stream0)
        del arg113_1
        del arg114_1
        del arg115_1
        buf158 = reinterpret_tensor(buf127, (16384, 3072), (3072, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (16384, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 3072), (1, 768), 0), out=buf158)
        del arg116_1
        buf159 = reinterpret_tensor(buf158, (32, 512, 3072), (1572864, 3072, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_36], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf159, arg117_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg117_1
        buf160 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg118_1, (3072, 768), (1, 3072), 0), out=buf160)
        del arg118_1
        buf164 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [add_16, hidden_states_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf160, arg119_1, buf157, arg120_1, arg121_1, buf164, 16384, 768, grid=grid(16384), stream=stream0)
        del arg119_1
        del arg120_1
        del arg121_1
        del buf157
        buf165 = reinterpret_tensor(buf151, (16384, 384), (384, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg123_1, reinterpret_tensor(buf164, (16384, 768), (768, 1), 0), reinterpret_tensor(arg122_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf165)
        del arg122_1
        del arg123_1
        buf166 = reinterpret_tensor(buf137, (16384, 384), (384, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg125_1, reinterpret_tensor(buf164, (16384, 768), (768, 1), 0), reinterpret_tensor(arg124_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf166)
        del arg124_1
        del arg125_1
        buf167 = reinterpret_tensor(buf145, (16384, 384), (384, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg127_1, reinterpret_tensor(buf164, (16384, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf167)
        del arg126_1
        del arg127_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf168 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf165, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf166, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf167, (32, 6, 512, 64), (196608, 64, 384, 1), 0), None, False, scale=0.125)
        del buf166
        buf169 = buf168[0]
        del buf168
        buf173 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (16384, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 384), (1, 768), 0), out=buf173)
        del arg133_1
        buf174 = reinterpret_tensor(buf160, (32, 768, 512), (393216, 512, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf164, buf174, 24576, 512, grid=grid(24576, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, arg129_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf175, (32, 768, 512), (393216, 512, 1))
        del arg129_1
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg130_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf176, (32, 384, 512), (196608, 512, 1))
        del arg130_1
        buf177 = reinterpret_tensor(buf165, (32, 512, 384), (196608, 384, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [conv_attn_layer_5, conv_kernel_layer_15], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf177, buf176, arg128_1, 16384, 384, grid=grid(16384, 384), stream=stream0)
        del arg128_1
        del buf176
        buf178 = reinterpret_tensor(buf150, (16384, 54), (54, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (16384, 384), (384, 1), 0), reinterpret_tensor(arg131_1, (384, 54), (1, 384), 0), out=buf178)
        del arg131_1
        buf182 = reinterpret_tensor(buf146, (98304, 9, 1), (9, 1, 884736), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_17], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf178, arg132_1, buf182, 98304, 9, grid=grid(98304), stream=stream0)
        del arg132_1
        buf181 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [conv_out_layer_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf173, arg134_1, buf181, 6291456, 9, grid=grid(6291456, 9), stream=stream0)
        del arg134_1
        buf183 = reinterpret_tensor(buf173, (98304, 64, 1), (64, 1, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_17, conv_out_layer_46], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf181, (98304, 64, 9), (576, 9, 1), 0), buf182, out=buf183)
        buf184 = reinterpret_tensor(buf175, (32, 512, 12, 64), (393216, 768, 64, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [context_layer_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf169, buf183, buf184, 12582912, grid=grid(12582912), stream=stream0)
        buf185 = reinterpret_tensor(buf174, (16384, 768), (768, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf184, (16384, 768), (768, 1), 0), reinterpret_tensor(arg135_1, (768, 768), (1, 768), 0), out=buf185)
        del arg135_1
        buf189 = reinterpret_tensor(buf184, (32, 512, 768), (393216, 768, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [add_18, hidden_states_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf185, arg136_1, buf164, arg137_1, arg138_1, buf189, 16384, 768, grid=grid(16384), stream=stream0)
        del arg136_1
        del arg137_1
        del arg138_1
        buf190 = reinterpret_tensor(buf159, (16384, 3072), (3072, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (16384, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 3072), (1, 768), 0), out=buf190)
        del arg139_1
        buf191 = reinterpret_tensor(buf190, (32, 512, 3072), (1572864, 3072, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_44], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf191, arg140_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg140_1
        buf192 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg141_1, (3072, 768), (1, 3072), 0), out=buf192)
        del arg141_1
        buf196 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [add_19, hidden_states_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf192, arg142_1, buf189, arg143_1, arg144_1, buf196, 16384, 768, grid=grid(16384), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del buf189
        buf197 = reinterpret_tensor(buf183, (16384, 384), (384, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg146_1, reinterpret_tensor(buf196, (16384, 768), (768, 1), 0), reinterpret_tensor(arg145_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf197)
        del arg145_1
        del arg146_1
        buf198 = reinterpret_tensor(buf169, (16384, 384), (384, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg148_1, reinterpret_tensor(buf196, (16384, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf198)
        del arg147_1
        del arg148_1
        buf199 = reinterpret_tensor(buf177, (16384, 384), (384, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg150_1, reinterpret_tensor(buf196, (16384, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf199)
        del arg149_1
        del arg150_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf200 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf197, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf198, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf199, (32, 6, 512, 64), (196608, 64, 384, 1), 0), None, False, scale=0.125)
        del buf198
        buf201 = buf200[0]
        del buf200
        buf205 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (16384, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 384), (1, 768), 0), out=buf205)
        del arg156_1
        buf206 = reinterpret_tensor(buf192, (32, 768, 512), (393216, 512, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf196, buf206, 24576, 512, grid=grid(24576, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, arg152_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf207, (32, 768, 512), (393216, 512, 1))
        del arg152_1
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, arg153_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf208, (32, 384, 512), (196608, 512, 1))
        del arg153_1
        buf209 = reinterpret_tensor(buf197, (32, 512, 384), (196608, 384, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [conv_attn_layer_6, conv_kernel_layer_18], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf209, buf208, arg151_1, 16384, 384, grid=grid(16384, 384), stream=stream0)
        del arg151_1
        del buf208
        buf210 = reinterpret_tensor(buf182, (16384, 54), (54, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (16384, 384), (384, 1), 0), reinterpret_tensor(arg154_1, (384, 54), (1, 384), 0), out=buf210)
        del arg154_1
        buf214 = reinterpret_tensor(buf178, (98304, 9, 1), (9, 1, 884736), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_20], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf210, arg155_1, buf214, 98304, 9, grid=grid(98304), stream=stream0)
        del arg155_1
        buf213 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [conv_out_layer_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf205, arg157_1, buf213, 6291456, 9, grid=grid(6291456, 9), stream=stream0)
        del arg157_1
        buf215 = reinterpret_tensor(buf205, (98304, 64, 1), (64, 1, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_20, conv_out_layer_54], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (98304, 64, 9), (576, 9, 1), 0), buf214, out=buf215)
        buf216 = reinterpret_tensor(buf207, (32, 512, 12, 64), (393216, 768, 64, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [context_layer_26], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf201, buf215, buf216, 12582912, grid=grid(12582912), stream=stream0)
        buf217 = reinterpret_tensor(buf206, (16384, 768), (768, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf216, (16384, 768), (768, 1), 0), reinterpret_tensor(arg158_1, (768, 768), (1, 768), 0), out=buf217)
        del arg158_1
        buf221 = reinterpret_tensor(buf216, (32, 512, 768), (393216, 768, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [add_21, hidden_states_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf217, arg159_1, buf196, arg160_1, arg161_1, buf221, 16384, 768, grid=grid(16384), stream=stream0)
        del arg159_1
        del arg160_1
        del arg161_1
        buf222 = reinterpret_tensor(buf191, (16384, 3072), (3072, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (16384, 768), (768, 1), 0), reinterpret_tensor(arg162_1, (768, 3072), (1, 768), 0), out=buf222)
        del arg162_1
        buf223 = reinterpret_tensor(buf222, (32, 512, 3072), (1572864, 3072, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_52], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf223, arg163_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg163_1
        buf224 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg164_1, (3072, 768), (1, 3072), 0), out=buf224)
        del arg164_1
        buf228 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [add_22, hidden_states_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf224, arg165_1, buf221, arg166_1, arg167_1, buf228, 16384, 768, grid=grid(16384), stream=stream0)
        del arg165_1
        del arg166_1
        del arg167_1
        del buf221
        buf229 = reinterpret_tensor(buf215, (16384, 384), (384, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg169_1, reinterpret_tensor(buf228, (16384, 768), (768, 1), 0), reinterpret_tensor(arg168_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf229)
        del arg168_1
        del arg169_1
        buf230 = reinterpret_tensor(buf201, (16384, 384), (384, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg171_1, reinterpret_tensor(buf228, (16384, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf230)
        del arg170_1
        del arg171_1
        buf231 = reinterpret_tensor(buf209, (16384, 384), (384, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg173_1, reinterpret_tensor(buf228, (16384, 768), (768, 1), 0), reinterpret_tensor(arg172_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf231)
        del arg172_1
        del arg173_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf232 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf229, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf230, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf231, (32, 6, 512, 64), (196608, 64, 384, 1), 0), None, False, scale=0.125)
        del buf230
        buf233 = buf232[0]
        del buf232
        buf237 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (16384, 768), (768, 1), 0), reinterpret_tensor(arg179_1, (768, 384), (1, 768), 0), out=buf237)
        del arg179_1
        buf238 = reinterpret_tensor(buf224, (32, 768, 512), (393216, 512, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf228, buf238, 24576, 512, grid=grid(24576, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, arg175_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf239, (32, 768, 512), (393216, 512, 1))
        del arg175_1
        # Topologically Sorted Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, arg176_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf240, (32, 384, 512), (196608, 512, 1))
        del arg176_1
        buf241 = reinterpret_tensor(buf229, (32, 512, 384), (196608, 384, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [conv_attn_layer_7, conv_kernel_layer_21], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf241, buf240, arg174_1, 16384, 384, grid=grid(16384, 384), stream=stream0)
        del arg174_1
        del buf240
        buf242 = reinterpret_tensor(buf214, (16384, 54), (54, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf241, (16384, 384), (384, 1), 0), reinterpret_tensor(arg177_1, (384, 54), (1, 384), 0), out=buf242)
        del arg177_1
        buf246 = reinterpret_tensor(buf210, (98304, 9, 1), (9, 1, 884736), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf242, arg178_1, buf246, 98304, 9, grid=grid(98304), stream=stream0)
        del arg178_1
        buf245 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [conv_out_layer_61], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf237, arg180_1, buf245, 6291456, 9, grid=grid(6291456, 9), stream=stream0)
        del arg180_1
        buf247 = reinterpret_tensor(buf237, (98304, 64, 1), (64, 1, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_23, conv_out_layer_62], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf245, (98304, 64, 9), (576, 9, 1), 0), buf246, out=buf247)
        buf248 = reinterpret_tensor(buf239, (32, 512, 12, 64), (393216, 768, 64, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [context_layer_30], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf233, buf247, buf248, 12582912, grid=grid(12582912), stream=stream0)
        buf249 = reinterpret_tensor(buf238, (16384, 768), (768, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (16384, 768), (768, 1), 0), reinterpret_tensor(arg181_1, (768, 768), (1, 768), 0), out=buf249)
        del arg181_1
        buf253 = reinterpret_tensor(buf248, (32, 512, 768), (393216, 768, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [add_24, hidden_states_58], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf249, arg182_1, buf228, arg183_1, arg184_1, buf253, 16384, 768, grid=grid(16384), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        buf254 = reinterpret_tensor(buf223, (16384, 3072), (3072, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (16384, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 3072), (1, 768), 0), out=buf254)
        del arg185_1
        buf255 = reinterpret_tensor(buf254, (32, 512, 3072), (1572864, 3072, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_60], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf255, arg186_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg186_1
        buf256 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf255, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg187_1, (3072, 768), (1, 3072), 0), out=buf256)
        del arg187_1
        buf260 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [add_25, hidden_states_63], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf256, arg188_1, buf253, arg189_1, arg190_1, buf260, 16384, 768, grid=grid(16384), stream=stream0)
        del arg188_1
        del arg189_1
        del arg190_1
        del buf253
        buf261 = reinterpret_tensor(buf247, (16384, 384), (384, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg192_1, reinterpret_tensor(buf260, (16384, 768), (768, 1), 0), reinterpret_tensor(arg191_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf261)
        del arg191_1
        del arg192_1
        buf262 = reinterpret_tensor(buf233, (16384, 384), (384, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg194_1, reinterpret_tensor(buf260, (16384, 768), (768, 1), 0), reinterpret_tensor(arg193_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf262)
        del arg193_1
        del arg194_1
        buf263 = reinterpret_tensor(buf241, (16384, 384), (384, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg196_1, reinterpret_tensor(buf260, (16384, 768), (768, 1), 0), reinterpret_tensor(arg195_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf263)
        del arg195_1
        del arg196_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf264 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf261, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf262, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf263, (32, 6, 512, 64), (196608, 64, 384, 1), 0), None, False, scale=0.125)
        del buf262
        buf265 = buf264[0]
        del buf264
        buf269 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (16384, 768), (768, 1), 0), reinterpret_tensor(arg202_1, (768, 384), (1, 768), 0), out=buf269)
        del arg202_1
        buf270 = reinterpret_tensor(buf256, (32, 768, 512), (393216, 512, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf260, buf270, 24576, 512, grid=grid(24576, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, arg198_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf271, (32, 768, 512), (393216, 512, 1))
        del arg198_1
        # Topologically Sorted Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, arg199_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf272, (32, 384, 512), (196608, 512, 1))
        del arg199_1
        buf273 = reinterpret_tensor(buf261, (32, 512, 384), (196608, 384, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [conv_attn_layer_8, conv_kernel_layer_24], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf273, buf272, arg197_1, 16384, 384, grid=grid(16384, 384), stream=stream0)
        del arg197_1
        del buf272
        buf274 = reinterpret_tensor(buf246, (16384, 54), (54, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (16384, 384), (384, 1), 0), reinterpret_tensor(arg200_1, (384, 54), (1, 384), 0), out=buf274)
        del arg200_1
        buf278 = reinterpret_tensor(buf242, (98304, 9, 1), (9, 1, 884736), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_26], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf274, arg201_1, buf278, 98304, 9, grid=grid(98304), stream=stream0)
        del arg201_1
        buf277 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [conv_out_layer_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf269, arg203_1, buf277, 6291456, 9, grid=grid(6291456, 9), stream=stream0)
        del arg203_1
        buf279 = reinterpret_tensor(buf269, (98304, 64, 1), (64, 1, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_26, conv_out_layer_70], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf277, (98304, 64, 9), (576, 9, 1), 0), buf278, out=buf279)
        buf280 = reinterpret_tensor(buf271, (32, 512, 12, 64), (393216, 768, 64, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [context_layer_34], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf265, buf279, buf280, 12582912, grid=grid(12582912), stream=stream0)
        buf281 = reinterpret_tensor(buf270, (16384, 768), (768, 1), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf280, (16384, 768), (768, 1), 0), reinterpret_tensor(arg204_1, (768, 768), (1, 768), 0), out=buf281)
        del arg204_1
        buf285 = reinterpret_tensor(buf280, (32, 512, 768), (393216, 768, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [add_27, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf281, arg205_1, buf260, arg206_1, arg207_1, buf285, 16384, 768, grid=grid(16384), stream=stream0)
        del arg205_1
        del arg206_1
        del arg207_1
        buf286 = reinterpret_tensor(buf255, (16384, 3072), (3072, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf285, (16384, 768), (768, 1), 0), reinterpret_tensor(arg208_1, (768, 3072), (1, 768), 0), out=buf286)
        del arg208_1
        buf287 = reinterpret_tensor(buf286, (32, 512, 3072), (1572864, 3072, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_68], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf287, arg209_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg209_1
        buf288 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf287, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg210_1, (3072, 768), (1, 3072), 0), out=buf288)
        del arg210_1
        buf292 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [add_28, hidden_states_71], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf288, arg211_1, buf285, arg212_1, arg213_1, buf292, 16384, 768, grid=grid(16384), stream=stream0)
        del arg211_1
        del arg212_1
        del arg213_1
        del buf285
        buf293 = reinterpret_tensor(buf279, (16384, 384), (384, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg215_1, reinterpret_tensor(buf292, (16384, 768), (768, 1), 0), reinterpret_tensor(arg214_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf293)
        del arg214_1
        del arg215_1
        buf294 = reinterpret_tensor(buf265, (16384, 384), (384, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg217_1, reinterpret_tensor(buf292, (16384, 768), (768, 1), 0), reinterpret_tensor(arg216_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf294)
        del arg216_1
        del arg217_1
        buf295 = reinterpret_tensor(buf273, (16384, 384), (384, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg219_1, reinterpret_tensor(buf292, (16384, 768), (768, 1), 0), reinterpret_tensor(arg218_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf295)
        del arg218_1
        del arg219_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf296 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf293, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf294, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf295, (32, 6, 512, 64), (196608, 64, 384, 1), 0), None, False, scale=0.125)
        del buf294
        buf297 = buf296[0]
        del buf296
        buf301 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (16384, 768), (768, 1), 0), reinterpret_tensor(arg225_1, (768, 384), (1, 768), 0), out=buf301)
        del arg225_1
        buf302 = reinterpret_tensor(buf288, (32, 768, 512), (393216, 512, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf292, buf302, 24576, 512, grid=grid(24576, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, arg221_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf303, (32, 768, 512), (393216, 512, 1))
        del arg221_1
        # Topologically Sorted Source Nodes: [x_55], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, arg222_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf304, (32, 384, 512), (196608, 512, 1))
        del arg222_1
        buf305 = reinterpret_tensor(buf293, (32, 512, 384), (196608, 384, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [conv_attn_layer_9, conv_kernel_layer_27], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf305, buf304, arg220_1, 16384, 384, grid=grid(16384, 384), stream=stream0)
        del arg220_1
        del buf304
        buf306 = reinterpret_tensor(buf278, (16384, 54), (54, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (16384, 384), (384, 1), 0), reinterpret_tensor(arg223_1, (384, 54), (1, 384), 0), out=buf306)
        del arg223_1
        buf310 = reinterpret_tensor(buf274, (98304, 9, 1), (9, 1, 884736), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_29], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf306, arg224_1, buf310, 98304, 9, grid=grid(98304), stream=stream0)
        del arg224_1
        buf309 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [conv_out_layer_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf301, arg226_1, buf309, 6291456, 9, grid=grid(6291456, 9), stream=stream0)
        del arg226_1
        buf311 = reinterpret_tensor(buf301, (98304, 64, 1), (64, 1, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_29, conv_out_layer_78], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf309, (98304, 64, 9), (576, 9, 1), 0), buf310, out=buf311)
        buf312 = reinterpret_tensor(buf303, (32, 512, 12, 64), (393216, 768, 64, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [context_layer_38], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf297, buf311, buf312, 12582912, grid=grid(12582912), stream=stream0)
        buf313 = reinterpret_tensor(buf302, (16384, 768), (768, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (16384, 768), (768, 1), 0), reinterpret_tensor(arg227_1, (768, 768), (1, 768), 0), out=buf313)
        del arg227_1
        buf317 = reinterpret_tensor(buf312, (32, 512, 768), (393216, 768, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [add_30, hidden_states_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf313, arg228_1, buf292, arg229_1, arg230_1, buf317, 16384, 768, grid=grid(16384), stream=stream0)
        del arg228_1
        del arg229_1
        del arg230_1
        buf318 = reinterpret_tensor(buf287, (16384, 3072), (3072, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (16384, 768), (768, 1), 0), reinterpret_tensor(arg231_1, (768, 3072), (1, 768), 0), out=buf318)
        del arg231_1
        buf319 = reinterpret_tensor(buf318, (32, 512, 3072), (1572864, 3072, 1), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf319, arg232_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg232_1
        buf320 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf319, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg233_1, (3072, 768), (1, 3072), 0), out=buf320)
        del arg233_1
        buf324 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [add_31, hidden_states_79], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf320, arg234_1, buf317, arg235_1, arg236_1, buf324, 16384, 768, grid=grid(16384), stream=stream0)
        del arg234_1
        del arg235_1
        del arg236_1
        del buf317
        buf325 = reinterpret_tensor(buf311, (16384, 384), (384, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg238_1, reinterpret_tensor(buf324, (16384, 768), (768, 1), 0), reinterpret_tensor(arg237_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf325)
        del arg237_1
        del arg238_1
        buf326 = reinterpret_tensor(buf297, (16384, 384), (384, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg240_1, reinterpret_tensor(buf324, (16384, 768), (768, 1), 0), reinterpret_tensor(arg239_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf326)
        del arg239_1
        del arg240_1
        buf327 = reinterpret_tensor(buf305, (16384, 384), (384, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg242_1, reinterpret_tensor(buf324, (16384, 768), (768, 1), 0), reinterpret_tensor(arg241_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf327)
        del arg241_1
        del arg242_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf328 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf325, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf326, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf327, (32, 6, 512, 64), (196608, 64, 384, 1), 0), None, False, scale=0.125)
        del buf326
        buf329 = buf328[0]
        del buf328
        buf333 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (16384, 768), (768, 1), 0), reinterpret_tensor(arg248_1, (768, 384), (1, 768), 0), out=buf333)
        del arg248_1
        buf334 = reinterpret_tensor(buf320, (32, 768, 512), (393216, 512, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf324, buf334, 24576, 512, grid=grid(24576, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf335 = extern_kernels.convolution(buf334, arg244_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf335, (32, 768, 512), (393216, 512, 1))
        del arg244_1
        # Topologically Sorted Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, arg245_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf336, (32, 384, 512), (196608, 512, 1))
        del arg245_1
        buf337 = reinterpret_tensor(buf325, (32, 512, 384), (196608, 384, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [conv_attn_layer_10, conv_kernel_layer_30], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf337, buf336, arg243_1, 16384, 384, grid=grid(16384, 384), stream=stream0)
        del arg243_1
        del buf336
        buf338 = reinterpret_tensor(buf310, (16384, 54), (54, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (16384, 384), (384, 1), 0), reinterpret_tensor(arg246_1, (384, 54), (1, 384), 0), out=buf338)
        del arg246_1
        buf342 = reinterpret_tensor(buf306, (98304, 9, 1), (9, 1, 884736), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_32], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf338, arg247_1, buf342, 98304, 9, grid=grid(98304), stream=stream0)
        del arg247_1
        buf341 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [conv_out_layer_85], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf333, arg249_1, buf341, 6291456, 9, grid=grid(6291456, 9), stream=stream0)
        del arg249_1
        buf343 = reinterpret_tensor(buf333, (98304, 64, 1), (64, 1, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_32, conv_out_layer_86], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf341, (98304, 64, 9), (576, 9, 1), 0), buf342, out=buf343)
        buf344 = reinterpret_tensor(buf335, (32, 512, 12, 64), (393216, 768, 64, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [context_layer_42], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf329, buf343, buf344, 12582912, grid=grid(12582912), stream=stream0)
        buf345 = reinterpret_tensor(buf334, (16384, 768), (768, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf344, (16384, 768), (768, 1), 0), reinterpret_tensor(arg250_1, (768, 768), (1, 768), 0), out=buf345)
        del arg250_1
        buf349 = reinterpret_tensor(buf344, (32, 512, 768), (393216, 768, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [add_33, hidden_states_82], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf345, arg251_1, buf324, arg252_1, arg253_1, buf349, 16384, 768, grid=grid(16384), stream=stream0)
        del arg251_1
        del arg252_1
        del arg253_1
        buf350 = reinterpret_tensor(buf319, (16384, 3072), (3072, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf349, (16384, 768), (768, 1), 0), reinterpret_tensor(arg254_1, (768, 3072), (1, 768), 0), out=buf350)
        del arg254_1
        buf351 = reinterpret_tensor(buf350, (32, 512, 3072), (1572864, 3072, 1), 0); del buf350  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf351, arg255_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg255_1
        buf352 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf351, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg256_1, (3072, 768), (1, 3072), 0), out=buf352)
        del arg256_1
        buf356 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [add_34, hidden_states_87], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf352, arg257_1, buf349, arg258_1, arg259_1, buf356, 16384, 768, grid=grid(16384), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del buf349
        buf357 = reinterpret_tensor(buf343, (16384, 384), (384, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg261_1, reinterpret_tensor(buf356, (16384, 768), (768, 1), 0), reinterpret_tensor(arg260_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf357)
        del arg260_1
        del arg261_1
        buf358 = reinterpret_tensor(buf329, (16384, 384), (384, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg263_1, reinterpret_tensor(buf356, (16384, 768), (768, 1), 0), reinterpret_tensor(arg262_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf358)
        del arg262_1
        del arg263_1
        buf359 = reinterpret_tensor(buf337, (16384, 384), (384, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg265_1, reinterpret_tensor(buf356, (16384, 768), (768, 1), 0), reinterpret_tensor(arg264_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf359)
        del arg264_1
        del arg265_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf360 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf357, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf358, (32, 6, 512, 64), (196608, 64, 384, 1), 0), reinterpret_tensor(buf359, (32, 6, 512, 64), (196608, 64, 384, 1), 0), None, False, scale=0.125)
        del buf358
        buf361 = buf360[0]
        del buf360
        buf365 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (16384, 768), (768, 1), 0), reinterpret_tensor(arg271_1, (768, 384), (1, 768), 0), out=buf365)
        del arg271_1
        buf366 = reinterpret_tensor(buf352, (32, 768, 512), (393216, 512, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(buf356, buf366, 24576, 512, grid=grid(24576, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf367 = extern_kernels.convolution(buf366, arg267_1, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
        assert_size_stride(buf367, (32, 768, 512), (393216, 512, 1))
        del arg267_1
        # Topologically Sorted Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf367, arg268_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf368, (32, 384, 512), (196608, 512, 1))
        del arg268_1
        buf369 = reinterpret_tensor(buf357, (32, 512, 384), (196608, 384, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [conv_attn_layer_11, conv_kernel_layer_33], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf369, buf368, arg266_1, 16384, 384, grid=grid(16384, 384), stream=stream0)
        del arg266_1
        del buf368
        buf370 = reinterpret_tensor(buf342, (16384, 54), (54, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (16384, 384), (384, 1), 0), reinterpret_tensor(arg269_1, (384, 54), (1, 384), 0), out=buf370)
        del arg269_1
        del buf369
        buf374 = reinterpret_tensor(buf338, (98304, 9, 1), (9, 1, 884736), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_35], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf370, arg270_1, buf374, 98304, 9, grid=grid(98304), stream=stream0)
        del arg270_1
        del buf370
        buf373 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [conv_out_layer_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf365, arg272_1, buf373, 6291456, 9, grid=grid(6291456, 9), stream=stream0)
        del arg272_1
        buf375 = reinterpret_tensor(buf365, (98304, 64, 1), (64, 1, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [conv_kernel_layer_35, conv_out_layer_94], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf373, (98304, 64, 9), (576, 9, 1), 0), buf374, out=buf375)
        del buf373
        del buf374
        buf376 = reinterpret_tensor(buf367, (32, 512, 12, 64), (393216, 768, 64, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [context_layer_46], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf361, buf375, buf376, 12582912, grid=grid(12582912), stream=stream0)
        del buf361
        del buf375
        buf377 = reinterpret_tensor(buf366, (16384, 768), (768, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf376, (16384, 768), (768, 1), 0), reinterpret_tensor(arg273_1, (768, 768), (1, 768), 0), out=buf377)
        del arg273_1
        buf381 = reinterpret_tensor(buf376, (32, 512, 768), (393216, 768, 1), 0); del buf376  # reuse
        # Topologically Sorted Source Nodes: [add_36, hidden_states_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf377, arg274_1, buf356, arg275_1, arg276_1, buf381, 16384, 768, grid=grid(16384), stream=stream0)
        del arg274_1
        del arg275_1
        del arg276_1
        buf382 = reinterpret_tensor(buf351, (16384, 3072), (3072, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf381, (16384, 768), (768, 1), 0), reinterpret_tensor(arg277_1, (768, 3072), (1, 768), 0), out=buf382)
        del arg277_1
        buf383 = reinterpret_tensor(buf382, (32, 512, 3072), (1572864, 3072, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_92], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf383, arg278_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg278_1
        buf384 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf383, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg279_1, (3072, 768), (1, 3072), 0), out=buf384)
        del arg279_1
        del buf383
        buf388 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [add_37, hidden_states_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf384, arg280_1, buf381, arg281_1, arg282_1, buf388, 16384, 768, grid=grid(16384), stream=stream0)
        del arg280_1
        del arg281_1
        del arg282_1
        del buf381
        buf389 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf388, (16384, 768), (768, 1), 0), reinterpret_tensor(arg284_1, (768, 768), (1, 768), 0), out=buf389)
        del arg284_1
        buf393 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_97, hidden_states_98], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_8.run(buf389, arg285_1, arg286_1, arg287_1, buf393, 16384, 768, grid=grid(16384), stream=stream0)
        del arg285_1
        del arg286_1
        del arg287_1
        del buf389
        buf394 = empty_strided_cuda((768, 30524), (30528, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(arg2_1, buf394, 23442432, grid=grid(23442432), stream=stream0)
        del arg2_1
        buf395 = empty_strided_cuda((30524, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(arg288_1, buf395, 30524, grid=grid(30524), stream=stream0)
        del arg288_1
        buf396 = empty_strided_cuda((16384, 30524), (30528, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(buf395, reinterpret_tensor(buf393, (16384, 768), (768, 1), 0), buf394, alpha=1, beta=1, out=buf396)
        del buf393
        del buf394
        del buf395
        buf397 = empty_strided_cuda((16384, 1), (1, 16384), torch.float32)
        buf398 = empty_strided_cuda((16384, 1), (1, 16384), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_11.run(buf396, buf397, buf398, 16384, 30522, grid=grid(16384), stream=stream0)
        buf399 = empty_strided_cuda((2, ), (1, ), torch.float32)
        buf401 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_12.run(arg289_1, buf396, buf397, buf398, buf399, buf401, 2, 8192, grid=grid(2), stream=stream0)
        del arg289_1
        del buf397
        del buf398
        buf400 = empty_strided_cuda((), (), torch.float32)
        buf403 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_13.run(buf403, buf399, buf401, 1, 2, grid=grid(1), stream=stream0)
        del buf399
        del buf401
    return (buf403, reinterpret_tensor(buf396, (32, 512, 30522), (15630336, 30528, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((384, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((768, 1, 9), (9, 9, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((384, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((54, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg284_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('YituTechConvBert', benchmark_compiled_module)
